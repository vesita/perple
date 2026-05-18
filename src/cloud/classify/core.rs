use std::sync::{Arc, Mutex};

use crate::{
    cloud::{
        CldBud,
        classify::cluster::Cluster,
        classify::strategy::{LvdotClusterStrategy, PruneQt, XYGridDBSCAN},
        ground::{GroundPickStrategy, create_ground_strategy},
        wall::{WallPickStrategy, XYGrid, BevLsd, BevEdLines, BevHough, EdLinesRef},
    },
    color::ClrBud,
    swapl::global_swapl,
    utils::{
        boxes::Box3D,
        stream::{DualBuf, Eap, Stream},
    },
};

fn create_wall_strategy_from_config() -> Box<dyn WallPickStrategy> {
    let cfg = crate::config::fixif();
    match cfg.wall_strategy.as_str() {
        "bev_lsd" => Box::new(BevLsd::with_params(cfg.wall_distance, 20)
            .with_grad_threshold(0.08)
            .with_angle_tolerance(cfg.wall_angle_tolerance)
            .with_min_extent(0.5)),
        "bev_edlines" => Box::new(BevEdLines::with_params(cfg.wall_distance, 20)
            .with_min_extent(0.5)),
        "edlines_ref" => Box::new(EdLinesRef::with_params(cfg.wall_distance, 20)
            .with_min_extent(0.5)),
        "bev_hough" => Box::new(BevHough::with_params(cfg.wall_distance, 20)),
        _ => {
            eprintln!("WARN: 未知墙体策略 '{}'，使用默认 bev_edlines (d={})",
                cfg.wall_strategy, cfg.wall_distance);
            Box::new(BevEdLines::with_params(cfg.wall_distance, 20)
                .with_min_extent(0.5))
        }
    }
}

#[derive(Debug)]
pub enum ClassifyError {
    Error,
}

pub struct Classify {
    in_stream: Eap<Stream<Vec<[f32; 3]>>>,
    cluster: Cluster,
    ground_strategy: Box<dyn GroundPickStrategy>,
    wall_strategy: Box<dyn WallPickStrategy>,
    ground_plane_out: Eap<Stream<[f32; 4]>>,
    /// 双缓冲：检测阶段写 producer（本模块），后融合阶段读 consumer（Tracker）
    clouds_filtered: DualBuf<Vec<[f32; 3]>>,
    /// 专用共享状态：Camera 写入最新 YOLO 结果，本模块读取用于簇分裂
    ///（非 DualBuf，避免与 Camera 跨任务并发竞争）
    last_yolo: Arc<Mutex<Vec<ClrBud>>>,
    /// 双缓冲：检测阶段写 producer（本模块），后融合阶段读 consumer（Fuse）
    cld_buds_raw: DualBuf<Vec<CldBud>>,
    /// 双缓冲：检测阶段写 producer（本模块），后融合阶段读 consumer
    ground_buds_out: DualBuf<Vec<CldBud>>,
    /// 双缓冲：检测阶段写 producer（本模块），后融合阶段读 consumer
    wall_buds_out: DualBuf<Vec<CldBud>>,
}

impl Classify {
    pub fn new() -> Self {
        let swapl = global_swapl();
        Self {
            in_stream: swapl.clouds_out.clone(),
            cluster: Cluster::new(),
            ground_strategy: create_ground_strategy(),
            wall_strategy: create_wall_strategy_from_config(),
            ground_plane_out: swapl.ground_plane.clone(),
            clouds_filtered: swapl.clouds_filtered.clone(),
            last_yolo: swapl.last_yolo.clone(),
            cld_buds_raw: swapl.cld_buds_raw.clone(),
            ground_buds_out: Arc::clone(&swapl.ground_buds),
            wall_buds_out: Arc::clone(&swapl.wall_buds),
        }
    }

    /// 替换墙体提取策略（测试用）
    pub fn with_wall_strategy(mut self, strategy: Box<dyn WallPickStrategy>) -> Self {
        self.wall_strategy = strategy;
        self
    }

    pub async fn act(&mut self) -> Result<(), ClassifyError> {
        let mut target = {
            let mut stream = self.in_stream.lock().unwrap();
            match stream.read() {
               Some(target) => target,
                None => return Ok(()), // 没有数据可处理
            }
        };

        // ─── 1. 地面提取 ──────────────────────────────────────────────────
        let (slice_index, grounds, _ground_plane) = self.ground_strategy.pick(&mut target);
        println!("完成地面提取，已过滤 {} 个点", slice_index);

        // ─── 2. 墙体提取 ──────────────────────────────────────────────────
        let (n_wall, _walls) = self.wall_strategy.pick(&mut target[slice_index..]);
        println!("完成墙壁提取，已过滤 {} 个点", n_wall);

        let remaining_start = slice_index + n_wall;

        // ─── 3a. 体素占用过滤（仅用于 clouds_filtered 跟踪器投票） ────────
        let t4 = std::time::Instant::now();
        let (filtered_pts, _map) = if remaining_start < target.len() {
            XYGrid::voxel_occupancy_filter(&target[remaining_start..], 0.10, 3)
        } else {
            (Vec::new(), Vec::new())
        };
        println!("体素过滤：{} → {} 点 [{:.1}ms]",
            target.len() - remaining_start, filtered_pts.len(),
            t4.elapsed().as_secs_f64() * 1000.0);
        *self.clouds_filtered.producer().lock().unwrap() = filtered_pts;

        // ─── 3b. 后聚类（从配置读取参数） ──────────────────────────────────
        let cfg = crate::config::fixif();
        match cfg.cluster.strategy.as_str() {
            "xy_grid_dbscan" => {
                let cell = cfg.cluster.voxel_size.max(0.05);
                let eps = cfg.cluster.merge_patience;
                let min_pts = cfg.cluster.min_points_per_cluster.unwrap_or(3) as usize;
                let dummy_wall = Box::new(BevLsd::with_params(cfg.wall_distance, 20));
                let pre_extracted = XYGridDBSCAN::with_params(dummy_wall, cell, min_pts, cfg.max_range, eps, min_pts)
                    .with_pre_extracted_wall();
                self.cluster.set_strategy(Box::new(pre_extracted));
            }
            "lvdot_grid" | "lvdot" => {
                let min_pts = cfg.cluster.min_points_per_cluster.unwrap_or(5) as usize;
                self.cluster.set_strategy(Box::new(
                    LvdotClusterStrategy::new()
                        .with_pre_extracted_wall()
                        .with_voxel(cfg.cluster.voxel_size, cfg.cluster.min_occ)
                        .with_dbscan(cfg.cluster.merge_patience, min_pts),
                ));
            }
            "prune_qt" | "lvdot_qt" => {
                self.cluster.set_strategy(Box::new(
                    PruneQt::new().with_pre_extracted_wall(),
                ));
            }
            _ => {}
        }
        let _ = self.cluster.cluster(&target[remaining_start..]);

        // ─── 4. YOLO 辅助簇分裂 ────────────────────────────────────────────
        // 读取 Camera 写入的最新 YOLO 结果（专用 last_yolo 共享状态，
        // 非 DualBuf，避免与 Camera 跨任务并发时读到 swap 后的乱帧数据）
        {
            let guard = self.last_yolo.lock().unwrap();
            if !guard.is_empty() {
                let config = crate::config::fixif();
                let intrinsic = nalgebra::Matrix3::from(config.camera.intrinsic).transpose();
                let cam_from_lidar = nalgebra::Matrix4::from(config.camera.extrinsic).transpose();
                self.cluster.refine_with_yolo(&guard, &intrinsic, &cam_from_lidar);
            }
        }

        // ─── 5. 输出地面/墙体 buds ─────────────────────────────────────────
        *self.ground_buds_out.producer().lock().unwrap() = grounds;
        let wall_buds: Vec<CldBud> = if n_wall > 0 {
            let mut box3d = Box3D::empty_box();
            box3d.cloud2box(&target[slice_index..slice_index + n_wall].to_vec());
            vec![CldBud::new(box3d, 1, "wall".to_string(), 1.0)]
        } else {
            Vec::new()
        };
        *self.wall_buds_out.producer().lock().unwrap() = wall_buds;

        // ─── 6. 输出 cld_buds_raw（仅障碍物簇，不含地面/墙体） ────────────
        // 检测阶段写 DualBuf producer（后融合阶段通过 consumer 读）
        *self.cld_buds_raw.producer().lock().unwrap() = self.cluster.to_cldbuds();
        Ok(())
    }
}
