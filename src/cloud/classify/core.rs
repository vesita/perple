use crate::{
    cloud::{
        CldBud,
        classify::cluster::Cluster,
        classify::strategy::XYGridDBSCAN,
        denoise::{DenoiseStrategy, RadiusOutlierRemoval},
        ground::{GroundPickStrategy, create_ground_strategy},
        wall::{
            WallPickStrategy, XYGrid, XYRansacWall,
            TopDownCluster, QuadtreeWall, NormalWall, XYDBSCANWall,
            AdaptiveDBSCANWall, SequentialFit,
        },
    },
    color::ClrBud,
    swapl::global_swapl,
    utils::{boxes::Box3D, stream::{Cream, Eap, Stream}},
};

fn create_wall_strategy_from_config() -> Box<dyn WallPickStrategy> {
    let cfg = crate::config::fixif();
    match cfg.wall_strategy.as_str() {
        "xy_ransac" => Box::new(XYRansacWall::with_params(cfg.wall_distance, cfg.wall_iterations, 30).with_seed(42)),
        "top_down" => Box::new(TopDownCluster::with_params(0.30, 10, 2)
            .with_width_ratio(5.0)),
        "quadtree" => Box::new(QuadtreeWall::with_params(0.30, 5, 0.5)
            .with_width_ratio(5.0)),
        "normal_wall" => Box::new(NormalWall::with_params(0.20, 10, 30.0)
            .with_normal_threshold(0.3)),
        "xy_dbscan_wall" => Box::new(XYDBSCANWall::with_params(cfg.wall_eps, cfg.wall_min_pts, cfg.wall_min_z_span)),
        "adaptive_dbscan" => Box::new(AdaptiveDBSCANWall::with_params(0.15, 2.0, 10)),
        "seq_fit" => Box::new(SequentialFit::with_params(0.10, 0.3, 5)),
        _ => Box::new(XYRansacWall::with_params(0.05, 50, 30).with_seed(42)),
    }
}

#[derive(Debug)]
pub enum ClassifyError {
    Error,
}

pub struct Classify {
    cream: Cream<Vec<[f32; 3]>, Vec<CldBud>>,
    cluster: Cluster,
    ground_strategy: Box<dyn GroundPickStrategy>,
    wall_strategy: Box<dyn WallPickStrategy>,
    ground_plane_out: Eap<Stream<[f32; 4]>>,
    clouds_filtered_out: Eap<Stream<Vec<[f32; 3]>>>,
    clr_objs_in: Eap<Stream<Vec<ClrBud>>>,
    ground_buds_out: Eap<Stream<Vec<CldBud>>>,
    wall_buds_out: Eap<Stream<Vec<CldBud>>>,
}

impl Classify {
    pub fn new() -> Self {
        let swapl = global_swapl();
        Self {
            cream: Cream {
                in_stream: swapl.clouds_out.clone(),
                out_stream: swapl.cld_buds_raw.clone(),
            },
            cluster: Cluster::new(),
            ground_strategy: create_ground_strategy(),
            wall_strategy: create_wall_strategy_from_config(),
            ground_plane_out: swapl.ground_plane.clone(),
            clouds_filtered_out: swapl.clouds_filtered.clone(),
            clr_objs_in: swapl.clr_objs.clone(),
            ground_buds_out: swapl.ground_buds.clone(),
            wall_buds_out: swapl.wall_buds.clone(),
        }
    }

    /// 替换墙体提取策略（测试用）
    pub fn with_wall_strategy(mut self, strategy: Box<dyn WallPickStrategy>) -> Self {
        self.wall_strategy = strategy;
        self
    }

    pub async fn act(&mut self) -> Result<(), ClassifyError> {
        let mut target = {
            let mut stream = self.cream.in_stream.lock().await;
            match stream.read() {
                Some(target) => target,
                None => return Ok(()),
            }
        };

        // ─── 1. 地面提取 ──────────────────────────────────────────────────
        let (slice_index, grounds, plane_eq) = self.ground_strategy.pick(&mut target);
        let n_ground = slice_index;
        println!("地面提取：{} 地面点 / {} 非地面点", n_ground, target.len() - n_ground);
        if let Some(eq) = plane_eq {
            let mut gp = self.ground_plane_out.lock().await;
            let _ = gp.write(eq);
        }

        // 存储 ground buds
        {
            let mut gb = self.ground_buds_out.lock().await;
            let _ = gb.write(grounds);
        }

        // ─── 2. 墙体提取前等步长降采样：15000→~2000，均匀保留空间分布 ───
        let n_non_ground = target.len() - n_ground;
        if n_non_ground > 3000 {
            let before = n_non_ground;
            let step = before / 2000;
            let sampled: Vec<[f32; 3]> = target.drain(n_ground..)
                .step_by(step.max(1))
                .collect();
            target.extend(sampled);
            println!("墙体输入降采样：{} → {} 点", before, target.len() - n_ground);
        }

        // ─── 2. 墙体提取 ──────────────────────────────────────────────────
        let n_wall = if target.len() > n_ground {
            let (n, _planes) = self.wall_strategy.pick(&mut target[n_ground..]);
            if n > 0 {
                println!("墙体提取：{} 墙体点 / {} 剩余", n, target.len() - n_ground - n);
                let wall_pts: Vec<[f32; 3]> = target[n_ground..n_ground + n].to_vec();
                let wall_box = Box3D::from_cloud_aabb(&wall_pts, 0.05);
                let wall_cld = CldBud::new(wall_box, 2, "wall".into(), 1.0);
                let mut wb = self.wall_buds_out.lock().await;
                let _ = wb.write(vec![wall_cld]);
            }
            n
        } else {
            0
        };

        // ─── 2.5 后处理降噪（聚类前清洁非墙面点） ────────────────────────
        let remaining_start = n_ground + n_wall;
        let mut post_denoise = RadiusOutlierRemoval::new(0.20, 3);
        let (denoised_nw, _) = if remaining_start < target.len() {
            post_denoise.denoise(&target[remaining_start..])
        } else {
            (Vec::new(), Vec::new())
        };
        println!("后处理降噪：{} → {} 非墙面点",
            target.len() - remaining_start, denoised_nw.len());

        // ─── 3. 体素占用过滤（仅用于 clouds_filtered 跟踪器投票） ──────────
        let t4 = std::time::Instant::now();
        let (filtered_pts, _map) = if !denoised_nw.is_empty() {
            XYGrid::voxel_occupancy_filter(&denoised_nw, 0.10, 3)
        } else {
            (Vec::new(), Vec::new())
        };
        println!("体素过滤：{} → {} 点 [{:.1}ms]",
            denoised_nw.len(), filtered_pts.len(),
            t4.elapsed().as_secs_f64() * 1000.0);
        {
            let mut cf = self.clouds_filtered_out.lock().await;
            let _ = cf.write(filtered_pts);
        }

        // ─── 4. 聚类输入：如非墙面点过多则按距离截断（仅极端情况） ──────
        let mut denoised_nw = denoised_nw;
        let max_cluster_input = 3000;
        if denoised_nw.len() > max_cluster_input {
            let before = denoised_nw.len();
            denoised_nw.sort_by(|a, b| {
                let da = a[0] * a[0] + a[1] * a[1] + a[2] * a[2];
                let db = b[0] * b[0] + b[1] * b[1] + b[2] * b[2];
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            });
            denoised_nw.truncate(max_cluster_input);
            println!("聚类输入上限：{} → {} 点", before, denoised_nw.len());
        }

        // ─── 5. 聚类 ──────────────────────────────────────────────────────
        let config = crate::config::fixif();
        let cluster_input = match config.cluster.strategy.as_str() {
            "xy_grid_dbscan" => {
                let dummy_wall = Box::new(XYRansacWall::with_params(0.05, 50, 30));
                let pre_extracted = XYGridDBSCAN::with_params(dummy_wall, 0.30, 3, 12.0, 0.15, 3)
                    .with_pre_extracted_wall();
                self.cluster.set_strategy(Box::new(pre_extracted));
                denoised_nw
            }
            "lvdot" => denoised_nw,
            _ => {
                denoised_nw
            }
        };

        // ─── 6. 聚类 ──────────────────────────────────────────────────────
        let _ = self.cluster.cluster(&cluster_input);

        // ─── 7. YOLO 辅助簇分裂 ───────────────────────────────────────────
        {
            let clr_data = self.clr_objs_in.lock().await;
            let clr_buds = clr_data.peek_latest();
            if let Some(ref buds) = clr_buds {
                if !buds.is_empty() {
                    let config = crate::config::fixif();
                    let intrinsic = nalgebra::Matrix3::from(config.camera.intrinsic);
                    let cam_from_lidar = nalgebra::Matrix4::from(config.camera.extrinsic);
                    self.cluster.refine_with_yolo(buds, &intrinsic, &cam_from_lidar);
                }
            }
        }

        // ─── 8. 输出 cld_buds_raw（仅障碍物簇，不含地面/墙体） ──────────
        let all_buds = self.cluster.to_cldbuds();
        {
            let mut stream = self.cream.out_stream.lock().await;
            let _ = stream.write(all_buds);
        }
        Ok(())
    }
}
