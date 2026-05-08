use crate::{
    cloud::{
        CldBud,
        classify::claster::Claster,
        ground::{GroundPickStrategy, create_ground_strategy},
        wall::{WallPickStrategy, XYGrid, XYRansacWall},
    },
    color::ClrBud,
    swapl::global_swapl,
    utils::{boxes::Box3D, stream::{Cream, Eap, Stream}},
};

#[derive(Debug)]
pub enum ClassifyError {
    Error,
}

pub struct Classify {
    cream: Cream<Vec<[f32; 3]>, Vec<CldBud>>,
    claster: Claster,
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
            claster: Claster::new(),
            ground_strategy: create_ground_strategy(),
            wall_strategy: Box::new(XYRansacWall::with_params(0.05, 50, 30).with_seed(42)),
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

        // ─── 2. 墙体提取（从非地面点） ────────────────────────────────────
        let n_non_ground = target.len() - n_ground;
        let n_wall = if n_non_ground > 0 {
            let (n, _planes) = self.wall_strategy.pick(&mut target[n_ground..]);
            if n > 0 {
                println!("墙体提取：{} 墙体点 / {} 剩余", n, n_non_ground - n);
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

        // ─── 3. LV-DOT 体素占用过滤 ──────────────────────────────────────
        let remaining_start = n_ground + n_wall;
        let (filtered_pts, _map) = if remaining_start < target.len() {
            XYGrid::voxel_occupancy_filter(&target[remaining_start..], 0.10, 3)
        } else {
            (Vec::new(), Vec::new())
        };
        println!("体素过滤：{} → {} 点", target.len() - remaining_start, filtered_pts.len());

        // ─── 4. 写入 clouds_filtered（干净非地面非墙体点，供跟踪器投票使用） ─
        {
            let mut cf = self.clouds_filtered_out.lock().await;
            let _ = cf.write(filtered_pts.clone());
        }

        // ─── 5. 聚类输入 ────────────────────────────────────────────────────
        let config = crate::config::fixif();
        let cluster_input = match config.claster.strategy.as_str() {
            "wall_cluster" | "lvdot" => {
                // 全管线策略传入全部非地面点（策略内部处理墙体+过滤）
                target[n_ground..].to_vec()
            }
            _ => {
                let mut pts = filtered_pts;
                if config.claster.ceiling_filter && config.claster.ceiling_height > 0.0 {
                    let h = config.claster.ceiling_height;
                    pts.retain(|p| p[2] <= h);
                }
                if config.claster.max_range > 0.0 {
                    let max_d2 = config.claster.max_range * config.claster.max_range;
                    pts.retain(|p| p[0] * p[0] + p[1] * p[1] + p[2] * p[2] <= max_d2);
                }
                pts
            }
        };

        // ─── 6. 聚类 ──────────────────────────────────────────────────────
        let _ = self.claster.claster(&cluster_input);

        // ─── 7. YOLO 辅助簇分裂 ───────────────────────────────────────────
        {
            let clr_data = self.clr_objs_in.lock().await;
            let clr_buds = clr_data.peek_latest();
            if let Some(ref buds) = clr_buds {
                if !buds.is_empty() {
                    let config = crate::config::fixif();
                    let intrinsic = nalgebra::Matrix3::from(config.camera.intrinsic);
                    let cam_from_lidar = nalgebra::Matrix4::from(config.camera.extrinsic);
                    self.claster.refine_with_yolo(buds, &intrinsic, &cam_from_lidar);
                }
            }
        }

        // ─── 8. 输出 cld_buds_raw（仅障碍物簇，不含地面/墙体） ──────────
        let all_buds = self.claster.to_cldbuds();
        {
            let mut stream = self.cream.out_stream.lock().await;
            let _ = stream.write(all_buds);
        }
        Ok(())
    }
}
