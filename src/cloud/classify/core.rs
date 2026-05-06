use crate::{
    cloud::{
        CldBud,
        classify::claster::Claster,
        ground::{GroundPickStrategy, create_ground_strategy},
    },
    color::ClrBud,
    swapl::global_swapl,
    utils::stream::{Cream, Eap, Stream},
};

#[derive(Debug)]
pub enum ClassifyError {
    Error,
}

pub struct Classify {
    cream: Cream<Vec<[f32; 3]>, Vec<CldBud>>,
    claster: Claster,
    ground_strategy: Box<dyn GroundPickStrategy>,
    ground_plane_out: Eap<Stream<[f32; 4]>>,
    clouds_filtered_out: Eap<Stream<Vec<[f32; 3]>>>,
    clr_objs_in: Eap<Stream<Vec<ClrBud>>>,
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
            ground_plane_out: swapl.ground_plane.clone(),
            clouds_filtered_out: swapl.clouds_filtered.clone(),
            clr_objs_in: swapl.clr_objs.clone(),
        }
    }

    pub async fn act(&mut self) -> Result<(), ClassifyError> {
        let mut target = {
            let mut stream = self.cream.in_stream.lock().await;
            match stream.read() {
                Some(target) => target,
                None => return Ok(()), // 没有数据可处理
            }
        };

        let (slice_index, grounds, plane_eq) = self.ground_strategy.pick(&mut target);
        println!("完成地面提取，已过滤{}个点", slice_index);
        if let Some(eq) = plane_eq {
            let mut gp = self.ground_plane_out.lock().await;
            let _ = gp.write(eq);
        }

        // 将地面滤除后的点云写入 Swapl（供跟踪器点云投票使用）
        {
            let mut cf = self.clouds_filtered_out.lock().await;
            let _ = cf.write(target[slice_index..].to_vec());
        }

        // 室内天花板过滤：剔除天花板附近的点，减少无效簇
        let config = crate::config::fixif();
        if config.claster.ceiling_filter && config.claster.ceiling_height > 0.0 {
            let h = config.claster.ceiling_height;
            let before = target.len();
            target.retain(|p| p[2] <= h);
            println!("天花板过滤：{} → {} 点（≤ {:.1}m）", before, target.len(), h);
        }

        // 按距离过滤非地面点：剔除远处不可靠点
        let max_range = crate::config::fixif().claster.max_range;
        if max_range > 0.0 {
            let max_d2 = max_range * max_range;
            let filtered: Vec<[f32; 3]> = target[slice_index..]
                .iter()
                .copied()
                .filter(|p| p[0]*p[0] + p[1]*p[1] + p[2]*p[2] <= max_d2)
                .collect();
            let _ = self.claster.claster(&filtered);
        } else {
            let _ = self.claster.claster(&target[slice_index..]);
        }

        // Phase 2: YOLO 辅助簇分裂（点级投影 + 按 YOLO 框分配）
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

        let mut all_buds = self.claster.to_cldbuds();

        // 地面保留在聚类结果中（作为 class_id=0 的 CldBud）
        all_buds.extend(grounds);

        {
            let mut stream = self.cream.out_stream.lock().await;
            let _ = stream.write(all_buds);
        }
        Ok(())
    }
}
