use nalgebra::{Matrix3, Matrix4, Vector4};

use crate::config::fixif;
use crate::swapl::global_swapl;
use crate::cloud::output::CldBud;
use crate::color::output::ClrBud;
use crate::utils::boxes::Box2D;

/// 2D→3D 语义融合模块
///
/// 将 Camera 的 2D 检测标签匹配到 Lidar 的 3D 检测上。
/// 使用固定标定矩阵 cam_from_lidar 直接投影，不经过世界帧。
pub struct Fuse {
    intrinsic: Matrix3<f32>,
    cam_from_lidar: Matrix4<f32>,
}

impl Fuse {
    pub fn new() -> Self {
        let config = fixif();
        // camera.extrinsic = lidar→camera 变换矩阵 [R|t]
        // 与标准针孔模型定义一致：P_cam = extrinsic * P_lidar
        let cam_from_lidar = Matrix4::from(config.camera.extrinsic);
        Self {
            intrinsic: Matrix3::from(config.camera.intrinsic),
            cam_from_lidar,
        }
    }

    /// 执行一次融合（blocking，用于 MultiLoop）
    pub fn act(&mut self) {
        let swapl = global_swapl();

        let clr_buds: Vec<ClrBud> = match swapl.clr_objs.blocking_lock().get_at(0) {
            Some(buds) => buds,
            None => return,
        };

        let mut cld_buds: Vec<CldBud> = match swapl.cld_objs.blocking_lock().get_at(0) {
            Some(buds) => buds,
            None => return,
        };

        if clr_buds.is_empty() || cld_buds.is_empty() {
            return;
        }

        let fx = self.intrinsic[(0, 0)];
        let fy = self.intrinsic[(1, 1)];
        let cx = self.intrinsic[(0, 2)];
        let cy = self.intrinsic[(1, 2)];

        for cld in cld_buds.iter_mut() {
            let verts = cld.the_box.vertices();

            let (mut l, mut t, mut r, mut b) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
            for v in &verts {
                let cam = self.cam_from_lidar * Vector4::new(v.x, v.y, v.z, 1.0);
                if cam.z <= 0.0 {
                    continue;
                }
                let u = fx * cam.x / cam.z + cx;
                let v_ = fy * cam.y / cam.z + cy;
                l = l.min(u);
                t = t.min(v_);
                r = r.max(u);
                b = b.max(v_);
            }
            if l == f32::MAX {
                continue;
            }

            let proj_box = Box2D::new(l, t, r, b);
            if !proj_box.is_valid() {
                continue;
            }

            let mut best_iou = 0.2;
            for clr in &clr_buds {
                let iou = proj_box.iou(&clr.the_box);
                if iou > best_iou {
                    best_iou = iou;
                    cld.class_name = clr.class_name.clone();
                    cld.class_id = clr.class_id;
                    cld.confidence = clr.confidence;
                }
            }
        }

        let _ = swapl.cld_objs.blocking_lock().write(cld_buds);
    }
}
