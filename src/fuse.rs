use std::collections::HashMap;

use nalgebra::{Matrix3, Matrix4, Vector4};

use crate::config::fixif;
use crate::swapl::global_swapl;
use crate::cloud::output::CldBud;
use crate::color::output::ClrBud;
use crate::utils::boxes::{Box2D, Box3D};

/// 2D→3D 语义融合模块
///
/// 将 Camera 的 2D 检测标签匹配到 Lidar 的 3D 检测上。
/// 执行保守融合：同 YOLO 目标的多 3D 簇合并；逐 3D 簇标签精炼。
/// 在 Tracker 之前运行，使 Tracker 消费已融合的检测结果。
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
    ///
    /// 步骤：
    /// 1. 投影：每个 3D 簇 → 2D 图像框
    /// 2. 匹配：计算 2D IoU 与 YOLO 检测框
    /// 3. 合并：匹配同 YOLO 框的多 3D 簇 → 保守并集
    /// 4. 标记：更新类别标签
    pub fn act(&mut self) {
        let swapl = global_swapl();

        let clr_buds: Vec<ClrBud> = match swapl.clr_objs.blocking_lock().get_at(0) {
            Some(buds) => buds,
            None => return,
        };

        let cld_buds: Vec<CldBud> = match swapl.cld_objs.blocking_lock().get_at(0) {
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

        // Step 1: 投影每个 3D 簇 → 2D box, 同时记录最佳匹配
        struct ProjMatch {
            proj_box: Box2D,
            clr_idx: usize,
            iou: f32,
        }

        let mut proj: Vec<Option<ProjMatch>> = Vec::with_capacity(cld_buds.len());

        for cld in &cld_buds {
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
                proj.push(None);
                continue;
            }

            let proj_box = Box2D::new(l, t, r, b);
            if !proj_box.is_valid() {
                proj.push(None);
                continue;
            }

            let mut best_iou = 0.2;
            let mut best_idx = usize::MAX;
            for (ci, clr) in clr_buds.iter().enumerate() {
                let iou = proj_box.iou(&clr.the_box);
                if iou > best_iou {
                    best_iou = iou;
                    best_idx = ci;
                }
            }

            if best_idx != usize::MAX {
                proj.push(Some(ProjMatch { proj_box, clr_idx: best_idx, iou: best_iou }));
            } else {
                proj.push(None);
            }
        }

        // Step 2: 构建 clr_idx → Vec<3D 簇索引> 映射（用于合并）
        let mut clr_to_cld: HashMap<usize, Vec<usize>> = HashMap::new();
        for (ci, pm) in proj.iter().enumerate() {
            if let Some(m) = pm {
                clr_to_cld.entry(m.clr_idx).or_default().push(ci);
            }
        }

        // Step 3: 合并 — 多个 3D 簇匹配同一个 YOLO 框
        let mut merged_mask = vec![false; cld_buds.len()];
        let mut merged_buds: Vec<CldBud> = Vec::new();

        for (clr_idx, cld_indices) in &clr_to_cld {
            if cld_indices.len() <= 1 {
                continue;
            }
            // 保守并集：取这些簇的 XYZ 最值
            let (mut min_x, mut min_y, mut min_z) = (f32::MAX, f32::MAX, f32::MAX);
            let (mut max_x, mut max_y, mut max_z) = (f32::MIN, f32::MIN, f32::MIN);
            let mut total_conf = 0.0;
            let class_name = clr_buds[*clr_idx].class_name.clone();

            for &ci in cld_indices {
                merged_mask[ci] = true;
                let b = &cld_buds[ci].the_box;
                // 用 8 个顶点获取极值
                for v in b.vertices() {
                    min_x = min_x.min(v.x);
                    min_y = min_y.min(v.y);
                    min_z = min_z.min(v.z);
                    max_x = max_x.max(v.x);
                    max_y = max_y.max(v.y);
                    max_z = max_z.max(v.z);
                }
                total_conf += cld_buds[ci].confidence;
            }

            let mut merged_box = Box3D::empty_box();
            merged_box.cloud2box(&vec![
                [min_x, min_y, min_z],
                [max_x, max_y, max_z],
            ]);

            // 质心按置信度加权平均
            let mut w_centroid = [0.0f32; 3];
            let mut w_sum = 0.0f32;
            for &ci in cld_indices {
                let w = cld_buds[ci].confidence;
                w_centroid[0] += cld_buds[ci].centroid[0] * w;
                w_centroid[1] += cld_buds[ci].centroid[1] * w;
                w_centroid[2] += cld_buds[ci].centroid[2] * w;
                w_sum += w;
            }
            if w_sum > 0.0 {
                w_centroid[0] /= w_sum;
                w_centroid[1] /= w_sum;
                w_centroid[2] /= w_sum;
            }

            merged_buds.push(CldBud {
                the_box: merged_box,
                class_id: clr_buds[*clr_idx].class_id,
                class_name,
                confidence: total_conf / cld_indices.len() as f32,
                centroid: w_centroid,
            });
        }

        // Step 4: 保留未合并的 + 标注类别
        let merged_count = merged_buds.len();
        if merged_count > 0 {
            log::info!("Fuse 合并了 {} 组 3D 簇", merged_count);
        }

        let mut result: Vec<CldBud> = Vec::new();

        // 4a: 推送合并结果
        result.extend(merged_buds);

        // 4b: 未合并的，更新标签
        for (ci, cld) in cld_buds.into_iter().enumerate() {
            if merged_mask[ci] {
                continue;
            }
            let mut cld = cld;
            if let Some(ref pm) = proj[ci] {
                cld.class_name = clr_buds[pm.clr_idx].class_name.clone();
                cld.class_id = clr_buds[pm.clr_idx].class_id;
                cld.confidence = clr_buds[pm.clr_idx].confidence;
            }
            result.push(cld);
        }

        let _ = swapl.cld_objs.blocking_lock().write(result);
    }
}
