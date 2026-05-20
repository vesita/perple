use std::collections::HashMap;

use nalgebra::{Matrix3, Matrix4, Vector4};

use crate::{cloud::CldBud, color::ClrBud, utils::Box3D};

use super::strategy::{ClusteringStrategy, create_strategy};

/// 聚类器 — 不依赖具体策略，通过策略 trait 扩展
pub struct Cluster {
    objects: Vec<Vec<usize>>,  // 每个簇的点索引
    all_points: Vec<[f32; 3]>, // 聚类实际使用的点集
    strategy: Box<dyn ClusteringStrategy>,
}

impl Cluster {
    pub fn new() -> Self {
        Cluster {
            objects: Vec::new(),
            all_points: Vec::new(),
            strategy: create_strategy(false),
        }
    }

    /// 清空结果
    pub fn clear(&mut self) {
        self.objects.clear();
        self.all_points.clear();
    }

    /// 聚类结果引用
    pub fn objects(&self) -> &Vec<Vec<usize>> {
        &self.objects
    }

    /// 替换聚类策略（管线已预处理时使用）
    pub fn set_strategy(&mut self, strategy: Box<dyn ClusteringStrategy>) {
        self.strategy = strategy;
    }

    /// 主入口：对一帧点云执行聚类
    pub fn cluster(&mut self, lifra: &[[f32; 3]]) {
        println!("开始聚类，共 {} 个点", lifra.len());
        self.clear();

        let (points, objects) = self.strategy.run(lifra);
        self.all_points = points;
        self.objects = objects;
    }

    /// 单遍扫描计算 Box3D AABB 和质心，避免中间 Vec 分配
    fn cluster_box_and_centroid(all_points: &[[f32; 3]], indices: &[usize], alpha: f32) -> (Box3D, [f32; 3]) {
        let n = indices.len();
        if n == 0 {
            return (Box3D::empty_box(), [0.0; 3]);
        }

        // 单遍扫描：同时累积 AABB 边界和质心
        let mut x_min = f32::MAX;
        let mut x_max = f32::MIN;
        let mut y_min = f32::MAX;
        let mut y_max = f32::MIN;
        let mut z_min = f32::MAX;
        let mut z_max = f32::MIN;

        let centroid = if alpha > 0.0 {
            let eps = 1e-6;
            let mut w_sum = 0.0f32;
            let mut weighted = [0.0f32; 3];
            for &idx in indices {
                let p = &all_points[idx];
                x_min = x_min.min(p[0]); x_max = x_max.max(p[0]);
                y_min = y_min.min(p[1]); y_max = y_max.max(p[1]);
                z_min = z_min.min(p[2]); z_max = z_max.max(p[2]);
                let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt().max(eps);
                let w = r.powf(alpha);
                weighted[0] += p[0] * w;
                weighted[1] += p[1] * w;
                weighted[2] += p[2] * w;
                w_sum += w;
            }
            if w_sum > 0.0 {
                weighted[0] /= w_sum;
                weighted[1] /= w_sum;
                weighted[2] /= w_sum;
            }
            weighted
        } else {
            let mut sum = [0.0f32; 3];
            for &idx in indices {
                let p = &all_points[idx];
                x_min = x_min.min(p[0]); x_max = x_max.max(p[0]);
                y_min = y_min.min(p[1]); y_max = y_max.max(p[1]);
                z_min = z_min.min(p[2]); z_max = z_max.max(p[2]);
                sum[0] += p[0];
                sum[1] += p[1];
                sum[2] += p[2];
            }
            [sum[0] / n as f32, sum[1] / n as f32, sum[2] / n as f32]
        };

        let cx = (x_min + x_max) * 0.5;
        let cy = (y_min + y_max) * 0.5;
        let cz = (z_min + z_max) * 0.5;
        let box3d = Box3D {
            pose: nalgebra::Matrix4::new_translation(&nalgebra::Vector3::new(cx, cy, cz)),
            length: (x_max - x_min).max(0.0),
            width:  (y_max - y_min).max(0.0),
            height: (z_max - z_min).max(0.0),
        };
        (box3d, centroid)
    }

    /// 输出为 CldBud 向量，过滤掉明显无效的簇
    pub fn to_cldbuds(&self) -> Vec<CldBud> {
        clusters_to_cldbuds(&self.all_points, &self.objects)
    }

    /// YOLO 辅助簇分裂（Phase 2）
    ///
    /// 对每个簇，将簇内点投影到 2D，按落在哪个 YOLO 框分组。
    /// 若一个簇的点映射到多个 YOLO 框，则分裂为多个子簇。
    /// 不匹配任何 YOLO 框的点保留为独立簇。
    pub fn refine_with_yolo(&mut self, clr_buds: &[ClrBud], intrinsic: &Matrix3<f32>, cam_from_lidar: &Matrix4<f32>) {
        if self.objects.is_empty() || clr_buds.is_empty() {
            return;
        }

        let fx = intrinsic[(0, 0)];
        let fy = intrinsic[(1, 1)];
        let cx = intrinsic[(0, 2)];
        let cy = intrinsic[(1, 2)];

        let mut new_objects: Vec<Vec<usize>> = Vec::with_capacity(self.objects.len());

        for cluster_indices in &self.objects {
            if cluster_indices.len() < 3 {
                new_objects.push(cluster_indices.clone());
                continue;
            }

            // 跳过相机平面后方 (Z_cam < 1) 的簇
            let centroid = Cluster::cluster_box_and_centroid(
                &self.all_points, cluster_indices, 0.0,
            ).1;
            let cam_c = cam_from_lidar * Vector4::new(
                centroid[0], centroid[1], centroid[2], 1.0,
            );
            if cam_c.z < 1.0 {
                new_objects.push(cluster_indices.clone());
                continue;
            }

            // 将每个点投影到 2D，找到命中的 YOLO 框
            let mut point_box_assignment: Vec<Option<usize>> = Vec::with_capacity(cluster_indices.len());
            let mut box_point_counts: HashMap<usize, usize> = HashMap::new();

            for &pt_idx in cluster_indices {
                let p = self.all_points[pt_idx];
                let cam = cam_from_lidar * Vector4::new(p[0], p[1], p[2], 1.0);
                if cam.z <= 0.0 {
                    point_box_assignment.push(None);
                    continue;
                }
                let u = fx * cam.x / cam.z + cx;
                let v = fy * cam.y / cam.z + cy;

                // 查哪个 YOLO 框包含该投影点
                let mut matched = None;
                for (bi, clr) in clr_buds.iter().enumerate() {
                    if u >= clr.the_box.x1 && u <= clr.the_box.x2
                        && v >= clr.the_box.y1 && v <= clr.the_box.y2
                    {
                        matched = Some(bi);
                        break;
                    }
                }
                point_box_assignment.push(matched);
                if let Some(bi) = matched {
                    *box_point_counts.entry(bi).or_insert(0) += 1;
                }
            }

            if box_point_counts.len() <= 1 {
                // 全部匹配同 1 个框或都不匹配 → 保持原簇
                new_objects.push(cluster_indices.clone());
                continue;
            }

            // 分裂：按 YOLO 框分组
            let mut box_groups: HashMap<usize, Vec<usize>> = HashMap::new();
            let mut no_match_group: Vec<usize> = Vec::new();
            let min_pts = 3usize;

            for (i, &pt_idx) in cluster_indices.iter().enumerate() {
                match point_box_assignment[i] {
                    Some(bi) => box_groups.entry(bi).or_default().push(pt_idx),
                    None => no_match_group.push(pt_idx),
                }
            }

            if no_match_group.len() >= min_pts {
                new_objects.push(no_match_group);
            }

            for (_, group) in box_groups {
                if group.len() >= min_pts {
                    new_objects.push(group);
                }
            }
        }

        if new_objects.len() > self.objects.len() {
            log::info!("refine_with_yolo: 分裂 {} → {} 个簇", self.objects.len(), new_objects.len());
        }
        self.objects = new_objects;
    }

    #[allow(unused)]
    pub fn add_box3d(&mut self, _box3d: Box3D) {}
}

/// 将聚类索引结果转为 CldBud 向量（供 bench 等外部复用）。
///
/// 过滤逻辑与 `Cluster::to_cldbuds()` 完全一致。
pub fn clusters_to_cldbuds(all_points: &[[f32; 3]], objects: &[Vec<usize>]) -> Vec<CldBud> {
    let cluster_cfg = &crate::config::fixif().cluster;
    let alpha = cluster_cfg.density_weight_alpha;
    objects
        .iter()
        .filter(|c| !c.is_empty())
        .enumerate()
        .filter_map(|(idx, cluster)| {
            let (box3d, centroid) = Cluster::cluster_box_and_centroid(all_points, cluster, alpha);
            let w = box3d.length.max(box3d.width);
            let h = box3d.height;
            // 排除超小噪点 + 扁度/体积过滤
            if w <= 0.2 || h <= 0.3 { return None; }
            if h < 0.15 * w { return None; }
            if box3d.length * box3d.width * h < 0.03 { return None; }

            // box 过大过滤（室内场景物体不应过大）
            if w > 3.0 { return None; }
            // 盒子中心过低 → 地面残留噪点（用 AABB 中心 Z，不受密度加权偏移影响）
            if box3d.center().z < 0.2 { return None; }

            // 边界过滤：盒子超出有效检测范围时丢弃（已禁用，用于对比实验）
            // let c = box3d.center();
            // let center_dist = (c.x * c.x + c.y * c.y).sqrt();
            // let half_diag = (box3d.length * box3d.length + box3d.width * box3d.width).sqrt() * 0.5;
            // if center_dist + half_diag > max_r { return None; }

            // 点云稀疏度过滤：大体积内点数过少 → 离群噪点
            let n_pts = cluster.len() as f32;
            let volume = box3d.length * box3d.width * h;
            if volume > 0.5 && n_pts / volume < 20.0 { return None; }

            Some(CldBud::with_centroid(box3d, 1, format!("cluster_{}", idx), 1.0, centroid))
        })
        .collect()
}

impl Default for Cluster {
    fn default() -> Self {
        Self::new()
    }
}
