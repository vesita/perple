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
            strategy: create_strategy(),
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

    /// 从聚类索引同时计算 Box3D 和质心
    fn cluster_box_and_centroid(&self, indices: &[usize], alpha: f32) -> (Box3D, [f32; 3]) {
        let pts: Vec<[f32; 3]> = indices.iter().map(|&idx| self.all_points[idx]).collect();
        let box3d = Box3D::from_cloud_aabb(&pts, 0.0);

        let centroid = if alpha > 0.0 {
            // 密度感知加权：LiDAR 近密远疏，用 1/r^α 补偿质心被拉向传感器的系统偏差
            let eps = 1e-6;
            let mut w_sum = 0.0f32;
            let mut weighted = [0.0f32; 3];
            for p in &pts {
                let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt().max(eps);
                let w = 1.0 / r.powf(alpha);
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
            // 原始算术平均
            [
                pts.iter().map(|p| p[0]).sum::<f32>() / pts.len() as f32,
                pts.iter().map(|p| p[1]).sum::<f32>() / pts.len() as f32,
                pts.iter().map(|p| p[2]).sum::<f32>() / pts.len() as f32,
            ]
        };

        (box3d, centroid)
    }

    /// 输出为 CldBud 向量，过滤掉明显无效的簇
    pub fn to_cldbuds(&self) -> Vec<CldBud> {
        let cfg = crate::config::fixif().cluster.clone();
        let alpha = cfg.density_weight_alpha;
        self.objects
            .iter()
            .filter(|cluster| !cluster.is_empty())
            .enumerate()
            .filter_map(|(idx, cluster)| {
                let (box3d, centroid) = self.cluster_box_and_centroid(cluster, alpha);
                let w = box3d.length.max(box3d.width);
                let h = box3d.height;
                // 排除超小噪点 + 扁度/体积过滤
                if w <= 0.25 || h <= 0.5 { return None; }
                if h < 0.15 * w { return None; }
                if box3d.length * box3d.width * h < 0.03 { return None; }

                Some(CldBud::with_centroid(box3d, 1, format!("cluster_{}", idx), 1.0, centroid))
            })
            .collect()
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

impl Default for Cluster {
    fn default() -> Self {
        Self::new()
    }
}
