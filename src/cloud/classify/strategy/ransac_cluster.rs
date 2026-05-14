use super::ClusteringStrategy;
use crate::cloud::wall::{XYGrid, best_xy_line};

/// RANSAC 线检测聚类策略（ransac）。
///
/// 每轮在 XY 平面执行 RANSAC 线检测 → 线内点作为一个簇 → 移除 → 重复。
/// 从墙体提取 `ransac_l2_grid` 改编，去掉了墙体专用过滤（Z 跨度、min_extent）。
///
/// 可选预降噪：若 `denoise_radius > 0`，先执行半径离群点剔除。
pub struct RansacCluster {
    distance: f32,
    iterations: usize,
    min_points: usize,
    max_clusters: usize,
    denoise_radius: f32,
    denoise_min_pts: usize,
}

impl RansacCluster {
    pub fn new(distance: f32, iterations: usize, min_points: usize) -> Self {
        Self {
            distance,
            iterations,
            min_points,
            max_clusters: 20,
            denoise_radius: 0.0,
            denoise_min_pts: 3,
        }
    }

    pub fn with_max_clusters(mut self, max: usize) -> Self {
        self.max_clusters = max;
        self
    }

    pub fn with_denoise(mut self, radius: f32, min_pts: usize) -> Self {
        self.denoise_radius = radius;
        self.denoise_min_pts = min_pts;
        self
    }
}

impl ClusteringStrategy for RansacCluster {
    fn run(&mut self, points: &[[f32; 3]]) -> (Vec<[f32; 3]>, Vec<Vec<usize>>) {
        // 1. 可选预降噪
        let working = if self.denoise_radius > 0.0 {
            let grid = XYGrid::new(points, self.denoise_radius);
            let mut kept = Vec::with_capacity(points.len());
            let mut nbr_buf = Vec::new();
            for p in points.iter() {
                nbr_buf.clear();
                grid.query_neighbors(points, p[0], p[1], self.denoise_radius, &mut nbr_buf);
                if nbr_buf.len() >= self.denoise_min_pts {
                    kept.push(*p);
                }
            }
            kept
        } else {
            points.to_vec()
        };

        let n = working.len();
        if n < self.min_points { return (working, Vec::new()); }

        // 2. 顺序 RANSAC：每次提取一条线 → 移除内点 → 继续
        let mut used = vec![false; n];
        let mut clusters: Vec<Vec<usize>> = Vec::new();

        for _ in 0..self.max_clusters {
            let remaining_indices: Vec<usize> = (0..n).filter(|&i| !used[i]).collect();
            if remaining_indices.len() < self.min_points { break; }

            let remaining_pts: Vec<[f32; 3]> = remaining_indices.iter().map(|&i| working[i]).collect();

            let (nx, ny, d, _) = match best_xy_line(&remaining_pts, self.distance, self.iterations, None) {
                Some(v) => v,
                None => break,
            };

            // 收集内点
            let mut inliers = Vec::new();
            for (rel_i, &abs_i) in remaining_indices.iter().enumerate() {
                let dist = (nx * remaining_pts[rel_i][0] + ny * remaining_pts[rel_i][1] + d).abs();
                if dist < self.distance {
                    inliers.push(abs_i);
                    used[abs_i] = true;
                }
            }

            if inliers.len() < self.min_points {
                for &idx in &inliers {
                    used[idx] = false;
                }
                continue;
            }

            clusters.push(inliers);
        }

        (working, clusters)
    }

    fn strategy_name(&self) -> &'static str {
        "ransac"
    }
}
