use nalgebra::Matrix3;
use super::ClusteringStrategy;
use crate::cloud::wall::XYGrid;

/// 顺序 SVD 平面拟合聚类策略（seq）。
///
/// 每轮 SVD 拟合主平面 → 内点作为一个簇 → 移除 → 重复。
/// 从墙体检测 `seq_pca_grid` 改编，去掉了墙体专用过滤（竖直性法线阈值）。
///
/// 可选预降噪：若 `denoise_radius > 0`，先执行半径离群点剔除。
pub struct SeqCluster {
    distance: f32,
    min_points: usize,
    max_clusters: usize,
    max_iters: usize,
    denoise_radius: f32,
    denoise_min_pts: usize,
}

impl SeqCluster {
    pub fn new(distance: f32, min_points: usize) -> Self {
        Self {
            distance,
            min_points,
            max_clusters: 10,
            max_iters: 50,
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

fn fit_plane_svd(points: &[[f32; 3]]) -> Option<([f32; 3], f32)> {
    let n = points.len();
    if n < 3 { return None; }

    let nf = n as f32;
    let mut cx = 0.0f32; let mut cy = 0.0f32; let mut cz = 0.0f32;
    for p in points { cx += p[0]; cy += p[1]; cz += p[2]; }
    cx /= nf; cy /= nf; cz /= nf;

    let mut cov = Matrix3::zeros();
    for p in points {
        let dx = p[0] - cx; let dy = p[1] - cy; let dz = p[2] - cz;
        cov[(0, 0)] += dx * dx; cov[(0, 1)] += dx * dy; cov[(0, 2)] += dx * dz;
        cov[(1, 1)] += dy * dy; cov[(1, 2)] += dy * dz;
        cov[(2, 2)] += dz * dz;
    }
    cov /= nf;
    cov[(1, 0)] = cov[(0, 1)];
    cov[(2, 0)] = cov[(0, 2)];
    cov[(2, 1)] = cov[(1, 2)];

    let eig = cov.symmetric_eigen();
    let mut min_idx = 0;
    let mut min_val = eig.eigenvalues[0];
    for i in 1..3 {
        if eig.eigenvalues[i] < min_val { min_val = eig.eigenvalues[i]; min_idx = i; }
    }
    let nv = eig.eigenvectors.column(min_idx);
    let normal = [nv[0], nv[1], nv[2]];
    let d = -(normal[0] * cx + normal[1] * cy + normal[2] * cz);
    Some((normal, d))
}

impl ClusteringStrategy for SeqCluster {
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

        // 2. 顺序 SVD 平面拟合：每次拟合 → 内点成簇 → 移除 → 继续
        let mut used = vec![false; n];
        let mut clusters: Vec<Vec<usize>> = Vec::new();
        let mut consecutive_fails = 0usize;

        for _ in 0..self.max_iters {
            if clusters.len() >= self.max_clusters { break; }
            if consecutive_fails >= 3 { break; }

            let pts: Vec<[f32; 3]> = (0..n)
                .filter(|&i| !used[i])
                .map(|i| working[i])
                .collect();
            if pts.len() < self.min_points { break; }

            let (normal, d) = match fit_plane_svd(&pts) {
                Some(v) => v,
                None => break,
            };

            // 收集内点
            let mut inliers = Vec::new();
            for i in 0..n {
                if used[i] { continue; }
                let dist = (normal[0] * working[i][0] + normal[1] * working[i][1]
                    + normal[2] * working[i][2] + d).abs();
                if dist < self.distance {
                    inliers.push(i);
                }
            }

            if inliers.len() < self.min_points {
                for &i in &inliers { used[i] = true; }
                consecutive_fails += 1;
                continue;
            }

            consecutive_fails = 0;
            for &i in &inliers { used[i] = true; }
            clusters.push(inliers);
        }

        (working, clusters)
    }

    fn strategy_name(&self) -> &'static str {
        "seq"
    }
}
