use std::collections::{VecDeque, HashSet};
use super::{WallPickStrategy, XYGrid};

/// 下采样策略
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Downsampler {
    /// LV-DOT 风格网格下采样：每个格子取质心，最快
    Grid,
    /// 最远点采样（XY 平面）：空间均匀性最好
    FPS,
}

/// 自适应 XY DBSCAN 墙体提取。
///
/// 核心思路：
/// - 先下采样（Grid/FPS），将点数压到 max_pts 以内
/// - eps 随距离动态调整：近处稠密用小 eps，远处稀疏用大 eps
/// - XYGrid 空间索引，O(1) 邻域查询
/// - 每簇用 PCA 细长比 + Z 跨度过滤，保留墙面特征
pub struct AdaptiveDBSCANWall {
    /// 基础邻域半径 (m)
    base_eps: f32,
    /// 距离缩放因子：eps(d) = base_eps + d * scale_factor
    scale_factor: f32,
    /// 核心点最小邻居数
    min_pts: usize,
    /// 最小墙面点数
    min_wall_pts: usize,
    /// 最大墙面数
    max_walls: usize,
    /// PCA 细长比阈值
    max_width_ratio: f32,
    /// 墙面最小 Z 跨度 (m)
    min_z_span: f32,
    /// 下采样策略
    downsampler: Downsampler,
    /// 下采样目标点数
    max_pts: usize,
}

impl AdaptiveDBSCANWall {
    pub fn new() -> Self {
        Self {
            base_eps: 0.08,
            scale_factor: 0.015,
            min_pts: 5,
            min_wall_pts: 30,
            max_walls: 8,
            max_width_ratio: 0.25,
            min_z_span: 1.0,
            downsampler: Downsampler::FPS,
            max_pts: 4000,
        }
    }

    pub fn with_params(base_eps: f32, scale_factor: f32, min_pts: usize) -> Self {
        Self { base_eps, scale_factor, min_pts, ..Self::new() }
    }

    pub fn with_width_ratio(mut self, ratio: f32) -> Self {
        self.max_width_ratio = ratio;
        self
    }

    pub fn with_downsampler(mut self, d: Downsampler) -> Self {
        self.downsampler = d;
        self
    }

    pub fn with_max_pts(mut self, n: usize) -> Self {
        self.max_pts = n;
        self
    }

    /// 自适应 eps：近处小，远处大
    fn adaptive_eps(&self, x: f32, y: f32) -> f32 {
        let dist = (x * x + y * y).sqrt();
        self.base_eps + dist * self.scale_factor
    }
}

impl WallPickStrategy for AdaptiveDBSCANWall {
    fn strategy_name(&self) -> &'static str { "adaptive_dbscan" }

    fn pick(&mut self, cloud: &mut [[f32; 3]]) -> (usize, Vec<[f32; 4]>) {
        let n = cloud.len();
        if n < self.min_wall_pts { return (0, Vec::new()); }

        // 1. 下采样（复用 XYGrid 的方法）
        let (sampled, idx_map) = match self.downsampler {
            Downsampler::Grid => XYGrid::grid_downsample(cloud, self.max_pts),
            Downsampler::FPS  => XYGrid::fps_downsample(cloud, self.max_pts),
        };
        let ns = sampled.len();

        // 2. 预计算每个采样点的自适应 eps
        let eps_per_point: Vec<f32> = sampled.iter()
            .map(|p| self.adaptive_eps(p[0], p[1]))
            .collect();

        // 3. 用最大 eps 构建统一网格（复用 XYGrid）
        let max_eps = eps_per_point.iter().cloned().fold(0.0f32, f32::max);
        let grid = XYGrid::new(&sampled, max_eps);

        // 4. DBSCAN — 自适应邻域半径
        let mut visited = vec![false; ns];
        let mut clusters: Vec<Vec<usize>> = Vec::new();
        let mut nbr_buf = Vec::new();

        for i in 0..ns {
            if visited[i] { continue; }
            visited[i] = true;

            nbr_buf.clear();
            grid.query_neighbors(&sampled, sampled[i][0], sampled[i][1], eps_per_point[i], &mut nbr_buf);
            if nbr_buf.len() < self.min_pts { continue; }

            let mut cluster = vec![i];
            let mut queue: VecDeque<usize> = VecDeque::new();
            let seed_nbrs: Vec<usize> = nbr_buf.drain(..).collect();
            for &j in &seed_nbrs {
                if !visited[j] {
                    visited[j] = true;
                    queue.push_back(j);
                }
            }

            while let Some(cur) = queue.pop_front() {
                cluster.push(cur);
                nbr_buf.clear();
                grid.query_neighbors(&sampled, sampled[cur][0], sampled[cur][1], eps_per_point[cur], &mut nbr_buf);
                if nbr_buf.len() >= self.min_pts {
                    for &j in &nbr_buf {
                        if !visited[j] {
                            visited[j] = true;
                            queue.push_back(j);
                        }
                    }
                }
            }
            clusters.push(cluster);
        }

        // 5. 每簇验证：PCA 细长比 + Z 跨度
        let mut walls: Vec<(Vec<usize>, [f32; 4])> = Vec::new();

        for cluster in &clusters {
            if cluster.len() < self.min_wall_pts { continue; }

            // Z 跨度
            let mut z_min = f32::MAX;
            let mut z_max = f32::MIN;
            for &i in cluster {
                if sampled[i][2] < z_min { z_min = sampled[i][2]; }
                if sampled[i][2] > z_max { z_max = sampled[i][2]; }
            }
            if z_max - z_min < self.min_z_span { continue; }

            // 2D PCA
            let nf = cluster.len() as f32;
            let cx: f32 = cluster.iter().map(|&i| sampled[i][0]).sum::<f32>() / nf;
            let cy: f32 = cluster.iter().map(|&i| sampled[i][1]).sum::<f32>() / nf;

            let mut cxx = 0.0f32;
            let mut cxy = 0.0f32;
            let mut cyy = 0.0f32;
            for &i in cluster {
                let dx = sampled[i][0] - cx;
                let dy = sampled[i][1] - cy;
                cxx += dx * dx;
                cxy += dx * dy;
                cyy += dy * dy;
            }
            cxx /= nf; cxy /= nf; cyy /= nf;

            let trace = cxx + cyy;
            let det = cxx * cyy - cxy * cxy;
            let disc = (trace * trace - 4.0 * det).max(0.0).sqrt();
            let lambda_max = (trace + disc) * 0.5;
            let lambda_min = (trace - disc) * 0.5;

            if lambda_max < 1e-8 { continue; }
            let ratio = lambda_min / lambda_max;
            if ratio >= self.max_width_ratio { continue; }

            // 2D 法线
            let nx = cxy;
            let ny = lambda_min - cxx;
            let len = (nx * nx + ny * ny).sqrt();
            let (nx, ny) = if len > 1e-8 { (nx / len, ny / len) } else { (1.0, 0.0) };
            let d = -(nx * cx + ny * cy);

            walls.push((cluster.clone(), [nx, ny, 0.0, d]));
        }

        walls.sort_by(|a, b| b.0.len().cmp(&a.0.len()));
        walls.truncate(self.max_walls);

        // 6. 将采样点的簇标签映射回原始点
        let mut wall_original_set = HashSet::new();
        for (cluster, _) in &walls {
            for &si in cluster {
                wall_original_set.insert(idx_map[si]);
            }
        }

        // 原地重排
        let mut write = 0usize;
        for read in 0..n {
            if wall_original_set.contains(&read) {
                cloud.swap(read, write);
                write += 1;
            }
        }

        let planes: Vec<[f32; 4]> = walls.into_iter().map(|(_, p)| p).collect();
        (write, planes)
    }
}
