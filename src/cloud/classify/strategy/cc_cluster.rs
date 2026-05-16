use std::collections::{HashMap, VecDeque, HashSet};
use super::ClusteringStrategy;
use crate::cloud::wall::XYGrid;

/// 连通域聚类策略（cc_grid）。
///
/// XY 哈希网格 + BFS 连通域合并，每个连通域为一个簇。
/// 从墙体提取 `cc_pca_grid` 改编，去掉了墙体专用过滤（Z 跨度、PCA 细长比）。
///
/// 可选预降噪：若 `denoise_radius > 0`，先执行半径离群点剔除。
pub struct CcCluster {
    cell_size: f32,
    min_points: usize,
    merge_dist: usize,
    denoise_radius: f32,
    denoise_min_pts: usize,
}

impl CcCluster {
    pub fn new(cell_size: f32, min_points: usize) -> Self {
        Self {
            cell_size,
            min_points,
            merge_dist: 1,
            denoise_radius: 0.0,
            denoise_min_pts: 3,
        }
    }

    pub fn with_params(cell_size: f32, min_points: usize, merge_dist: usize) -> Self {
        Self { cell_size, min_points, merge_dist, ..Self::new(cell_size, min_points) }
    }

    pub fn with_denoise(mut self, radius: f32, min_pts: usize) -> Self {
        self.denoise_radius = radius;
        self.denoise_min_pts = min_pts;
        self
    }
}

impl ClusteringStrategy for CcCluster {
    fn run(&mut self, points: &[[f32; 3]]) -> (Vec<[f32; 3]>, Vec<Vec<usize>>) {
        // 1. 可选预降噪
        let working = if self.denoise_radius > 0.0 {
            let grid = XYGrid::new(points, self.denoise_radius);
            let mut kept = Vec::with_capacity(points.len());
            let mut keep_map = Vec::with_capacity(points.len());
            let mut nbr_buf = Vec::new();
            for (i, p) in points.iter().enumerate() {
                nbr_buf.clear();
                grid.query_neighbors(points, p[0], p[1], self.denoise_radius, &mut nbr_buf);
                if nbr_buf.len() >= self.denoise_min_pts {
                    kept.push(*p);
                    keep_map.push(i);
                }
            }
            kept
        } else {
            points.to_vec()
        };

        let n = working.len();
        if n == 0 { return (working, Vec::new()); }

        // 2. XY 哈希网格
        let mut grid: HashMap<(i32, i32), Vec<usize>> = HashMap::new();
        let inv = 1.0 / self.cell_size;
        for (i, p) in working.iter().enumerate() {
            let key = ((p[0] * inv).floor() as i32, (p[1] * inv).floor() as i32);
            grid.entry(key).or_default().push(i);
        }

        // 3. 密集格筛选
        let valid: HashSet<(i32, i32)> = grid.iter()
            .filter(|(_, v)| v.len() >= self.min_points)
            .map(|(&k, _)| k)
            .collect();

        // 4. BFS 连通域合并
        let md = self.merge_dist as i32;
        let mut visited = HashSet::new();
        let mut clusters: Vec<Vec<usize>> = Vec::new();

        for &key in &valid {
            if visited.contains(&key) { continue; }
            let mut component = Vec::new();
            let mut queue = VecDeque::new();
            queue.push_back(key);
            visited.insert(key);

            while let Some(cur) = queue.pop_front() {
                component.push(cur);
                for dx in -md..=md {
                    for dy in -md..=md {
                        if dx == 0 && dy == 0 { continue; }
                        let nbr = (cur.0 + dx, cur.1 + dy);
                        if valid.contains(&nbr) && !visited.contains(&nbr) {
                            visited.insert(nbr);
                            queue.push_back(nbr);
                        }
                    }
                }
            }
            // 展开为点索引
            let mut indices = Vec::new();
            for key in &component {
                if let Some(cell) = grid.get(key) {
                    indices.extend_from_slice(cell);
                }
            }
            if !indices.is_empty() {
                clusters.push(indices);
            }
        }

        (working, clusters)
    }

    fn strategy_name(&self) -> &'static str {
        "cc_grid"
    }
}
