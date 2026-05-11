use super::wall::XYGrid;

/// 降噪策略 trait
///
/// 输入点云，剔除稀疏离群点，返回 (保留点, 保留点在原始输入中的索引映射)。
pub trait DenoiseStrategy: Send {
    fn denoise(&mut self, points: &[[f32; 3]]) -> (Vec<[f32; 3]>, Vec<usize>);
    fn strategy_name(&self) -> &'static str { "unknown" }
}

/// 半径离群点剔除：若某点的邻域半径内点数 < min_pts，视为噪点剔除。
///
/// 使用 XYGrid（哈希网格）实现 O(n) 平均查询。
pub struct RadiusOutlierRemoval {
    radius: f32,
    min_pts: usize,
}

impl RadiusOutlierRemoval {
    pub fn new(radius: f32, min_pts: usize) -> Self {
        Self { radius, min_pts }
    }
}

impl DenoiseStrategy for RadiusOutlierRemoval {
    fn denoise(&mut self, points: &[[f32; 3]]) -> (Vec<[f32; 3]>, Vec<usize>) {
        let n = points.len();
        if n < self.min_pts || self.radius <= 0.0 {
            return (points.to_vec(), (0..n).collect());
        }
        let grid = XYGrid::new(points, self.radius);
        let mut kept = Vec::with_capacity(n);
        let mut map = Vec::with_capacity(n);
        let mut neighbors = Vec::new();
        for (i, p) in points.iter().enumerate() {
            neighbors.clear();
            grid.query_neighbors(points, p[0], p[1], self.radius, &mut neighbors);
            if neighbors.len() >= self.min_pts {
                kept.push(*p);
                map.push(i);
            }
        }
        (kept, map)
    }
}

/// 统计离群点剔除（SOR）：若某点到 k 近邻平均距离 > mean + std_ratio * std，视为噪点。
///
/// 使用暴力计算，适用于小规模点云。大点云建议用 RadiusOutlierRemoval。
pub struct StatisticalOutlierRemoval {
    k: usize,
    std_ratio: f32,
}

impl StatisticalOutlierRemoval {
    pub fn new(k: usize, std_ratio: f32) -> Self {
        Self { k, std_ratio }
    }
}

impl DenoiseStrategy for StatisticalOutlierRemoval {
    fn denoise(&mut self, points: &[[f32; 3]]) -> (Vec<[f32; 3]>, Vec<usize>) {
        let n = points.len();
        if n < self.k || self.k == 0 {
            return (points.to_vec(), (0..n).collect());
        }
        // 使用 XYGrid 对每个点搜索 k 近邻
        let grid = XYGrid::new(points, 10.0); // 大 cell_size 保证能搜到足够邻居
        let mut mean_dists = Vec::with_capacity(n);
        let mut sum_dist = 0.0f64;

        for p in points.iter() {
            let mut neighbors = Vec::new();
            // 逐步扩大搜索半径直到找到 k 个邻居或超过最大范围
            let mut radius = 0.5;
            loop {
                neighbors.clear();
                grid.query_neighbors(points, p[0], p[1], radius, &mut neighbors);
                if neighbors.len() >= self.k || radius > 20.0 {
                    break;
                }
                radius *= 2.0;
            }
            if neighbors.is_empty() {
                mean_dists.push(0.0);
                continue;
            }
            // 取最近的 self.k 个邻居的距离均值
            let mut dists: Vec<f32> = neighbors.iter()
                .map(|&j| {
                    let dx = points[j][0] - p[0];
                    let dy = points[j][1] - p[1];
                    let dz = points[j][2] - p[2];
                    (dx * dx + dy * dy + dz * dz).sqrt()
                })
                .collect();
            dists.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
            let k_actual = self.k.min(dists.len());
            let mean: f32 = dists[..k_actual].iter().sum::<f32>() / k_actual as f32;
            mean_dists.push(mean);
            sum_dist += mean as f64;
        }

        let global_mean = (sum_dist / n as f64) as f32;
        let var: f32 = mean_dists.iter().map(|d| (*d - global_mean).powi(2)).sum::<f32>() / n as f32;
        let std = var.max(0.0).sqrt();
        let threshold = global_mean + self.std_ratio * std;

        let mut kept = Vec::with_capacity(n);
        let mut map = Vec::with_capacity(n);
        for (i, &d) in mean_dists.iter().enumerate() {
            if d <= threshold {
                kept.push(points[i]);
                map.push(i);
            }
        }
        (kept, map)
    }
}
