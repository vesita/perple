use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::config::fixif;
use crate::cloud::classify::split_policy::FixedDepthPolicy;
use super::ClusteringStrategy;

use super::super::quadtree::QuadTreeNode;

/// 四叉树 DBSCAN 策略（dbscan_qt，支持固定 eps 和自适应 eps）
pub struct DbscanStrategy {
    patience: f32,
    eps_slope: f32,
    min_points: usize,
    max_points_per_node: usize,
    max_tree_depth: usize,
    voxel_size: f32,
    downsample_method: String,
    gaussian_sigma: f32,
    quad_tree: Option<QuadTreeNode>,
    x_min: f32,
    x_max: f32,
    y_min: f32,
    y_max: f32,
}

impl DbscanStrategy {
    pub fn new() -> Self {
        let cfg = fixif();
        Self {
            patience: cfg.cluster.merge_patience,
            eps_slope: cfg.cluster.eps_slope,
            min_points: cfg.cluster.min_points_per_cluster.unwrap_or(3),
            max_points_per_node: cfg.cluster.max_points_per_node.unwrap_or(20),
            max_tree_depth: cfg.cluster.max_tree_depth.unwrap_or(10),
            voxel_size: cfg.cluster.voxel_size,
            downsample_method: cfg.cluster.downsample_method.clone(),
            gaussian_sigma: cfg.cluster.gaussian_downsample_rate,
            quad_tree: None,
            x_min: -100.0,
            x_max: 100.0,
            y_min: -100.0,
            y_max: 100.0,
        }
    }

    /// 轻量构造器：跳过内部下采样，直接对输入做 DBSCAN。
    ///
    /// 适用于上游已做完墙体检测+LV-DOT过滤的管线场景。
    pub fn new_light() -> Self {
        Self { voxel_size: 0.0, downsample_method: "none".to_string(), ..Self::new() }
    }

    /// 跳过内部下采样（用于管线已预处理过的场景）
    pub fn with_no_downsample(mut self) -> Self {
        self.voxel_size = 0.0;
        self
    }

    /// 带参数的构造器，用于 benchmark 直接测试不同参数组合
    pub fn with_params(
        patience: f32,
        eps_slope: f32,
        min_points: usize,
        max_points_per_node: usize,
        max_tree_depth: usize,
        voxel_size: f32,
    ) -> Self {
        Self {
            patience,
            eps_slope,
            min_points,
            max_points_per_node,
            max_tree_depth,
            voxel_size,
            downsample_method: "voxel".to_string(),
            gaussian_sigma: 6.0,
            quad_tree: None,
            x_min: -100.0,
            x_max: 100.0,
            y_min: -100.0,
            y_max: 100.0,
        }
    }

    /// 执行聚类，返回 (处理后的点集, 簇索引列表)
    pub fn run(&mut self, lifra: &[[f32; 3]]) -> (Vec<[f32; 3]>, Vec<Vec<usize>>) {
        // 一次下采样（voxel 或 gaussian）
        let sampled = self.downsample_points(lifra);
        // 二次确定性体素滤波：每个体素取质心，消除随机采样残余噪声
        let mut sampled = self.voxel_centroid_downsample(&sampled);
        let n = sampled.len();
        if n == 0 {
            return (sampled, Vec::new());
        }

        println!("下采样后剩余 {} 个点", n);

        // 构建四叉树（API 需要 &Vec）
        self.build_quad_tree(&sampled);

        let objects = if self.eps_slope > 0.0 {
            self.cluster_adaptive(&mut sampled)
        } else {
            self.cluster_fixed(&sampled)
        };

        println!("聚类完成，共发现 {} 个聚类", objects.len());
        (sampled, objects)
    }

    /// 下采样入口：根据配置选择 voxel 或 Gaussian 概率采样
    fn downsample_points(&self, points: &[[f32; 3]]) -> Vec<[f32; 3]> {
        if self.voxel_size <= 0.0 {
            return points.to_vec();
        }
        match self.downsample_method.as_str() {
            "gaussian" => self.gaussian_downsample(points),
            _ => self.voxel_downsample(points),
        }
    }

    /// 均匀体素下采样：每个体素格子保留一个点
    fn voxel_downsample(&self, points: &[[f32; 3]]) -> Vec<[f32; 3]> {
        let mut seen = HashMap::new();
        let mut result = Vec::new();
        for p in points {
            let key = [
                (p[0] / self.voxel_size).floor() as i32,
                (p[1] / self.voxel_size).floor() as i32,
                (p[2] / self.voxel_size).floor() as i32,
            ];
            if seen.insert(key, true).is_none() {
                result.push(*p);
            }
        }
        result
    }

    /// Gaussian 概率下采样（LV-DOT 风格）
    ///
    /// 直接作用于原始点，用 2D XY 水平距离计算保留概率 exp(-d_xy²/2σ²)。
    /// 近处点保留率高、远处点保留率低，补偿 LiDAR 近密远疏的特性。
    /// 使用坐标哈希确定性采样，保证相同位置的点每帧行为一致。
    fn gaussian_downsample(&self, points: &[[f32; 3]]) -> Vec<[f32; 3]> {
        if self.gaussian_sigma <= 0.0 {
            return points.to_vec();
        }
        let sigma2 = self.gaussian_sigma * self.gaussian_sigma;
        points.iter()
            .filter(|p| {
                // LV-DOT: 仅用 XY 水平距离，Z 不参与概率计算
                let d2 = p[0] * p[0] + p[1] * p[1];
                let keep_prob = (-d2 / (2.0 * sigma2)).exp();
                // 确定性哈希代替随机数
                let hash = ((p[0] * 73856093.0) as i32 ^ (p[1] * 19349663.0) as i32)
                           .unsigned_abs() as f32 / u32::MAX as f32;
                hash < keep_prob
            })
            .copied()
            .collect()
    }

    /// 确定性体素质心下采样：每个体素内所有点取质心作为代表
    /// 与 voxel_downsample（取第一个点）不同，这里取平均值，消除点集组成噪声
    fn voxel_centroid_downsample(&self, points: &[[f32; 3]]) -> Vec<[f32; 3]> {
        if self.voxel_size <= 0.0 {
            return points.to_vec();
        }
        let inv = 1.0 / self.voxel_size;
        let mut voxels: HashMap<[i32; 3], (f64, f64, f64, usize)> = HashMap::new();
        for p in points {
            let key = [
                (p[0] * inv).floor() as i32,
                (p[1] * inv).floor() as i32,
                (p[2] * inv).floor() as i32,
            ];
            let entry = voxels.entry(key).or_insert((0.0, 0.0, 0.0, 0));
            entry.0 += p[0] as f64;
            entry.1 += p[1] as f64;
            entry.2 += p[2] as f64;
            entry.3 += 1;
        }
        voxels.into_iter()
            .map(|(_, (sx, sy, sz, cnt))| {
                let n = cnt as f64;
                [(sx / n) as f32, (sy / n) as f32, (sz / n) as f32]
            })
            .collect()
    }

    /// 构建四叉树
    fn build_quad_tree(&mut self, points: &Vec<[f32; 3]>) {
        let mut root = QuadTreeNode::new(self.x_min, self.x_max, self.y_min, self.y_max)
            .with_max_pts_per_node(self.max_points_per_node)
            .with_policy(Arc::new(FixedDepthPolicy::new(self.max_tree_depth)));
        for i in 0..points.len() {
            root.insert_point(i, points);
        }
        self.quad_tree = Some(root);
    }

    /// 固定 eps DBSCAN
    fn cluster_fixed(&self, points: &Vec<[f32; 3]>) -> Vec<Vec<usize>> {
        let n = points.len();
        let mut visited = vec![false; n];
        let mut labels = vec![-1i32; n];
        let mut cluster_id = 0i32;
        let mut objects = Vec::new();

        for i in 0..n {
            if visited[i] { continue; }
            visited[i] = true;

            let mut neighbors = Vec::new();
            if let Some(qt) = &self.quad_tree {
                qt.query_range(points[i][0], points[i][1], self.patience, points, &mut neighbors);
            }
            // 四叉树仅做 XY 搜索，Z 轴单独过滤：避免不同高度点被误聚类
            let zi = points[i][2];
            neighbors.retain(|&j| (points[j][2] - zi).abs() < self.patience);

            if neighbors.len() < self.min_points {
                continue;
            }

            labels[i] = cluster_id;
            let mut cluster = vec![i];
            let mut neighbor_set: HashSet<usize> = neighbors.into_iter().collect();
            let mut nvec: Vec<usize> = neighbor_set.iter().copied().collect();

            let mut k = 0;
            while k < nvec.len() {
                let ni = nvec[k];
                if !visited[ni] {
                    visited[ni] = true;

                    let mut more = Vec::new();
                    if let Some(qt) = &self.quad_tree {
                        qt.query_range(points[ni][0], points[ni][1], self.patience, points, &mut more);
                    }
                    // Z 轴过滤
                    let zni = points[ni][2];
                    more.retain(|&j| (points[j][2] - zni).abs() < self.patience);

                    if more.len() >= self.min_points {
                        for &m in &more {
                            if neighbor_set.insert(m) {
                                nvec.push(m);
                            }
                        }
                    }
                }

                if labels[ni] == -1 {
                    labels[ni] = cluster_id;
                    cluster.push(ni);
                }
                k += 1;
            }

            objects.push(cluster);
            cluster_id += 1;
        }

        objects
    }

    /// 自适应 eps DBSCAN：eps(p) = patience + eps_slope * range(p)
    fn cluster_adaptive(&self, points: &mut Vec<[f32; 3]>) -> Vec<Vec<usize>> {
        let n = points.len();
        let ranges: Vec<f32> = points.iter()
            .map(|p| (p[0] * p[0] + p[1] * p[1]).sqrt())
            .collect();
        let max_range = ranges.iter().copied().fold(0.0f32, f32::max);
        let query_radius = self.patience + self.eps_slope * max_range;

        let mut visited = vec![false; n];
        let mut labels = vec![-1i32; n];
        let mut cluster_id = 0i32;
        let mut objects = Vec::new();

        for i in 0..n {
            if visited[i] { continue; }
            visited[i] = true;

            let mut candidates = Vec::new();
            if let Some(qt) = &self.quad_tree {
                qt.query_range(points[i][0], points[i][1], query_radius, points, &mut candidates);
            }

            let mut neighbors = Vec::new();
            for &j in &candidates {
                if i == j { continue; }
                let dx = points[i][0] - points[j][0];
                let dy = points[i][1] - points[j][1];
                let dz = (points[i][2] - points[j][2]).abs();
                let eps_ij = self.patience + self.eps_slope * ranges[i].max(ranges[j]);
                if dx * dx + dy * dy < eps_ij * eps_ij && dz < eps_ij {
                    neighbors.push(j);
                }
            }

            if neighbors.len() < self.min_points {
                continue;
            }

            labels[i] = cluster_id;
            let mut cluster = vec![i];
            let mut neighbor_set: HashSet<usize> = neighbors.iter().copied().collect();
            let mut nvec = neighbors;

            let mut k = 0;
            while k < nvec.len() {
                let ni = nvec[k];
                if !visited[ni] {
                    visited[ni] = true;

                    let mut more_candidates = Vec::new();
                    if let Some(qt) = &self.quad_tree {
                        qt.query_range(points[ni][0], points[ni][1], query_radius, points, &mut more_candidates);
                    }

                    let mut more = Vec::new();
                    for &j in &more_candidates {
                        if ni == j { continue; }
                        let dx = points[ni][0] - points[j][0];
                        let dy = points[ni][1] - points[j][1];
                        let dz = (points[ni][2] - points[j][2]).abs();
                        let eps_ij = self.patience + self.eps_slope * ranges[ni].max(ranges[j]);
                        if dx * dx + dy * dy < eps_ij * eps_ij && dz < eps_ij {
                            more.push(j);
                        }
                    }

                    if more.len() >= self.min_points {
                        for &m in &more {
                            if neighbor_set.insert(m) {
                                nvec.push(m);
                            }
                        }
                    }
                }

                if labels[ni] == -1 {
                    labels[ni] = cluster_id;
                    cluster.push(ni);
                }
                k += 1;
            }

            objects.push(cluster);
            cluster_id += 1;
        }

        objects
    }
}

impl ClusteringStrategy for DbscanStrategy {
    fn run(&mut self, points: &[[f32; 3]]) -> (Vec<[f32; 3]>, Vec<Vec<usize>>) {
        self.run(points)
    }

    fn strategy_name(&self) -> &'static str {
        "dbscan_qt"
    }
}
