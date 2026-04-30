use std::collections::{HashMap, HashSet};

use crate::config::fixif;
use super::ClusteringStrategy;

use super::super::quadtree::QuadTreeNode;

/// 四叉树 DBSCAN 策略（支持固定 eps 和自适应 eps）
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
            patience: cfg.claster.merge_patience,
            eps_slope: cfg.claster.eps_slope,
            min_points: cfg.claster.min_points_per_cluster.unwrap_or(3),
            max_points_per_node: cfg.claster.max_points_per_node.unwrap_or(50),
            max_tree_depth: cfg.claster.max_tree_depth.unwrap_or(10),
            voxel_size: cfg.claster.voxel_size,
            downsample_method: cfg.claster.downsample_method.clone(),
            gaussian_sigma: cfg.claster.gaussian_downsample_rate,
            quad_tree: None,
            x_min: -100.0,
            x_max: 100.0,
            y_min: -100.0,
            y_max: 100.0,
        }
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
        // 下采样
        let mut sampled = self.downsample_points(lifra);
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

    /// Gaussian 概率下采样：近处稀疏、远处密集
    ///
    /// 先用 voxel 做一次基础过滤，再以 exp(-d²/2σ²) 概率保留每个点，
    /// sigma 越大保留越多远处点（LV-DOT 启发）
    fn gaussian_downsample(&self, points: &[[f32; 3]]) -> Vec<[f32; 3]> {
        // 先用体素粗过滤
        let base = self.voxel_downsample(points);
        if self.gaussian_sigma <= 0.0 {
            return base;
        }
        let sigma2 = self.gaussian_sigma * self.gaussian_sigma;
        base.into_iter()
            .filter(|p| {
                let d2 = p[0] * p[0] + p[1] * p[1] + p[2] * p[2];
                let keep_prob = (-d2 / (2.0 * sigma2)).exp();
                rand::random::<f32>() < keep_prob
            })
            .collect()
    }

    /// 构建四叉树
    fn build_quad_tree(&mut self, points: &Vec<[f32; 3]>) {
        let mut root = QuadTreeNode::new(self.x_min, self.x_max, self.y_min, self.y_max);
        for i in 0..points.len() {
            root.insert_point(i, points, self.max_points_per_node, self.max_tree_depth, 0);
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
        if self.eps_slope > 0.0 {
            "dbscan_adaptive"
        } else {
            "dbscan"
        }
    }
}
