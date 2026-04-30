use std::collections::HashMap;

use crate::config::fixif;
use super::ClusteringStrategy;

/// Range Image 聚类策略（FLIC 风格）
///
/// 将点云投影为 2D 球面 range image，做 4-连通区域标记（CCL）
pub struct RangeImageStrategy {
    az_res: f32,       // 水平角分辨率（弧度）
    el_res: f32,       // 垂直角分辨率（弧度）
    threshold: f32,    // 3D 距离阈值（米）
    min_points: usize,
}

impl RangeImageStrategy {
    pub fn new() -> Self {
        let cfg = fixif();
        Self {
            az_res: cfg.claster.azimuth_resolution * std::f32::consts::PI / 180.0,
            el_res: cfg.claster.elevation_resolution * std::f32::consts::PI / 180.0,
            threshold: cfg.claster.cluster_threshold,
            min_points: cfg.claster.min_points_per_cluster.unwrap_or(3),
        }
    }

    /// 带参数的构造器，用于 benchmark 直接测试不同参数组合
    pub fn with_params(
        az_res_deg: f32,
        el_res_deg: f32,
        threshold: f32,
        min_points: usize,
    ) -> Self {
        Self {
            az_res: az_res_deg * std::f32::consts::PI / 180.0,
            el_res: el_res_deg * std::f32::consts::PI / 180.0,
            threshold,
            min_points,
        }
    }

    /// 执行聚类，返回 (原样点集, 簇索引列表)
    pub fn run(&mut self, lifra: &[[f32; 3]]) -> (Vec<[f32; 3]>, Vec<Vec<usize>>) {
        let n = lifra.len();
        if n < 2 {
            return (lifra.to_vec(), Vec::new());
        }

        let az_min = -std::f32::consts::PI;
        let el_min = -std::f32::consts::FRAC_PI_2;
        let cols = ((2.0 * std::f32::consts::PI) / self.az_res).ceil() as usize;
        let rows = (std::f32::consts::PI / self.el_res).ceil() as usize;

        // Range image：每个 cell 存最近点的索引
        let mut img: Vec<Vec<Option<usize>>> = vec![vec![None; rows]; cols];
        let mut img_r: Vec<Vec<f32>> = vec![vec![f32::MAX; rows]; cols];

        for (i, p) in lifra.iter().enumerate() {
            let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
            let az = p[1].atan2(p[0]);
            let el = (p[2] / r.max(1e-8)).asin();
            let u = ((az - az_min) / self.az_res) as usize;
            let v = ((el - el_min) / self.el_res) as usize;
            if u < cols && v < rows && r < img_r[u][v] {
                img_r[u][v] = r;
                img[u][v] = Some(i);
            }
        }

        // CCL：只需保留有点的 cell 做连通性分析
        let total = cols.saturating_mul(rows);
        if total == 0 {
            return (lifra.to_vec(), Vec::new());
        }

        let mut parent: Vec<usize> = (0..total).collect();
        let mut occupied = vec![false; total];

        for u in 0..cols {
            for v in 0..rows {
                if img[u][v].is_some() {
                    occupied[u * rows + v] = true;
                }
            }
        }

        fn find(parent: &mut [usize], x: usize) -> usize {
            let mut p = x;
            while parent[p] != p {
                parent[p] = parent[parent[p]];
                p = parent[p];
            }
            p
        }

        fn union(parent: &mut [usize], a: usize, b: usize) {
            let ra = find(parent, a);
            let rb = find(parent, b);
            if ra != rb {
                parent[ra] = rb;
            }
        }

        // 4-连通 CCL
        for u in 0..cols {
            for v in 0..rows {
                let idx = u * rows + v;
                if !occupied[idx] {
                    continue;
                }
                let p1 = lifra[img[u][v].unwrap()];

                // 右邻
                if u + 1 < cols {
                    let nidx = (u + 1) * rows + v;
                    if occupied[nidx] {
                        let p2 = lifra[img[u + 1][v].unwrap()];
                        let d2 = (p1[0] - p2[0]).powi(2)
                            + (p1[1] - p2[1]).powi(2)
                            + (p1[2] - p2[2]).powi(2);
                        if d2 < self.threshold * self.threshold {
                            union(&mut parent, idx, nidx);
                        }
                    }
                }
                // 上邻
                if v + 1 < rows {
                    let nidx = u * rows + (v + 1);
                    if occupied[nidx] {
                        let p2 = lifra[img[u][v + 1].unwrap()];
                        let d2 = (p1[0] - p2[0]).powi(2)
                            + (p1[1] - p2[1]).powi(2)
                            + (p1[2] - p2[2]).powi(2);
                        if d2 < self.threshold * self.threshold {
                            union(&mut parent, idx, nidx);
                        }
                    }
                }
            }
        }

        // 收集连通分量
        let mut comps: HashMap<usize, Vec<usize>> = HashMap::new();
        for u in 0..cols {
            for v in 0..rows {
                let idx = u * rows + v;
                if !occupied[idx] {
                    continue;
                }
                let root = find(&mut parent, idx);
                comps.entry(root).or_default().push(img[u][v].unwrap());
            }
        }

        let mut objects: Vec<Vec<usize>> = Vec::new();
        for (_, indices) in comps {
            if indices.len() >= self.min_points {
                objects.push(indices);
            }
        }

        println!("range image 聚类完成，{} 像素 → {} 簇", occupied.iter().filter(|o| **o).count(), objects.len());
        (lifra.to_vec(), objects)
    }
}

impl ClusteringStrategy for RangeImageStrategy {
    fn run(&mut self, points: &[[f32; 3]]) -> (Vec<[f32; 3]>, Vec<Vec<usize>>) {
        self.run(points)
    }

    fn strategy_name(&self) -> &'static str {
        "range_image"
    }
}
