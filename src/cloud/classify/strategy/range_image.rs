use crate::config::fixif;
use super::ClusteringStrategy;

/// Range Image 聚类策略（FLIC 风格）— 自适应 FOV。
///
/// 将点云投影为 2D 球面 range image，做 4-连通区域标记（CCL）。
///
/// 改进：不再使用固定全球面 FOV，而是根据点云实际分布自动计算
/// 方位角/俯仰角范围，使网格分辨率集中在有点的区域，大幅降低
/// 稀疏场景（去地面+去墙体后 ~1500 点）下的碎片化和内存占用。
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

    /// 计算点云的实际 FOV（方位角/俯仰角范围），外扩 margin 防止边缘截断。
    fn compute_fov(&self, points: &[[f32; 3]]) -> (f32, f32, f32, f32) {
        let mut az_min = f32::MAX;
        let mut az_max = f32::MIN;
        let mut el_min = f32::MAX;
        let mut el_max = f32::MIN;

        for p in points {
            let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt().max(1e-8);
            let az = p[1].atan2(p[0]);
            let el = (p[2] / r).asin();
            az_min = az_min.min(az);
            az_max = az_max.max(az);
            el_min = el_min.min(el);
            el_max = el_max.max(el);
        }

        // 单点或无点保护
        if az_min == f32::MAX {
            return (-std::f32::consts::PI, std::f32::consts::PI,
                    -std::f32::consts::FRAC_PI_2, std::f32::consts::FRAC_PI_2);
        }

        // 外扩 margin：10% + 一个格子的分辨率
        let az_margin = ((az_max - az_min) * 0.1).max(self.az_res);
        let el_margin = ((el_max - el_min) * 0.1).max(self.el_res);

        // 钳制到合法范围
        let az_min = (az_min - az_margin).max(-std::f32::consts::PI);
        let az_max = (az_max + az_margin).min(std::f32::consts::PI);
        let el_min = (el_min - el_margin).max(-std::f32::consts::FRAC_PI_2);
        let el_max = (el_max + el_margin).min(std::f32::consts::FRAC_PI_2);

        (az_min, az_max, el_min, el_max)
    }

    /// 执行聚类，返回 (原样点集, 簇索引列表)
    pub fn run(&mut self, lifra: &[[f32; 3]]) -> (Vec<[f32; 3]>, Vec<Vec<usize>>) {
        let n = lifra.len();
        if n < 2 {
            return (lifra.to_vec(), Vec::new());
        }

        // 自适应 FOV：只分配点云实际占用的网格
        let (az_min, az_max, el_min, el_max) = self.compute_fov(lifra);
        let az_range = az_max - az_min;
        let el_range = el_max - el_min;

        // 保护：避免分辨率过细导致网格爆炸
        if az_range < self.az_res || el_range < self.el_res {
            // 点云太集中，所有点归为一个簇
            return (lifra.to_vec(), vec![(0..n).collect()]);
        }

        let cols = (az_range / self.az_res).ceil() as usize;
        let rows = (el_range / self.el_res).ceil() as usize;
        let total = cols.saturating_mul(rows);

        // 保护：网格过大时回退到全量簇
        if total > 200_000 {
            log::warn!("range_image FOV 过大 ({}×{}={})，跳过网格分配", cols, rows, total);
            return (lifra.to_vec(), vec![(0..n).collect()]);
        }

        // Range image：每个 cell 存最近点的索引
        let mut img: Vec<Option<usize>> = vec![None; total];
        let mut img_r: Vec<f32> = vec![f32::MAX; total];

        for (i, p) in lifra.iter().enumerate() {
            let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
            let az = p[1].atan2(p[0]);
            let el = (p[2] / r.max(1e-8)).asin();
            let u = ((az - az_min) / self.az_res) as usize;
            let v = ((el - el_min) / self.el_res) as usize;
            if u < cols && v < rows {
                let idx = u * rows + v;
                if r < img_r[idx] {
                    img_r[idx] = r;
                    img[idx] = Some(i);
                }
            }
        }

        // CCL
        let mut parent: Vec<usize> = (0..total).collect();
        let mut occupied = vec![false; total];

        for (idx, cell) in img.iter().enumerate() {
            if cell.is_some() {
                occupied[idx] = true;
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
                let p1 = lifra[img[idx].unwrap()];

                // 右邻
                if u + 1 < cols {
                    let nidx = (u + 1) * rows + v;
                    if occupied[nidx] {
                        let p2 = lifra[img[nidx].unwrap()];
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
                        let p2 = lifra[img[nidx].unwrap()];
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
        let mut comps: std::collections::HashMap<usize, Vec<usize>> = std::collections::HashMap::new();
        for (idx, cell) in img.iter().enumerate() {
            if !occupied[idx] {
                continue;
            }
            let root = find(&mut parent, idx);
            comps.entry(root).or_default().push(cell.unwrap());
        }

        let mut objects: Vec<Vec<usize>> = Vec::new();
        for (_, indices) in comps {
            if indices.len() >= self.min_points {
                objects.push(indices);
            }
        }

        println!("range_image: FOV [{:.1}°~{:.1}°]×[{:.1}°~{:.1}°], {}×{}={} cells → {} clusters",
            az_min.to_degrees(), az_max.to_degrees(),
            el_min.to_degrees(), el_max.to_degrees(),
            cols, rows, total,
            objects.len());
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
