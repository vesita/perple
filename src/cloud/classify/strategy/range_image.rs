use crate::config::fixif;
use super::ClusteringStrategy;

/// Range Image 聚类策略（FLIC 风格）— 自适应 FOV，修复 -π/π 缠绕。
///
/// 将点云投影为 2D 球面 range image，做 8-连通区域标记（CCL）+ Z 轴独立约束。
///
/// 改进：不再使用固定全球面 FOV，而是根据点云实际分布自动计算
/// 方位角/俯仰角范围，使网格分辨率集中在有点的区域，大幅降低
/// 稀疏场景（去地面+去墙体后 ~1500 点）下的碎片化和内存占用。
///
/// 缠绕修复：atan2 返回 [-π, π]，跨边界时直接取 min/max 会算出 ~360°
/// 的伪 FOV。检测到缠绕后将负方位角 +2π 映射到 [0, 2π) 空间，使
/// 连续物体在 range image 中保持连续。
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
            az_res: cfg.cluster.azimuth_resolution * std::f32::consts::PI / 180.0,
            el_res: cfg.cluster.elevation_resolution * std::f32::consts::PI / 180.0,
            threshold: cfg.cluster.cluster_threshold,
            min_points: cfg.cluster.min_points_per_cluster.unwrap_or(3),
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

        // ── 1. 一次性计算所有球面坐标，检测 -π/π 缠绕 ────────────
        let mut azs = Vec::with_capacity(n); // 原始方位角 [-π, π]
        let mut els = Vec::with_capacity(n); // 俯仰角
        let mut rs = Vec::with_capacity(n);  // 距离
        let mut az_min_raw = f32::MAX;
        let mut az_max_raw = f32::MIN;

        for p in lifra {
            let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt().max(1e-8);
            let az = p[1].atan2(p[0]);
            let el = (p[2] / r).asin();
            az_min_raw = az_min_raw.min(az);
            az_max_raw = az_max_raw.max(az);
            azs.push(az);
            els.push(el);
            rs.push(r);
        }

        // 单点或无点保护
        if az_min_raw == f32::MAX {
            return (lifra.to_vec(), vec![(0..n).collect()]);
        }

        // 判断是否跨 -π/π 边界
        let wrap = az_max_raw - az_min_raw > std::f32::consts::PI;

        // ── 2. 自适应 FOV（缠绕时用 [0, 2π) 空间） ───────────────
        let (az_min, az_max, el_min, el_max) = if wrap {
            // 缠绕模式：将负方位角 +2π，使角度在 [0, 2π) 连续
            let mut a_min = f32::MAX;
            let mut a_max = f32::MIN;
            let mut e_min = f32::MAX;
            let mut e_max = f32::MIN;
            for i in 0..n {
                let adj_az = if azs[i] < 0.0 { azs[i] + 2.0 * std::f32::consts::PI } else { azs[i] };
                a_min = a_min.min(adj_az);
                a_max = a_max.max(adj_az);
                e_min = e_min.min(els[i]);
                e_max = e_max.max(els[i]);
            }
            let az_margin = ((a_max - a_min) * 0.1).max(self.az_res);
            let el_margin = ((e_max - e_min) * 0.1).max(self.el_res);
            (
                (a_min - az_margin).max(0.0),
                (a_max + az_margin).min(2.0 * std::f32::consts::PI),
                (e_min - el_margin).max(-std::f32::consts::FRAC_PI_2),
                (e_max + el_margin).min(std::f32::consts::FRAC_PI_2),
            )
        } else {
            // 无缠绕，直接用原始 [-π, π] 值
            let az_margin = ((az_max_raw - az_min_raw) * 0.1).max(self.az_res);
            let el_margin = {
                let mut e_min = f32::MAX;
                let mut e_max = f32::MIN;
                for &el in &els {
                    e_min = e_min.min(el);
                    e_max = e_max.max(el);
                }
                ((e_max - e_min) * 0.1).max(self.el_res)
            };
            (
                (az_min_raw - az_margin).max(-std::f32::consts::PI),
                (az_max_raw + az_margin).min(std::f32::consts::PI),
                {
                    let mut e_min = f32::MAX;
                    for &el in &els { e_min = e_min.min(el); }
                    (e_min - el_margin).max(-std::f32::consts::FRAC_PI_2)
                },
                {
                    let mut e_max = f32::MIN;
                    for &el in &els { e_max = e_max.max(el); }
                    (e_max + el_margin).min(std::f32::consts::FRAC_PI_2)
                },
            )
        };

        let az_range = az_max - az_min;
        let el_range = el_max - el_min;

        // 保护：避免分辨率过细导致网格爆炸
        if az_range < self.az_res || el_range < self.el_res {
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

        // ── 3. 投影到 range image ───────────────────────────────
        let mut img: Vec<Option<usize>> = vec![None; total];
        let mut img_r: Vec<f32> = vec![f32::MAX; total];

        for i in 0..n {
            let r = rs[i];
            // 缠绕模式下用调整后的方位角，保持与 FOV 空间一致
            let az = if wrap && azs[i] < 0.0 { azs[i] + 2.0 * std::f32::consts::PI } else { azs[i] };
            let el = els[i];
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

        // ── 4. 8-连通 CCL ───────────────────────────────────────
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
                        let dz = (p1[2] - p2[2]).abs();
                        if dz < self.threshold {
                            let d2 = (p1[0] - p2[0]).powi(2)
                                + (p1[1] - p2[1]).powi(2)
                                + (p1[2] - p2[2]).powi(2);
                            if d2 < self.threshold * self.threshold {
                                union(&mut parent, idx, nidx);
                            }
                        }
                    }
                }
                // 上邻
                if v + 1 < rows {
                    let nidx = u * rows + (v + 1);
                    if occupied[nidx] {
                        let p2 = lifra[img[nidx].unwrap()];
                        let dz = (p1[2] - p2[2]).abs();
                        if dz < self.threshold {
                            let d2 = (p1[0] - p2[0]).powi(2)
                                + (p1[1] - p2[1]).powi(2)
                                + (p1[2] - p2[2]).powi(2);
                            if d2 < self.threshold * self.threshold {
                                union(&mut parent, idx, nidx);
                            }
                        }
                    }
                }
                // 右上对角 (u+1, v+1)
                if u + 1 < cols && v + 1 < rows {
                    let nidx = (u + 1) * rows + (v + 1);
                    if occupied[nidx] {
                        let p2 = lifra[img[nidx].unwrap()];
                        let dz = (p1[2] - p2[2]).abs();
                        if dz < self.threshold {
                            let d2 = (p1[0] - p2[0]).powi(2)
                                + (p1[1] - p2[1]).powi(2)
                                + (p1[2] - p2[2]).powi(2);
                            if d2 < self.threshold * self.threshold {
                                union(&mut parent, idx, nidx);
                            }
                        }
                    }
                }
                // 右下对角 (u+1, v-1)
                if u + 1 < cols && v > 0 {
                    let nidx = (u + 1) * rows + (v - 1);
                    if occupied[nidx] {
                        let p2 = lifra[img[nidx].unwrap()];
                        let dz = (p1[2] - p2[2]).abs();
                        if dz < self.threshold {
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
        }

        // ── 5. 收集连通分量 ──────────────────────────────────────
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

        println!("range_image: FOV [{:.1}°~{:.1}°]×[{:.1}°~{:.1}°], {}×{}={} cells → {} clusters{}",
            az_min.to_degrees(), az_max.to_degrees(),
            el_min.to_degrees(), el_max.to_degrees(),
            cols, rows, total,
            objects.len(),
            if wrap { " (wrapped)" } else { "" });
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
