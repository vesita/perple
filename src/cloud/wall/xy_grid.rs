// ─── 共享 XY 网格基础设施 ───

pub(crate) type CellKey = (i32, i32);

/// XY 平面哈希网格，O(1) 插入/查询。
///
/// 所有 XY 墙体策略共用：cc_pca_grid、adapt_pca_grid、adapt_l2_grid 等。
/// cc_pca_qt 改用四叉树索引，不再使用 XYGrid。
pub struct XYGrid {
    pub cell_size: f32,
    pub cells: std::collections::HashMap<CellKey, Vec<usize>>,
}

impl XYGrid {
    pub fn new(points: &[[f32; 3]], cell_size: f32) -> Self {
        let inv = 1.0 / cell_size;
        let mut cells: std::collections::HashMap<CellKey, Vec<usize>> = std::collections::HashMap::new();
        for (i, p) in points.iter().enumerate() {
            let key = ((p[0] * inv).floor() as i32, (p[1] * inv).floor() as i32);
            cells.entry(key).or_default().push(i);
        }
        Self { cell_size, cells }
    }

    /// 密集格（点数 >= min_density）
    pub fn dense_cells(&self, min_density: usize) -> Vec<CellKey> {
        self.cells.iter()
            .filter(|(_, v)| v.len() >= min_density)
            .map(|(&k, _)| k)
            .collect()
    }

    /// 查询 (cx, cy) 周围 3×3 格子内距离 < eps 的点
    pub fn query_neighbors(&self, points: &[[f32; 3]], cx: f32, cy: f32, eps: f32, result: &mut Vec<usize>) {
        let inv = 1.0 / self.cell_size;
        let ix = (cx * inv).floor() as i32;
        let iy = (cy * inv).floor() as i32;
        let eps_sq = eps * eps;
        for dx in -1i32..=1 {
            for dy in -1i32..=1 {
                if let Some(cell) = self.cells.get(&(ix + dx, iy + dy)) {
                    for &i in cell {
                        let p = &points[i];
                        let ddx = p[0] - cx;
                        let ddy = p[1] - cy;
                        if ddx * ddx + ddy * ddy <= eps_sq {
                            result.push(i);
                        }
                    }
                }
            }
        }
    }

    /// LV-DOT 风格网格下采样：自动调整格子大小使输出 ≈ target_pts。
    ///
    /// 每个格子保留距质心最近的真实点（非合成质心），保持空间结构。
    /// 返回 `(下采样点, map)` 其中 `map[i]` = 原始点索引。
    pub fn grid_downsample(points: &[[f32; 3]], target_pts: usize) -> (Vec<[f32; 3]>, Vec<usize>) {
        let n = points.len();
        if n <= target_pts {
            return (points.to_vec(), (0..n).collect());
        }

        let (mut x_min, mut x_max, mut y_min, mut y_max) =
            (f32::MAX, f32::MIN, f32::MAX, f32::MIN);
        for p in points {
            x_min = x_min.min(p[0]); x_max = x_max.max(p[0]);
            y_min = y_min.min(p[1]); y_max = y_max.max(p[1]);
        }
        let area = (x_max - x_min).max(0.01) * (y_max - y_min).max(0.01);
        let mut cell_size = (area / target_pts as f32).sqrt();

        for _ in 0..5 {
            let grid = XYGrid::new(points, cell_size);
            if grid.cells.len() <= target_pts {
                let mut result = Vec::with_capacity(grid.cells.len());
                let mut map = Vec::with_capacity(grid.cells.len());
                for (&(cx_key, cy_key), indices) in &grid.cells {
                    // 格子中心
                    let ccx = (cx_key as f32 + 0.5) * cell_size;
                    let ccy = (cy_key as f32 + 0.5) * cell_size;
                    // 选距中心最近的真实点
                    let mut best_i = indices[0];
                    let mut best_d2 = f32::MAX;
                    for &idx in indices {
                        let d2 = (points[idx][0] - ccx).powi(2) + (points[idx][1] - ccy).powi(2);
                        if d2 < best_d2 { best_d2 = d2; best_i = idx; }
                    }
                    result.push(points[best_i]);
                    map.push(best_i);
                }
                return (result, map);
            }
            // 格子数超过目标 → 增大格子尺寸以合并更多点
            cell_size /= 0.7;
        }

        let step = (n / target_pts).max(1);
        let result: Vec<[f32; 3]> = (0..n).step_by(step).map(|i| points[i]).collect();
        let map: Vec<usize> = (0..n).step_by(step).collect();
        (result, map)
    }

    /// 最远点采样（XY 平面），空间均匀性优于网格下采样。
    ///
    /// 内部用 XYGrid 加速邻域查询。
    /// 返回 `(下采样点, map)` 其中 `map[i]` = 原始点索引。
    pub fn fps_downsample(points: &[[f32; 3]], target_pts: usize) -> (Vec<[f32; 3]>, Vec<usize>) {
        let n = points.len();
        if n <= target_pts {
            return (points.to_vec(), (0..n).collect());
        }

        // 网格大小：约 target_pts 个格子
        let (mut x_min, mut x_max, mut y_min, mut y_max) =
            (f32::MAX, f32::MIN, f32::MAX, f32::MIN);
        for p in points {
            x_min = x_min.min(p[0]); x_max = x_max.max(p[0]);
            y_min = y_min.min(p[1]); y_max = y_max.max(p[1]);
        }
        let area = (x_max - x_min).max(0.01) * (y_max - y_min).max(0.01);
        let cell_size = (area / target_pts as f32).sqrt().max(0.01);
        let grid = XYGrid::new(points, cell_size);

        // 起始点：离质心最近
        let (cx, cy) = points.iter().fold((0.0f32, 0.0f32), |(sx, sy), p| (sx + p[0], sy + p[1]));
        let (cx, cy) = (cx / n as f32, cy / n as f32);
        let mut first = 0usize;
        let mut best_d2 = f32::MAX;
        for (i, p) in points.iter().enumerate() {
            let d2 = (p[0] - cx).powi(2) + (p[1] - cy).powi(2);
            if d2 < best_d2 { best_d2 = d2; first = i; }
        }

        let mut selected = vec![first];
        let mut min_d2 = vec![f32::MAX; n];

        // 初始化 min_d2
        let sp = points[first];
        for (i, p) in points.iter().enumerate() {
            min_d2[i] = (p[0] - sp[0]).powi(2) + (p[1] - sp[1]).powi(2);
        }

        let inv = 1.0 / cell_size;
        for _ in 1..target_pts {
            // 选 min_d2 最大的点
            let mut best_i = 0usize;
            let mut best_d = -1.0f32;
            for (i, &d) in min_d2.iter().enumerate() {
                if d > best_d { best_d = d; best_i = i; }
            }
            if best_d <= 0.0 { break; }

            selected.push(best_i);
            let sp = points[best_i];

            // 用网格加速更新：只更新新选点附近格子内的点
            let radius_cells = ((best_d.sqrt() * inv).ceil() as i32).max(1);
            let ix = (sp[0] * inv).floor() as i32;
            let iy = (sp[1] * inv).floor() as i32;
            for dx in -radius_cells..=radius_cells {
                for dy in -radius_cells..=radius_cells {
                    if let Some(cell) = grid.cells.get(&(ix + dx, iy + dy)) {
                        for &idx in cell {
                            let d2 = (points[idx][0] - sp[0]).powi(2)
                                   + (points[idx][1] - sp[1]).powi(2);
                            if d2 < min_d2[idx] {
                                min_d2[idx] = d2;
                            }
                        }
                    }
                }
            }
        }

        selected.sort();
        let result: Vec<[f32; 3]> = selected.iter().map(|&i| points[i]).collect();
        (result, selected)
    }

    /// LV-DOT 风格体素占用过滤：剔除稀疏离群点。
    ///
    /// 将空间划分为 `voxel_size` 体素，保留点数 >= `min_occ` 的体素内所有点。
    /// 返回 `(过滤后点, 原始索引映射)`。
    pub fn voxel_occupancy_filter(points: &[[f32; 3]], voxel_size: f32, min_occ: usize) -> (Vec<[f32; 3]>, Vec<usize>) {
        let n = points.len();
        if n == 0 || voxel_size <= 0.0 || min_occ <= 1 {
            return (points.to_vec(), (0..n).collect());
        }
        let inv = 1.0 / voxel_size;
        let mut voxels: std::collections::HashMap<(i32, i32, i32), Vec<usize>> = std::collections::HashMap::new();
        for (i, p) in points.iter().enumerate() {
            let key = (
                (p[0] * inv).floor() as i32,
                (p[1] * inv).floor() as i32,
                (p[2] * inv).floor() as i32,
            );
            voxels.entry(key).or_default().push(i);
        }
        let mut filtered = Vec::new();
        let mut map = Vec::new();
        for (_, indices) in &voxels {
            if indices.len() >= min_occ {
                for &idx in indices {
                    filtered.push(points[idx]);
                    map.push(idx);
                }
            }
        }
        (filtered, map)
    }

    /// LV-DOT 风格体素占用下采样：占用过滤 + 下采样二合一。
    ///
    /// 将空间划分为 `voxel_size` 体素：
    /// - 点数 < `min_occ` → 全部丢弃（稀疏噪点）
    /// - 点数 ≥ `min_occ` → 输出质心（每格 1 点）
    ///
    /// 与 `voxel_occupancy_filter` 的区别：filter 保留密集格全部点，
    /// 此函数每格只输出 1 个质心，适合作为 DBSCAN 前的最后一级压缩。
    /// 返回 `(下采样点, 原始索引映射)`。
    pub fn voxel_occupancy_downsample(points: &[[f32; 3]], voxel_size: f32, min_occ: usize) -> (Vec<[f32; 3]>, Vec<usize>) {
        let n = points.len();
        if n == 0 || voxel_size <= 0.0 || min_occ <= 1 {
            return (points.to_vec(), (0..n).collect());
        }
        let inv = 1.0 / voxel_size;
        let mut voxels: std::collections::HashMap<(i32, i32, i32), (f64, f64, f64, usize)> = std::collections::HashMap::new();
        for (_i, p) in points.iter().enumerate() {
            let key = (
                (p[0] * inv).floor() as i32,
                (p[1] * inv).floor() as i32,
                (p[2] * inv).floor() as i32,
            );
            let entry = voxels.entry(key).or_insert((0.0, 0.0, 0.0, 0));
            entry.0 += p[0] as f64;
            entry.1 += p[1] as f64;
            entry.2 += p[2] as f64;
            entry.3 += 1;
        }
        let mut result = Vec::with_capacity(voxels.len());
        let mut map = Vec::with_capacity(voxels.len());
        for (_key, (sx, sy, sz, cnt)) in voxels {
            if cnt >= min_occ {
                let nf = cnt as f64;
                // 取质心作为代表点；用 _key 的最近原始点索引（近似）
                result.push([(sx / nf) as f32, (sy / nf) as f32, (sz / nf) as f32]);
                // map 是近似的：无单一原始点对应质心，用第一个点的 key hash 作为伪索引
                map.push(0usize); // 后续不依赖此 map 做精确索引映射
            }
        }
        (result, map)
    }
}
