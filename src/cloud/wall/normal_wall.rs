use std::collections::{VecDeque, HashSet};

use super::{WallPickStrategy, XYGrid};

/// 法线驱动墙体提取策略。
///
/// 核心思路（3+1 步）：
/// 0. [可选] 下采样 — 网格/FPS 压缩到目标点数，加速后续步骤
/// 1. XY 连通域聚类 — 捕捉候选目标（密度无关，仅连通性）
/// 2. 距离丢弃        — 远距离簇标记为 "far"，不进法线检测
/// 3. Z 分层切分+法线检测  — Z 断崖拆分子簇 → 3D PCA 取 |n_z| < threshold 判定为墙体
///
/// Z 分层切分解决墙体旁障碍物粘连问题：同一 XY 连通域内，墙体和紧邻桌椅
/// 因 XY 连续被合并，但在 Z 方向存在间断（桌面下方空隙）。沿 Z 断崖拆开后，
/// 墙体子簇保持竖直法线，障碍物子簇被独立检测。
pub struct NormalWall {
    /// XY 网格边长（m）
    cell_size: f32,
    /// 连通域最小点数
    min_pts: usize,
    /// 远距离阈值（m），簇质心距离超过此值 → 丢弃
    pub far_distance: f32,
    /// 法线 Z 分量阈值，|n_z| < 此值 → 墙面
    pub normal_z_threshold: f32,
    /// Z 方向断崖间隙阈值（m），连续两点 Z 差 > 此值视为断崖
    pub z_gap_threshold: f32,
    /// Z 分层后子簇最小 Z 跨度（m），低于此值的子簇不进法线检测
    pub min_z_span: f32,
    /// 最大墙面数
    max_walls: usize,
    /// 下采样目标点数（None = 不下采样）
    pub downsample_target: Option<usize>,
    /// 最近一帧的远距离簇数（诊断用）
    pub last_far_clusters: usize,
    /// 最近一帧的丢弃簇数（法线不满足）
    pub last_rejected_clusters: usize,
    /// 最近一帧的 Z 分层拆分数（诊断用）
    pub last_z_splits: usize,
}

impl NormalWall {
    pub fn new() -> Self {
        Self {
            cell_size: 0.15,
            min_pts: 10,
            far_distance: 30.0,
            normal_z_threshold: 0.17, // cos(80°) 近似
            z_gap_threshold: 0.30,    // Z 方向 30cm 空隙视为断崖
            min_z_span: 0.80,          // 子簇至少 80cm 高才算墙体候选
            max_walls: 8,
            downsample_target: None,
            last_far_clusters: 0,
            last_rejected_clusters: 0,
            last_z_splits: 0,
        }
    }

    pub fn with_params(cell_size: f32, min_pts: usize, far_distance: f32) -> Self {
        Self { cell_size, min_pts, far_distance, ..Self::new() }
    }

    pub fn with_normal_threshold(mut self, threshold: f32) -> Self {
        self.normal_z_threshold = threshold;
        self
    }

    pub fn with_downsample_target(mut self, n: usize) -> Self {
        self.downsample_target = Some(n);
        self
    }

    pub fn with_z_split(mut self, z_gap: f32, min_z_span: f32) -> Self {
        self.z_gap_threshold = z_gap;
        self.min_z_span = min_z_span;
        self
    }
}

impl Default for NormalWall {
    fn default() -> Self { Self::new() }
}

type CellKey = (i32, i32);

impl WallPickStrategy for NormalWall {
    fn strategy_name(&self) -> &'static str { "normal_wall" }

    fn pick(&mut self, cloud: &mut [[f32; 3]]) -> (usize, Vec<[f32; 4]>) {
        let n = cloud.len();
        self.last_far_clusters = 0;
        self.last_rejected_clusters = 0;
        self.last_z_splits = 0;

        if n < self.min_pts { return (0, Vec::new()); }

        // ── 0. 可选下采样 ──
        let (work_points, idx_map) = match self.downsample_target {
            Some(target) if target < n => {
                let (pts, map) = XYGrid::grid_downsample(cloud, target);
                (pts, Some(map))
            }
            _ => (cloud.to_vec(), None),
        };

        // ── 1. XY 网格化 ──
        let grid = XYGrid::new(&work_points, self.cell_size);

        // 所有非空格 → 有效格集合（无密度阈值，仅需 > 0）
        let valid: HashSet<CellKey> = grid.cells.keys().copied().collect();

        // ── 2. BFS 连通域提取（8 邻域） ──
        let mut visited: HashSet<CellKey> = HashSet::new();
        let mut clusters: Vec<Vec<usize>> = Vec::new();

        for &key in &valid {
            if visited.contains(&key) { continue; }
            let mut indices = Vec::new();
            let mut queue = VecDeque::new();
            queue.push_back(key);
            visited.insert(key);

            while let Some(cur) = queue.pop_front() {
                if let Some(cell) = grid.cells.get(&cur) {
                    indices.extend_from_slice(cell);
                }
                for dx in -1i32..=1 {
                    for dy in -1i32..=1 {
                        if dx == 0 && dy == 0 { continue; }
                        let nbr = (cur.0 + dx, cur.1 + dy);
                        if valid.contains(&nbr) && !visited.contains(&nbr) {
                            visited.insert(nbr);
                            queue.push_back(nbr);
                        }
                    }
                }
            }
            if indices.len() >= self.min_pts {
                clusters.push(indices);
            }
        }

        // ── 3. 距离丢弃 + Z 分层切分 + 法线检测 ──
        let far_d2 = self.far_distance * self.far_distance;
        let mut walls: Vec<(Vec<usize>, [f32; 4])> = Vec::new();

        for (ci, indices) in clusters.iter().enumerate() {
            // 计算质心距离
            let nf = indices.len() as f32;
            let cx: f32 = indices.iter().map(|&i| work_points[i][0]).sum::<f32>() / nf;
            let cy: f32 = indices.iter().map(|&i| work_points[i][1]).sum::<f32>() / nf;
            let dist2 = cx * cx + cy * cy;

            // 远距离 → 丢弃
            if dist2 > far_d2 {
                self.last_far_clusters += 1;
                log::debug!(
                    "normal_wall cluster {} FAR: pts={} dist={:.1}m > {:.1}m",
                    ci, indices.len(), dist2.sqrt(), self.far_distance
                );
                continue;
            }

            // 近距 → 尝试法线检测，失败则 Z 分层后再试
            let sub_clusters = classify_or_zsplit(
                ci, indices, &work_points,
                self.min_pts, self.normal_z_threshold,
                self.z_gap_threshold, self.min_z_span,
                &mut self.last_z_splits, &mut self.last_rejected_clusters,
                &mut walls,
            );
            // 子簇中满足法线条件的已推入 walls，不满足的静默丢弃（障碍物）
            if !sub_clusters.is_empty() {
                log::debug!(
                    "normal_wall cluster {} Z-SPLIT: {} sub-clusters from {} pts",
                    ci, sub_clusters.len(), indices.len()
                );
            }
        }

        walls.sort_by(|a, b| b.0.len().cmp(&a.0.len()));
        walls.truncate(self.max_walls);

        // ── 4. 映射回原始点 + 原地重排 ──
        let mut wall_set: HashSet<usize> = HashSet::new();
        match &idx_map {
            Some(map) => {
                // 下采样模式：将采样点索引映射回原始点索引
                for (cluster, _) in &walls {
                    for &si in cluster {
                        wall_set.insert(map[si]);
                    }
                }
            }
            None => {
                // 无下采样：索引即原始点
                for (cluster, _) in &walls {
                    for &i in cluster { wall_set.insert(i); }
                }
            }
        }

        let mut write = 0usize;
        for read in 0..n {
            if wall_set.contains(&read) {
                cloud.swap(read, write);
                write += 1;
            }
        }

        let planes: Vec<[f32; 4]> = walls.into_iter().map(|(_, plane)| plane).collect();
        (write, planes)
    }
}

/// 对单个 XY 连通域尝试法线分类，失败则 Z 分层切分后重试。
///
/// 返回被 Z 拆分的子簇数量（0 = 未拆分，直接分类成功或失败）。
/// 成功的子簇直接推入 `walls`。
fn classify_or_zsplit(
    ci: usize,
    indices: &[usize],
    points: &[[f32; 3]],
    min_pts: usize,
    normal_z_threshold: f32,
    z_gap: f32,
    min_z_span: f32,
    last_z_splits: &mut usize,
    last_rejected: &mut usize,
    walls: &mut Vec<(Vec<usize>, [f32; 4])>,
) -> Vec<Vec<usize>> {
    // 先尝试整簇法线检测
    if let Some((normal, d)) = fit_plane_3d(indices, points) {
        if normal[2].abs() < normal_z_threshold {
            log::debug!(
                "normal_wall cluster {} ACCEPT: pts={} |n_z|={:.2} < {:.2}",
                ci, indices.len(), normal[2].abs(), normal_z_threshold
            );
            walls.push((indices.to_vec(), [normal[0], normal[1], normal[2], d]));
            return Vec::new();
        }
    }

    // 整簇法线不满足 → Z 分层切分
    if indices.len() < min_pts * 2 {
        *last_rejected += 1;
        return Vec::new();
    }

    let sub = z_split(indices, points, z_gap, min_pts, min_z_span);
    if sub.len() < 2 {
        *last_rejected += 1;
        return Vec::new();
    }

    *last_z_splits += sub.len();

    // 对每个子簇独立做法线检测
    for (si, sub_indices) in sub.iter().enumerate() {
        if let Some((normal, d)) = fit_plane_3d(sub_indices, points) {
            if normal[2].abs() < normal_z_threshold {
                log::debug!(
                    "normal_wall cluster {} sub{} ACCEPT: pts={} |n_z|={:.2} < {:.2}",
                    ci, si, sub_indices.len(), normal[2].abs(), normal_z_threshold
                );
                walls.push((sub_indices.clone(), [normal[0], normal[1], normal[2], d]));
            } else {
                *last_rejected += 1;
                log::debug!(
                    "normal_wall cluster {} sub{} REJECT: pts={} |n_z|={:.2} >= {:.2}",
                    ci, si, sub_indices.len(), normal[2].abs(), normal_z_threshold
                );
            }
        }
    }
    sub
}

/// Z 方向断崖切分：对排序后的点沿 Z 轴扫描，间隙 > z_gap 即拆开。
///
/// 每个子簇需满足 >= min_pts 个点且 Z 跨度 >= min_z_span。
fn z_split(
    indices: &[usize],
    points: &[[f32; 3]],
    z_gap: f32,
    min_pts: usize,
    min_z_span: f32,
) -> Vec<Vec<usize>> {
    let mut sorted: Vec<usize> = indices.to_vec();
    sorted.sort_by(|a, b| points[*a][2].partial_cmp(&points[*b][2]).unwrap());

    let mut segments = Vec::new();
    let mut seg_start = 0usize;
    for i in 1..sorted.len() {
        let gap = points[sorted[i]][2] - points[sorted[i - 1]][2];
        if gap > z_gap {
            let seg = &sorted[seg_start..i];
            let z_min = points[seg[0]][2];
            let z_max = points[seg[seg.len() - 1]][2];
            if seg.len() >= min_pts && z_max - z_min >= min_z_span {
                segments.push(seg.to_vec());
            }
            seg_start = i;
        }
    }
    // 最后一段
    let last_seg = &sorted[seg_start..];
    if last_seg.len() >= min_pts {
        let z_min = points[last_seg[0]][2];
        let z_max = points[last_seg[last_seg.len() - 1]][2];
        if z_max - z_min >= min_z_span {
            segments.push(last_seg.to_vec());
        }
    }
    segments
}

/// 3D PCA 拟合平面，返回 (normal, d)。
fn fit_plane_3d(indices: &[usize], points: &[[f32; 3]]) -> Option<([f32; 3], f32)> {
    let n = indices.len();
    if n < 3 { return None; }
    let nf = n as f32;
    let mut cx = 0.0f32;
    let mut cy = 0.0f32;
    let mut cz = 0.0f32;
    for &i in indices {
        let p = &points[i];
        cx += p[0]; cy += p[1]; cz += p[2];
    }
    cx /= nf; cy /= nf; cz /= nf;

    let mut cov = nalgebra::Matrix3::zeros();
    for &i in indices {
        let p = &points[i];
        let dx = p[0] - cx;
        let dy = p[1] - cy;
        let dz = p[2] - cz;
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
        if eig.eigenvalues[i] < min_val {
            min_val = eig.eigenvalues[i];
            min_idx = i;
        }
    }
    let nv = eig.eigenvectors.column(min_idx);
    let normal = [nv[0], nv[1], nv[2]];
    let d = -(normal[0] * cx + normal[1] * cy + normal[2] * cz);
    Some((normal, d))
}
