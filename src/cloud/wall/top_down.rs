use super::{WallPickStrategy, XYGrid, CellKey};

/// 俯视聚类墙体提取。
///
/// 思路：投影到 XY 平面 → 网格化 → 密集竖直柱体检测 → 相邻密集格合并为墙面。
/// 墙面在 XY 投影中呈线状密集分布，Z 轴信息主要用于竖直性验证。
pub struct TopDownCluster {
    /// XY 网格边长（m）
    cell_size: f32,
    /// 单格最小点数（密集阈值）
    min_density: usize,
    /// 合并相邻密集格的连通距离（格数）
    merge_dist: usize,
    /// 最小墙面点数
    min_wall_pts: usize,
    /// 最大墙面数
    max_walls: usize,
    /// XY 法线校验：λ_min / λ_max < threshold → 细长 → 墙面
    max_width_ratio: f32,
}

impl TopDownCluster {
    pub fn new() -> Self {
        Self {
            cell_size: 0.05,
            min_density: 5,
            merge_dist: 2,
            min_wall_pts: 30,
            max_walls: 8,
            max_width_ratio: 0.3,
        }
    }

    pub fn with_params(cell_size: f32, min_density: usize, merge_dist: usize) -> Self {
        Self { cell_size, min_density, merge_dist, ..Self::new() }
    }

    pub fn with_width_ratio(mut self, ratio: f32) -> Self {
        self.max_width_ratio = ratio;
        self
    }
}

/// BFS 合并相邻密集格
fn merge_adjacent_dense(
    dense: &std::collections::HashSet<CellKey>,
    merge_dist: usize,
) -> Vec<Vec<CellKey>> {
    let mut visited = std::collections::HashSet::new();
    let mut clusters = Vec::new();

    for &key in dense {
        if visited.contains(&key) { continue; }

        let mut cluster = Vec::new();
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(key);
        visited.insert(key);

        while let Some(cur) = queue.pop_front() {
            cluster.push(cur);
            // 检查 merge_dist 范围内的邻居
            for dx in -(merge_dist as i32)..=(merge_dist as i32) {
                for dy in -(merge_dist as i32)..=(merge_dist as i32) {
                    if dx == 0 && dy == 0 { continue; }
                    let nbr = (cur.0 + dx, cur.1 + dy);
                    if dense.contains(&nbr) && !visited.contains(&nbr) {
                        visited.insert(nbr);
                        queue.push_back(nbr);
                    }
                }
            }
        }

        clusters.push(cluster);
    }
    clusters
}

/// 2D 法线校验：XY 平面 PCA 检查细长程度。
///
/// 墙面在 XY 投影中应呈细长条状（λ_min ≪ λ_max），而非圆胖块状。
/// 返回 (is_wall, [nx, ny], ratio)。
fn check_xy_normal(points: &[[f32; 3]], max_width_ratio: f32) -> (bool, [f32; 2], f32) {
    let n = points.len();
    if n < 3 { return (false, [0.0, 0.0], 0.0); }

    let nf = n as f32;
    let cx: f32 = points.iter().map(|p| p[0]).sum::<f32>() / nf;
    let cy: f32 = points.iter().map(|p| p[1]).sum::<f32>() / nf;

    // 2x2 协方差矩阵
    let mut cxx = 0.0f32; let mut cxy = 0.0f32; let mut cyy = 0.0f32;
    for p in points {
        let dx = p[0] - cx; let dy = p[1] - cy;
        cxx += dx * dx; cxy += dx * dy; cyy += dy * dy;
    }
    cxx /= nf; cxy /= nf; cyy /= nf;

    // 特征值：λ = (trace ± sqrt(trace² - 4det)) / 2
    let trace = cxx + cyy;
    let det = cxx * cyy - cxy * cxy;
    let disc = (trace * trace - 4.0 * det).max(0.0).sqrt();
    let lambda_max = (trace + disc) * 0.5;
    let lambda_min = (trace - disc) * 0.5;

    // 法线方向对应最小特征值：(cxx - λ, cxy) 的垂直方向
    let nx = cxy;
    let ny = lambda_min - cxx;
    let len = (nx * nx + ny * ny).sqrt();
    let normal = if len > 1e-8 { [nx / len, ny / len] } else { [1.0, 0.0] };

    // 细长程度检查：λ_min / λ_max < threshold → 细长 → 墙面
    if lambda_max < 1e-8 { return (false, normal, 0.0); }
    let ratio = lambda_min / lambda_max;

    (ratio < max_width_ratio, normal, ratio)
}

impl WallPickStrategy for TopDownCluster {
    fn strategy_name(&self) -> &'static str { "top_down" }

    fn pick(&mut self, cloud: &mut [[f32; 3]]) -> (usize, Vec<[f32; 4]>) {
        let n = cloud.len();
        if n < 10 { return (0, Vec::new()); }

        // 1. XY 网格化
        let grid = XYGrid::new(cloud, self.cell_size);
        let dense_keys = grid.dense_cells(self.min_density);

        if dense_keys.is_empty() { return (0, Vec::new()); }

        // 2. BFS 合并相邻密集格
        let dense_set: std::collections::HashSet<CellKey> = dense_keys.into_iter().collect();
        let cell_clusters = merge_adjacent_dense(&dense_set, self.merge_dist);

        // 3. 展开为原始点 + 竖直性验证
        let mut walls: Vec<(Vec<usize>, [f32; 4])> = Vec::new();

        for (ci, cell_cluster) in cell_clusters.iter().enumerate() {
            let mut all_indices = Vec::new();
            for key in cell_cluster {
                if let Some(indices) = grid.cells.get(key) {
                    all_indices.extend_from_slice(indices);
                }
            }

            if all_indices.len() < self.min_wall_pts {
                if all_indices.len() >= 10 {
                    log::debug!("td cluster {} REJECT: pts={} < min_wall_pts={}", ci, all_indices.len(), self.min_wall_pts);
                }
                continue;
            }

            // 2D 法线校验：XY 投影应呈细长条状
            let pts: Vec<[f32; 3]> = all_indices.iter().map(|&i| cloud[i]).collect();
            let (is_wall, _xy_normal, ratio) = check_xy_normal(&pts, self.max_width_ratio);
            if !is_wall {
                log::debug!("td cluster {} REJECT: pts={} ratio={:.4} >= max_width_ratio={:.2} cells={}", ci, all_indices.len(), ratio, self.max_width_ratio, cell_cluster.len());
                continue;
            }

            // PCA 拟合平面（在 3D 上）
            let plane = fit_plane_3d(&pts);
            let (normal, d) = match plane {
                Some(v) => v,
                None => {
                    log::debug!("td cluster {} REJECT: pts={} fit_plane_3d failed", ci, all_indices.len());
                    continue;
                }
            };

            // 竖直性：|nz| 小 → 墙
            if normal[2].abs() > 0.3 {
                log::debug!("td cluster {} REJECT: pts={} nz={:.2} > 0.3", ci, all_indices.len(), normal[2]);
                continue;
            }
            log::debug!("td cluster {} ACCEPT: pts={} ratio={:.4} nz={:.2} cells={}", ci, all_indices.len(), ratio, normal[2], cell_cluster.len());

            walls.push((all_indices, [normal[0], normal[1], normal[2], d]));
        }

        walls.sort_by(|a, b| b.0.len().cmp(&a.0.len()));
        walls.truncate(self.max_walls);

        // 4. 原地重排
        let mut wall_set = std::collections::HashSet::new();
        for (cluster, _) in &walls {
            for &i in cluster { wall_set.insert(i); }
        }
        let mut write = 0usize;
        for read in 0..n {
            if wall_set.contains(&read) {
                cloud.swap(read, write);
                write += 1;
            }
        }

        let planes: Vec<[f32; 4]> = walls.into_iter().map(|(_, p)| p).collect();
        (write, planes)
    }
}

fn fit_plane_3d(points: &[[f32; 3]]) -> Option<([f32; 3], f32)> {
    use nalgebra::Matrix3;
    let n = points.len();
    if n < 3 { return None; }
    let nf = n as f32;
    let mut cx = 0.0f32; let mut cy = 0.0f32; let mut cz = 0.0f32;
    for p in points { cx += p[0]; cy += p[1]; cz += p[2]; }
    cx /= nf; cy /= nf; cz /= nf;

    let mut cov = Matrix3::zeros();
    for p in points {
        let dx = p[0] - cx; let dy = p[1] - cy; let dz = p[2] - cz;
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
        if eig.eigenvalues[i] < min_val { min_val = eig.eigenvalues[i]; min_idx = i; }
    }
    let nv = eig.eigenvectors.column(min_idx);
    let normal = [nv[0], nv[1], nv[2]];
    let d = -(normal[0] * cx + normal[1] * cy + normal[2] * cz);
    Some((normal, d))
}
