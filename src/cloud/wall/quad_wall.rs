use std::collections::{HashMap, VecDeque, HashSet};
use super::WallPickStrategy;

/// 四叉树 + 2D 法线墙体提取。
///
/// 思路：XY 哈希网格 → 连通域 BFS → 2D PCA 线状判别 + Z 跨度验证。
/// - 墙面在 XY 投影呈细长条状（λ_min / λ_max 小）
/// - 墙面 Z 跨度大（从地面到接近天花板）
/// - 障碍物（桌椅等）XY 呈块状、Z 跨度小
///
/// 相比 XYDBSCANWall 的改进：基于网格连通域而非点级 DBSCAN，
/// 更抗稀疏抖动，参数更少，速度更快。
pub struct QuadtreeWall {
    /// XY 网格边长（m），略大于点云平均间距以抗稀疏断裂
    cell_size: f32,
    /// 单格最小点数，低于此值的格子视为噪声
    min_points: usize,
    /// 最小墙面点数
    min_wall_pts: usize,
    /// 最大墙面数
    max_walls: usize,
    /// 墙面最小 Z 跨度（m）
    min_z_span: f32,
    /// XY PCA 细长比阈值：λ_min / λ_max < threshold → 墙面
    max_width_ratio: f32,
}

type CellKey = (i32, i32);

struct CellInfo {
    indices: Vec<usize>,
    z_min: f32,
    z_max: f32,
}

impl QuadtreeWall {
    pub fn new() -> Self {
        Self {
            cell_size: 0.10,
            min_points: 3,
            min_wall_pts: 30,
            max_walls: 8,
            min_z_span: 1.5,
            max_width_ratio: 0.20,
        }
    }

    pub fn with_params(cell_size: f32, min_points: usize, min_z_span: f32) -> Self {
        Self { cell_size, min_points, min_z_span, ..Self::new() }
    }

    pub fn with_width_ratio(mut self, ratio: f32) -> Self {
        self.max_width_ratio = ratio;
        self
    }
}

impl WallPickStrategy for QuadtreeWall {
    fn strategy_name(&self) -> &'static str { "quadtree" }

    fn pick(&mut self, cloud: &mut [[f32; 3]]) -> (usize, Vec<[f32; 4]>) {
        let n = cloud.len();
        if n < self.min_wall_pts { return (0, Vec::new()); }

        // ── 1. 构建 XY 哈希网格 ──
        let inv = 1.0 / self.cell_size;
        let mut grid: HashMap<CellKey, CellInfo> = HashMap::new();
        for (i, p) in cloud.iter().enumerate() {
            let key = ((p[0] * inv).floor() as i32, (p[1] * inv).floor() as i32);
            let cell = grid.entry(key).or_insert_with(|| CellInfo {
                indices: Vec::new(),
                z_min: f32::MAX,
                z_max: f32::MIN,
            });
            cell.indices.push(i);
            if p[2] < cell.z_min { cell.z_min = p[2]; }
            if p[2] > cell.z_max { cell.z_max = p[2]; }
        }

        // ── 2. 过滤噪声格 → 收集有效格 ──
        let valid: HashSet<CellKey> = grid.iter()
            .filter(|(_, c)| c.indices.len() >= self.min_points)
            .map(|(&k, _)| k)
            .collect();

        // ── 3. BFS 连通域提取（8 邻域） ──
        let mut visited = HashSet::new();
        let mut clusters: Vec<Vec<CellKey>> = Vec::new();

        for &key in &valid {
            if visited.contains(&key) { continue; }
            let mut component = Vec::new();
            let mut queue = VecDeque::new();
            queue.push_back(key);
            visited.insert(key);

            while let Some(cur) = queue.pop_front() {
                component.push(cur);
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
            clusters.push(component);
        }

        // ── 4. 每个连通域：2D PCA + Z 跨度 → 墙/障碍物分类 ──
        let mut walls: Vec<(Vec<usize>, [f32; 4])> = Vec::new();

        for component in &clusters {
            let mut all_indices = Vec::new();
            let mut z_min = f32::MAX;
            let mut z_max = f32::MIN;
            for key in component {
                if let Some(cell) = grid.get(key) {
                    all_indices.extend_from_slice(&cell.indices);
                    if cell.z_min < z_min { z_min = cell.z_min; }
                    if cell.z_max > z_max { z_max = cell.z_max; }
                }
            }

            if all_indices.len() < self.min_wall_pts { continue; }
            if z_max - z_min < self.min_z_span { continue; }

            // XY 中心
            let nf = all_indices.len() as f32;
            let cx: f32 = all_indices.iter().map(|&i| cloud[i][0]).sum::<f32>() / nf;
            let cy: f32 = all_indices.iter().map(|&i| cloud[i][1]).sum::<f32>() / nf;

            // 2x2 协方差矩阵
            let mut cxx = 0.0f32;
            let mut cxy = 0.0f32;
            let mut cyy = 0.0f32;
            for &i in &all_indices {
                let dx = cloud[i][0] - cx;
                let dy = cloud[i][1] - cy;
                cxx += dx * dx;
                cxy += dx * dy;
                cyy += dy * dy;
            }
            cxx /= nf;
            cxy /= nf;
            cyy /= nf;

            let trace = cxx + cyy;
            let det = cxx * cyy - cxy * cxy;
            let disc = (trace * trace - 4.0 * det).max(0.0).sqrt();
            let lambda_max = (trace + disc) * 0.5;
            let lambda_min = (trace - disc) * 0.5;

            if lambda_max < 1e-8 { continue; }
            if lambda_min / lambda_max >= self.max_width_ratio { continue; }

            // 2D 法线（最小特征值方向的垂直方向）
            let nx = cxy;
            let ny = lambda_min - cxx;
            let len = (nx * nx + ny * ny).sqrt();
            let (nx, ny) = if len > 1e-8 { (nx / len, ny / len) } else { (1.0, 0.0) };
            let d = -(nx * cx + ny * cy);

            walls.push((all_indices, [nx, ny, 0.0, d]));
        }

        walls.sort_by(|a, b| b.0.len().cmp(&a.0.len()));
        walls.truncate(self.max_walls);

        // ── 5. 原地重排：墙面点 → 前部，障碍物点 → 后部 ──
        let mut wall_set = HashSet::new();
        for (indices, _) in &walls {
            for &i in indices { wall_set.insert(i); }
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
