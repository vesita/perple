use super::WallPickStrategy;

/// XY 平面 RANSAC 线检测墙体提取 — perple 默认墙体识别方法。
///
/// # 设计理念
///
/// 室内墙面在垂直方向（Z）延伸，在 XY 平面投影呈直线状。
/// 利用这一几何特性，将 3D 墙体检测转化为 2D 线提取问题：
///
/// 1. **RANSAC 粗检**：在 XY 投影点集中随机采样点对，构造候选直线，
///    统计内点数。迭代多次取最佳候选。
/// 2. **TLS 最小二乘精化**：收集 RANSAC 内点 → 协方差特征分解 →
///    最小特征值方向即为精化法线。消除随机采样带来的抖动，
///    使 i50 迭代也能达到 i100 的质量。
/// 3. **Z 跨度验证**：候选线内点的 Z 跨度须 ≥ 1.0m。
///    排除地面、桌面、家具排列等水平结构误检。
/// 4. **顺序提取 + search_end 收缩**：每次提取一面墙后将其点移到前部；
///    Z-span 失败的候选点移到搜索区末尾排除，继续尝试次优线。
///    确保多面墙都能被检出，低矮障碍不阻塞后续提取。
///
/// # 确定性
///
/// 通过 `with_seed(42)` 使用 SplitMix64 确定性伪随机，相同输入
/// 总是产生相同输出。常用于基准测试和回归对比。
/// 不设种子时使用系统随机。
///
/// # 推荐参数
///
/// `XYRansacWall::with_params(0.05, 50, 30).with_seed(42)`
/// — distance=0.05m, 50 次迭代, 最少 30 个墙面点
pub struct XYRansacWall {
    /// 点到线距离阈值 (m)
    distance: f32,
    /// RANSAC 迭代次数
    iterations: usize,
    /// 最小墙面点数
    min_wall_pts: usize,
    /// 最大墙面数
    max_walls: usize,
    /// 墙面最小 Z 跨度 (m)，排除地面/桌面等水平面
    min_z_span: f32,
    /// RNG 种子（Some 时确定性，None 时系统随机）
    rng_seed: Option<u64>,
}

impl XYRansacWall {
    pub fn new() -> Self {
        Self {
            distance: 0.10,
            iterations: 100,
            min_wall_pts: 30,
            max_walls: 8,
            min_z_span: 1.0,
            rng_seed: None,
        }
    }

    pub fn with_params(distance: f32, iterations: usize, min_wall_pts: usize) -> Self {
        Self { distance, iterations, min_wall_pts, ..Self::new() }
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.rng_seed = Some(seed);
        self
    }
}

/// 在 XY 点集中找最佳直线，返回 (法线 nx, ny, d, 内点数)。
///
/// 两步：1) RANSAC 找候选  2) 最小二乘精化 → 消除随机采样敏感度
fn best_xy_line(
    points: &[[f32; 3]],
    distance: f32,
    iterations: usize,
    rng_seed: Option<u64>,
) -> Option<(f32, f32, f32, usize)> {
    let n = points.len();
    if n < 2 { return None; }

    // ── 1. RANSAC 找最佳随机采样对 ──
    let mut best_count = 0usize;
    let mut best_nx = 0.0f32;
    let mut best_ny = 0.0f32;
    let mut best_d = 0.0f32;

    for i in 0..iterations {
        let iter_seed = rng_seed.map(|s| s.wrapping_add(i as u64).wrapping_mul(0x9E3779B97F4A7C15));
        let sel = select_some_seeded(0, n, 2, iter_seed);
        let (x1, y1) = (points[sel[0]][0], points[sel[0]][1]);
        let (x2, y2) = (points[sel[1]][0], points[sel[1]][1]);

        let dx = x2 - x1;
        let dy = y2 - y1;
        let len = (dx * dx + dy * dy).sqrt();
        if len < 1e-6 { continue; }

        let nx = -dy / len;
        let ny = dx / len;
        let d = -(nx * x1 + ny * y1);

        let mut count = 0usize;
        for p in points {
            let dist = (nx * p[0] + ny * p[1] + d).abs();
            if dist < distance { count += 1; }
        }

        if count > best_count {
            best_count = count;
            best_nx = nx;
            best_ny = ny;
            best_d = d;
        }
    }

    if best_count == 0 { return None; }

    // ── 2. 最小二乘精化：用 RANSAC 内点重拟合 ──
    let (refined_nx, refined_ny, refined_d, refined_count) =
        refine_line_ls(points, best_nx, best_ny, best_d, distance);

    if refined_count == 0 {
        Some((best_nx, best_ny, best_d, best_count))
    } else {
        Some((refined_nx, refined_ny, refined_d, refined_count))
    }
}

/// 最小二乘精化：收集 RANSAC 内点 → TLS 拟合 → 重新计数。
fn refine_line_ls(
    points: &[[f32; 3]],
    init_nx: f32, init_ny: f32, init_d: f32,
    distance: f32,
) -> (f32, f32, f32, usize) {
    // 收集初始内点
    let mut inliers: Vec<usize> = Vec::new();
    for (i, p) in points.iter().enumerate() {
        let dist = (init_nx * p[0] + init_ny * p[1] + init_d).abs();
        if dist < distance { inliers.push(i); }
    }
    if inliers.len() < 2 { return (0.0, 0.0, 0.0, 0); }

    let nf = inliers.len() as f32;
    let mut cx = 0.0f32; let mut cy = 0.0f32;
    for &i in &inliers {
        cx += points[i][0]; cy += points[i][1];
    }
    cx /= nf; cy /= nf;

    // 2x2 协方差 → 最小特征值方向 = TLS 法线
    let mut cxx = 0.0f32; let mut cxy = 0.0f32; let mut cyy = 0.0f32;
    for &i in &inliers {
        let dx = points[i][0] - cx;
        let dy = points[i][1] - cy;
        cxx += dx * dx; cxy += dx * dy; cyy += dy * dy;
    }
    cxx /= nf; cxy /= nf; cyy /= nf;

    let trace = cxx + cyy;
    let det = cxx * cyy - cxy * cxy;
    let disc = (trace * trace - 4.0 * det).max(0.0).sqrt();
    let lambda_min = (trace - disc) * 0.5;

    let nx = cxy;
    let ny = lambda_min - cxx;
    let len = (nx * nx + ny * ny).sqrt();
    let (nx, ny) = if len > 1e-8 { (nx / len, ny / len) } else { (1.0, 0.0) };
    let d = -(nx * cx + ny * cy);

    // 重新计数
    let mut count = 0usize;
    for p in points {
        let dist = (nx * p[0] + ny * p[1] + d).abs();
        if dist < distance { count += 1; }
    }

    (nx, ny, d, count)
}

/// 支持种子的随机采样（与 select_some 同接口）。
///
/// 使用 SplitMix64 直接生成索引，O(count) 无需分配完整 Vec。
fn select_some_seeded(start: usize, end: usize, count: usize, seed: Option<u64>) -> Vec<usize> {
    if seed.is_none() {
        return crate::utils::random::select_some(start, end, count);
    }
    let n = end - start;
    if n <= count {
        return (start..end).collect();
    }
    let mut state = seed.unwrap();
    let mut result = Vec::with_capacity(count);
    while result.len() < count {
        state = state.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(0xBF58476D1CE4E5B9);
        let idx = start + ((state >> 32) as usize % n);
        if !result.contains(&idx) {
            result.push(idx);
        }
    }
    result
}

impl WallPickStrategy for XYRansacWall {
    fn strategy_name(&self) -> &'static str { "xy_ransac" }

    fn pick(&mut self, cloud: &mut [[f32; 3]]) -> (usize, Vec<[f32; 4]>) {
        let n = cloud.len();
        if n < self.min_wall_pts { return (0, Vec::new()); }

        let mut total_wall = 0usize;
        let mut search_end = n; // 动态收缩搜索区，排除 Z-span 失败的候选点
        let mut planes = Vec::new();

        for _ in 0..self.max_walls {
            if total_wall >= search_end { break; }
            let remaining = &cloud[total_wall..search_end];
            if remaining.len() < self.min_wall_pts { break; }

            let (nx, ny, d, _count) = match best_xy_line(remaining, self.distance, self.iterations, self.rng_seed) {
                Some(v) => v,
                None => break,
            };

            // 收集内点并验证 Z 跨度（索引相对于 total_wall）
            let mut inlier_rel: Vec<usize> = Vec::new();
            let mut z_min = f32::MAX;
            let mut z_max = f32::MIN;
            for (i, p) in remaining.iter().enumerate() {
                let dist = (nx * p[0] + ny * p[1] + d).abs();
                if dist < self.distance {
                    inlier_rel.push(i);
                    if p[2] < z_min { z_min = p[2]; }
                    if p[2] > z_max { z_max = p[2]; }
                }
            }

            if inlier_rel.len() < self.min_wall_pts { break; }

            if z_max - z_min < self.min_z_span {
                // Z 跨度不足 → 非墙面（家具排等），移到搜索区末尾排除
                for &rel_idx in &inlier_rel {
                    search_end -= 1;
                    cloud.swap(total_wall + rel_idx, search_end);
                }
                continue;
            }

            // 合格墙面：内点交换到前部
            let mut write = total_wall;
            for &rel_idx in &inlier_rel {
                cloud.swap(total_wall + rel_idx, write);
                write += 1;
            }
            total_wall = write;
            planes.push([nx, ny, 0.0, d]);
        }

        (total_wall, planes)
    }
}
