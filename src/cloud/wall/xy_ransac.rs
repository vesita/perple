use super::WallPickStrategy;
use crate::utils::random::select_some;

/// XY 平面 RANSAC 线检测墙体提取。
///
/// 墙面在 XY 投影中呈直线状 → 用 2D 线 RANSAC 直接检测。
/// 只需 2 个采样点（vs 3D 平面需 3 个），更稳定更快。
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
}

impl XYRansacWall {
    pub fn new() -> Self {
        Self {
            distance: 0.10,
            iterations: 100,
            min_wall_pts: 30,
            max_walls: 8,
            min_z_span: 1.0,
        }
    }

    pub fn with_params(distance: f32, iterations: usize, min_wall_pts: usize) -> Self {
        Self { distance, iterations, min_wall_pts, ..Self::new() }
    }
}

/// 在 XY 点集中找最佳直线，返回 (法线 nx, ny, d, 内点数)。
fn best_xy_line(
    points: &[[f32; 3]],
    distance: f32,
    iterations: usize,
) -> Option<(f32, f32, f32, usize)> {
    let n = points.len();
    if n < 2 { return None; }

    let mut best_count = 0usize;
    let mut best_nx = 0.0f32;
    let mut best_ny = 0.0f32;
    let mut best_d = 0.0f32;

    for _ in 0..iterations {
        let sel = select_some(0, n, 2);
        let (x1, y1) = (points[sel[0]][0], points[sel[0]][1]);
        let (x2, y2) = (points[sel[1]][0], points[sel[1]][1]);

        let dx = x2 - x1;
        let dy = y2 - y1;
        let len = (dx * dx + dy * dy).sqrt();
        if len < 1e-6 { continue; }

        // 法线方向：垂直于线方向
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

    if best_count == 0 { None } else { Some((best_nx, best_ny, best_d, best_count)) }
}

impl WallPickStrategy for XYRansacWall {
    fn strategy_name(&self) -> &'static str { "xy_ransac" }

    fn pick(&mut self, cloud: &mut [[f32; 3]]) -> (usize, Vec<[f32; 4]>) {
        let n = cloud.len();
        if n < self.min_wall_pts { return (0, Vec::new()); }

        let mut total_wall = 0usize;
        let mut planes = Vec::new();

        for _ in 0..self.max_walls {
            let remaining = &cloud[total_wall..];
            if remaining.len() < self.min_wall_pts { break; }

            let (nx, ny, d, _count) = match best_xy_line(remaining, self.distance, self.iterations) {
                Some(v) => v,
                None => break,
            };

            // 收集内点并验证 Z 跨度
            let mut inlier_indices = Vec::new();
            let mut z_min = f32::MAX;
            let mut z_max = f32::MIN;
            for (i, p) in remaining.iter().enumerate() {
                let dist = (nx * p[0] + ny * p[1] + d).abs();
                if dist < self.distance {
                    inlier_indices.push(total_wall + i);
                    if p[2] < z_min { z_min = p[2]; }
                    if p[2] > z_max { z_max = p[2]; }
                }
            }

            if inlier_indices.len() < self.min_wall_pts { break; }
            if z_max - z_min < self.min_z_span { break; }

            // 将内点交换到前部
            let mut write = total_wall;
            for &idx in &inlier_indices {
                cloud.swap(idx, write);
                write += 1;
            }
            total_wall = write;
            planes.push([nx, ny, 0.0, d]);
        }

        (total_wall, planes)
    }
}
