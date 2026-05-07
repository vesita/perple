use nalgebra::Matrix3;
use super::WallPickStrategy;

/// 迭代 SVD 平面拟合墙体提取。
///
/// 每轮拟合主平面 → 收集内点 → 移除 → 重复。无随机采样，确定性输出。
pub struct SequentialFit {
    /// 内点距离阈值（m）
    distance: f32,
    /// 竖直性约束：|nz| < threshold
    normal_thresh: f32,
    /// 最小墙面点数
    min_wall_pts: usize,
    /// 最大墙面数
    max_walls: usize,
    /// 最大迭代轮数
    max_iters: usize,
}

impl SequentialFit {
    pub fn new() -> Self {
        Self {
            distance: 0.15,
            normal_thresh: 0.3,
            min_wall_pts: 30,
            max_walls: 5,
            max_iters: 20,
        }
    }

    pub fn with_params(distance: f32, normal_thresh: f32, max_walls: usize) -> Self {
        Self { distance, normal_thresh, max_walls, ..Self::new() }
    }
}

/// SVD 拟合平面，返回 (normal, d)
fn fit_plane_svd(points: &[[f32; 3]]) -> Option<([f32; 3], f32)> {
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

impl WallPickStrategy for SequentialFit {
    fn strategy_name(&self) -> &'static str { "seq_fit" }

    fn pick(&mut self, cloud: &mut [[f32; 3]]) -> (usize, Vec<[f32; 4]>) {
        let n = cloud.len();
        if n < 10 { return (0, Vec::new()); }

        // used[i] = true 表示已处理（墙面或已跳过的非竖直平面内点）
        let mut used = vec![false; n];
        let mut is_wall = vec![false; n];
        let mut planes = Vec::new();

        for _ in 0..self.max_iters {
            if planes.len() >= self.max_walls { break; }

            // 收集剩余点
            let pts: Vec<[f32; 3]> = (0..n)
                .filter(|&i| !used[i])
                .map(|i| cloud[i])
                .collect();
            if pts.len() < self.min_wall_pts { break; }

            let (normal, d) = match fit_plane_svd(&pts) {
                Some(v) => v,
                None => break,
            };

            // 收集内点
            let mut inlier_indices = Vec::new();
            for i in 0..n {
                if used[i] { continue; }
                let dist = (normal[0] * cloud[i][0] + normal[1] * cloud[i][1]
                    + normal[2] * cloud[i][2] + d)
                    .abs();
                if dist < self.distance {
                    inlier_indices.push(i);
                }
            }

            if inlier_indices.len() < self.min_wall_pts {
                // 内点太少，标记为已用并跳过
                for &i in &inlier_indices { used[i] = true; }
                continue;
            }

            // 竖直性检查：非竖直平面移除内点但不记为墙面
            if normal[2].abs() > self.normal_thresh {
                for &i in &inlier_indices { used[i] = true; }
                continue;
            }

            // 竖直平面：标记为墙面
            for &i in &inlier_indices {
                used[i] = true;
                is_wall[i] = true;
            }
            planes.push([normal[0], normal[1], normal[2], d]);
        }

        // 原地重排：墙面点到前部（同时交换 is_wall 保持同步）
        let mut write = 0usize;
        for read in 0..n {
            if is_wall[read] {
                cloud.swap(read, write);
                is_wall.swap(read, write);
                write += 1;
            }
        }

        (write, planes)
    }
}
