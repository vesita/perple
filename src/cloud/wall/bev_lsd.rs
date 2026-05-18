/// BEV 鸟瞰图 + LSD 线段检测墙体提取（bev_lsd）。
///
/// # 设计思路
///
/// 实现 LSD (Line Segment Detector, Grompone von Gioi et al. 2010) 风格算法：
/// 1. BEV 栅格化 → 梯度计算（Sobel）
/// 2. 按梯度幅值降序排序种子
/// 3. 从高梯度种子出发进行 BFS 区域生长（level-line 角度一致）
/// 4. PCA 矩形拟合 → 墙壁验证
///
/// # 与 bev_edlines 的区别
///
/// - bev_lsd：全图种子排序 + BFS 区域生长，对弱边缘更鲁棒
/// - bev_edlines：锚点检测 + 链式边缘绘制（EDLines），速度更快、边缘定位更精确
///
/// # 流程
///
/// 1. **BEV 栅格化**：XY 投影 → 密度编码 → log 归一化
/// 2. **梯度计算**：Sobel 3×3 → 幅值 + level-line 角度
/// 3. **种子排序**：按梯度幅值降序，跳过低梯度像素
/// 4. **区域生长**：从高梯度种子 BFS 扩展角度一致的邻域像素
/// 5. **矩形拟合**：协方差矩阵 → 主轴 → 最小外接矩形
/// 6. **墙壁验证**：长宽比 + 3D 点 Z 跨度 + 沿墙跨度
/// 7. **墙体点分类**：对每条候选直线，收集距离 < `distance` 的 3D 点
///
/// 聚类策略：lsd（LSD 线段检测）— 图像域区域生长
/// 几何检测：l2（2D 线拟合）— 点到直线距离分类
/// 空间索引：bev（鸟瞰图栅格）— XY 平面栅格化
use super::common::{bev_encode, fit_rectangle, classify_wall_points};
use super::WallPickStrategy;
use std::collections::VecDeque;

pub struct BevLsd {
    /// BEV 分辨率（米/像素）
    resolution: f32,
    /// BEV 范围（米），绘制 [-max_range, max_range]
    max_range: f32,
    /// 点到直线距离阈值（米）
    distance: f32,
    /// 每面墙最少点数
    min_wall_pts: usize,
    /// 最多墙面板数
    max_walls: usize,
    /// 最小 Z 跨度（米）
    min_z_span: f32,
    /// 沿墙面方向最小投影跨度（米）
    min_extent: f32,
    /// 梯度幅值阈值比例 [0, 1]，相对最大梯度，默认 0.05
    grad_threshold: f32,
    /// 区域生长角度容差（度），默认 22.5
    angle_tolerance: f32,
    /// 线支持区域最少像素数，默认 8
    min_region_pts: usize,
    /// 矩形最小长宽比，默认 2.5
    min_length_ratio: f32,
}

impl BevLsd {
    pub fn new() -> Self {
        Self {
            resolution: 0.05,
            max_range: 10.0,
            distance: 0.10,
            min_wall_pts: 30,
            max_walls: 8,
            min_z_span: 1.0,
            min_extent: 0.7,
            grad_threshold: 0.05,
            angle_tolerance: 22.5,
            min_region_pts: 8,
            min_length_ratio: 2.5,
        }
    }

    pub fn with_params(distance: f32, min_wall_pts: usize) -> Self {
        Self { distance, min_wall_pts, ..Self::new() }
    }

    pub fn with_min_extent(mut self, extent: f32) -> Self {
        self.min_extent = extent;
        self
    }

    pub fn with_grad_threshold(mut self, t: f32) -> Self {
        self.grad_threshold = t;
        self
    }

    pub fn with_angle_tolerance(mut self, deg: f32) -> Self {
        self.angle_tolerance = deg;
        self
    }

    #[allow(dead_code)]
    pub fn with_min_length_ratio(mut self, r: f32) -> Self {
        self.min_length_ratio = r;
        self
    }
}

impl WallPickStrategy for BevLsd {
    fn strategy_name(&self) -> &'static str { "bev_lsd" }

    fn pick(&mut self, cloud: &mut [[f32; 3]]) -> (usize, Vec<[f32; 4]>) {
        let n = cloud.len();
        if n < self.min_wall_pts { return (0, Vec::new()); }

        let size = (2.0 * self.max_range / self.resolution) as usize;

        // ── 1. BEV 密度编码 ──
        let img = bev_encode(cloud, size, self.max_range, self.resolution);

        // ── 2. Sobel 梯度 ──
        let grad_mag = sobel_magnitude(&img, size, size);
        let grad_angle = sobel_angle(&img, size, size);

        let max_mag = grad_mag.iter().fold(0.0f32, |a, &b| a.max(b));
        if max_mag < 1e-6 { return (0, Vec::new()); }
        let mag_threshold = max_mag * self.grad_threshold;
        let angle_tol_rad = self.angle_tolerance * std::f32::consts::PI / 180.0;

        // ── 3. 种子排序（按梯度幅值降序）──
        let mut seeds: Vec<(usize, usize, f32)> = Vec::new();
        for y in 1..size - 1 {
            for x in 1..size - 1 {
                let m = grad_mag[y * size + x];
                if m > mag_threshold {
                    seeds.push((x, y, m));
                }
            }
        }
        seeds.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

        // ── 4. 区域生长（LSD 风格）──
        let mut used = vec![false; size * size];
        for x in 0..size { used[x] = true; used[(size - 1) * size + x] = true; }
        for y in 0..size { used[y * size] = true; used[y * size + size - 1] = true; }
        for i in 0..grad_mag.len() {
            if grad_mag[i] <= mag_threshold { used[i] = true; }
        }

        let mut line_segments: Vec<(f32, f32, f32, f32, f32)> = Vec::new();

        for &(sx, sy, _) in &seeds {
            if used[sy * size + sx] { continue; }

            // BFS 区域生长
            let mut region = Vec::new();
            let mut queue = VecDeque::new();
            used[sy * size + sx] = true;
            region.push((sx, sy));
            queue.push_back((sx, sy));

            let init_angle = grad_angle[sy * size + sx];
            let (mut mean_sin, mut mean_cos) = ((2.0 * init_angle).sin(), (2.0 * init_angle).cos());

            while let Some((cx, cy)) = queue.pop_front() {
                let ref_angle = (mean_sin.atan2(mean_cos) / 2.0 + std::f32::consts::PI) % std::f32::consts::PI;
                for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        if dx == 0 && dy == 0 { continue; }
                        let nx = cx as i32 + dx;
                        let ny = cy as i32 + dy;
                        if nx < 1 || nx >= size as i32 - 1 || ny < 1 || ny >= size as i32 - 1 { continue; }
                        let nu = ny as usize * size + nx as usize;
                        if used[nu] { continue; }
                        let diff = angle_dist(grad_angle[nu], ref_angle);
                        if diff < angle_tol_rad {
                            used[nu] = true;
                            region.push((nx as usize, ny as usize));
                            queue.push_back((nx as usize, ny as usize));
                            let a2 = 2.0 * grad_angle[nu];
                            mean_sin += a2.sin();
                            mean_cos += a2.cos();
                        }
                    }
                }
            }

            if region.len() < self.min_region_pts { continue; }

            // ── 5. 矩形拟合 ──
            let (cxp, cyp, length, width, angle) = fit_rectangle(&region);
            if length < 3.0 || width < 0.5 { continue; }
            if length / width < self.min_length_ratio { continue; }

            line_segments.push((cxp, cyp, length, width, angle));
        }

        if line_segments.is_empty() { return (0, Vec::new()); }

        // 按长度降序，取前 max_walls
        line_segments.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

        log::debug!("bev_lsd: size={} seeds={} segments={}", size, seeds.len(), line_segments.len());

        // ── 6. 墙体点分类 ──
        let mut total_wall = 0usize;
        let mut planes = Vec::new();

        for &(cxp, cyp, length, _width, angle) in line_segments.iter().take(self.max_walls * 2) {
            if let Some(plane) = classify_wall_points(
                cloud, &mut total_wall,
                cxp, cyp, length, angle,
                self.resolution, self.max_range,
                self.distance, self.min_wall_pts,
                self.min_z_span, self.min_extent,
            ) {
                planes.push(plane);
            }
        }

        (total_wall, planes)
    }
}

// ─── 图像处理辅助 ──────────────────────────────────────

fn sobel_magnitude(src: &[u8], w: usize, h: usize) -> Vec<f32> {
    let mut mag = vec![0.0f32; src.len()];
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let i = y * w + x;
            let gx = -1.0 * src[i - w - 1] as f32 + 1.0 * src[i - w + 1] as f32
                     -2.0 * src[i - 1] as f32     + 2.0 * src[i + 1] as f32
                     -1.0 * src[i + w - 1] as f32 + 1.0 * src[i + w + 1] as f32;
            let gy = -1.0 * src[i - w - 1] as f32 - 2.0 * src[i - w] as f32 - 1.0 * src[i - w + 1] as f32
                     +1.0 * src[i + w - 1] as f32 + 2.0 * src[i + w] as f32 + 1.0 * src[i + w + 1] as f32;
            mag[i] = (gx * gx + gy * gy).sqrt();
        }
    }
    mag
}

fn sobel_angle(src: &[u8], w: usize, h: usize) -> Vec<f32> {
    let mut angle = vec![0.0f32; src.len()];
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let i = y * w + x;
            let gx = -1.0 * src[i - w - 1] as f32 + 1.0 * src[i - w + 1] as f32
                     -2.0 * src[i - 1] as f32     + 2.0 * src[i + 1] as f32
                     -1.0 * src[i + w - 1] as f32 + 1.0 * src[i + w + 1] as f32;
            let gy = -1.0 * src[i - w - 1] as f32 - 2.0 * src[i - w] as f32 - 1.0 * src[i - w + 1] as f32
                     +1.0 * src[i + w - 1] as f32 + 2.0 * src[i + w] as f32 + 1.0 * src[i + w + 1] as f32;
            let a = gy.atan2(gx);
            angle[i] = if a < 0.0 { a + std::f32::consts::PI } else { a };
        }
    }
    angle
}

fn angle_dist(a: f32, b: f32) -> f32 {
    let d = (a - b).abs();
    d.min(std::f32::consts::PI - d)
}
