/// BEV 鸟瞰图 + ED 线段检测墙体提取（bev_edlines）。
///
/// # 设计思路
///
/// 受 EDLines 和 LBD (Line Band Descriptor) 启发，
/// 在 BEV 图像上检测局部线段而非全局 Hough 直线。
/// 先计算梯度幅值/方向，从高梯度种子出发进行区域生长
/// （类似 Edge Drawing 的锚点连接思想），再对每个线支持区域做矩形拟合。
///
/// # 与 bev_hough 的区别
///
/// - bev_hough：全局 Hough 投票，检测无限长直线，适合稀疏/断续墙体
/// - bev_edlines：局部梯度区域生长，检测精确线段，避免无关共线点误检
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
/// 聚类策略：edlines（ED 线段检测）— 图像域局部线段生长
/// 几何检测：l2（2D 线拟合）— 点到直线距离分类
/// 空间索引：bev（鸟瞰图栅格）— XY 平面栅格化
use super::WallPickStrategy;
use std::collections::VecDeque;

pub struct BevEdLines {
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

impl BevEdLines {
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

impl WallPickStrategy for BevEdLines {
    fn strategy_name(&self) -> &'static str { "bev_edlines" }

    fn pick(&mut self, cloud: &mut [[f32; 3]]) -> (usize, Vec<[f32; 4]>) {
        let n = cloud.len();
        if n < self.min_wall_pts { return (0, Vec::new()); }

        let size = (2.0 * self.max_range / self.resolution) as usize;

        // ── 1. BEV 密度编码 ──
        let mut bev = vec![0u32; size * size];
        for p in cloud.iter() {
            if p[0].abs() >= self.max_range || p[1].abs() >= self.max_range { continue; }
            let x = ((p[0] + self.max_range) / self.resolution) as isize;
            let y = ((p[1] + self.max_range) / self.resolution) as isize;
            if x >= 0 && (x as usize) < size && y >= 0 && (y as usize) < size {
                bev[y as usize * size + x as usize] += 1;
            }
        }

        // log1p 归一化到 [0, 255]
        let mut img_f32 = vec![0.0f32; size * size];
        let mut max_val = 0.0f32;
        for i in 0..bev.len() {
            let l = (bev[i] as f32 + 1.0).ln();
            img_f32[i] = l;
            if l > max_val { max_val = l; }
        }
        let mut img = vec![0u8; size * size];
        if max_val > 1e-6 {
            let scale = 255.0 / max_val;
            for i in 0..img_f32.len() {
                img[i] = (img_f32[i] * scale) as u8;
            }
        }

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
        // 边界像素标记为已使用
        for x in 0..size { used[x] = true; used[(size - 1) * size + x] = true; }
        for y in 0..size { used[y * size] = true; used[y * size + size - 1] = true; }
        // 低梯度像素也标记为已使用
        for i in 0..grad_mag.len() {
            if grad_mag[i] <= mag_threshold { used[i] = true; }
        }

        let mut line_segments: Vec<(f32, f32, f32, f32, f32)> = Vec::new(); // cx, cy, length, width, angle

        for &(sx, sy, _) in &seeds {
            if used[sy * size + sx] { continue; }

            // BFS 区域生长
            let mut region = Vec::new();
            let mut queue = VecDeque::new();
            used[sy * size + sx] = true;
            region.push((sx, sy));
            queue.push_back((sx, sy));

            // 用 sin/cos 跟踪圆环均值（角度在 [0, π)）
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
        let wall_end = n;

        for &(cxp, cyp, length, _width, angle) in line_segments.iter().take(self.max_walls * 2) {
            if total_wall >= wall_end { break; }

            // 线段两端点（像素坐标）
            let half = length / 2.0;
            let cos_a = angle.cos();
            let sin_a = angle.sin();
            let px1 = cxp - half * cos_a;
            let py1 = cyp - half * sin_a;
            let px2 = cxp + half * cos_a;
            let py2 = cyp + half * sin_a;

            // 像素 → 米
            let x1 = px1 * self.resolution - self.max_range;
            let y1 = py1 * self.resolution - self.max_range;
            let x2 = px2 * self.resolution - self.max_range;
            let y2 = py2 * self.resolution - self.max_range;

            let dx = x2 - x1;
            let dy = y2 - y1;
            let len_m = (dx * dx + dy * dy).sqrt();
            if len_m < 1e-6 { continue; }

            // 法线方向
            let rnx = -dy / len_m;
            let rny = dx / len_m;
            let rd = -(rnx * x1 + rny * y1);

            // 收集附近 3D 点（点到无限直线距离）
            let remaining = &cloud[total_wall..wall_end];
            let mut inlier_rel = Vec::new();
            let mut z_min = f32::MAX;
            let mut z_max = f32::MIN;

            for (i, p) in remaining.iter().enumerate() {
                let dist = (rnx * p[0] + rny * p[1] + rd).abs();
                if dist < self.distance {
                    inlier_rel.push(i);
                    if p[2] < z_min { z_min = p[2]; }
                    if p[2] > z_max { z_max = p[2]; }
                }
            }

            if inlier_rel.len() < self.min_wall_pts { continue; }
            if z_max - z_min < self.min_z_span { continue; }

            // 沿墙投影跨度
            let line_dir_x = -rny;
            let line_dir_y = rnx;
            let (mut t_min, mut t_max) = (f32::MAX, f32::MIN);
            for &rel_idx in &inlier_rel {
                let t = remaining[rel_idx][0] * line_dir_x + remaining[rel_idx][1] * line_dir_y;
                if t < t_min { t_min = t; }
                if t > t_max { t_max = t; }
            }
            if t_max - t_min < self.min_extent { continue; }

            // 标记墙体点
            let mut write = total_wall;
            for &rel_idx in &inlier_rel {
                cloud.swap(total_wall + rel_idx, write);
                write += 1;
            }
            total_wall = write;
            planes.push([rnx, rny, 0.0, rd]);
        }

        (total_wall, planes)
    }
}

// ─── 图像处理辅助 ──────────────────────────────────────

/// Sobel 梯度幅值
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

/// Sobel level-line 角度 [0, π)，梯度方向（垂直于边缘）
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
            // 角度归一化到 [0, π)
            let a = gy.atan2(gx);
            angle[i] = if a < 0.0 { a + std::f32::consts::PI } else { a };
        }
    }
    angle
}

/// 矩形拟合：返回 (cx, cy, length, width, angle_rad)
fn fit_rectangle(region: &[(usize, usize)]) -> (f32, f32, f32, f32, f32) {
    let n = region.len() as f32;
    let mut cx = 0.0f32;
    let mut cy = 0.0f32;
    for &(x, y) in region {
        cx += x as f32;
        cy += y as f32;
    }
    cx /= n;
    cy /= n;

    let mut xx = 0.0f32;
    let mut xy = 0.0f32;
    let mut yy = 0.0f32;
    for &(x, y) in region {
        let dx = x as f32 - cx;
        let dy = y as f32 - cy;
        xx += dx * dx;
        xy += dx * dy;
        yy += dy * dy;
    }

    // 角度 = 第一主成分方向
    let angle = if xy.abs() > 1e-6 {
        let trace = xx + yy;
        let det = xx * yy - xy * xy;
        let sqrt_term = ((trace * trace / 4.0 - det).max(0.0)).sqrt();
        let lambda1 = trace / 2.0 + sqrt_term;
        (lambda1 - xx).atan2(xy)
    } else {
        0.0
    };

    // 沿主轴和垂直方向投影
    let cos_a = angle.cos();
    let sin_a = angle.sin();
    let (mut min_proj, mut max_proj) = (f32::MAX, f32::MIN);
    let (mut min_perp, mut max_perp) = (f32::MAX, f32::MIN);
    for &(x, y) in region {
        let dx = x as f32 - cx;
        let dy = y as f32 - cy;
        let proj = dx * cos_a + dy * sin_a;
        let perp = -dx * sin_a + dy * cos_a;
        min_proj = min_proj.min(proj);
        max_proj = max_proj.max(proj);
        min_perp = min_perp.min(perp);
        max_perp = max_perp.max(perp);
    }

    (cx, cy, max_proj - min_proj, max_perp - min_perp, angle)
}

/// 角度距离 [0, π)，处理 π 环绕
fn angle_dist(a: f32, b: f32) -> f32 {
    let d = (a - b).abs();
    d.min(std::f32::consts::PI - d)
}
