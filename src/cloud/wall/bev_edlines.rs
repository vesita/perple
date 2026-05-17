/// BEV 鸟瞰图 + EDLines 线段检测墙体提取（bev_edlines）。
///
/// # 设计思路
///
/// 实现 EDLines (Akinlar & Topal, 2011) 算法的核心步骤：
/// 1. BEV 栅格化 → 梯度计算（Sobel）
/// 2. **锚点检测**：沿梯度方向（垂直于边缘）的局部极大值点
/// 3. **边缘绘制（Edge Drawing）**：从锚点出发沿边缘方向链式追踪，连接成连续边缘链
/// 4. **线段拟合**：对每条边缘链做最小二乘拟合 + 曲率分裂
/// 5. 墙壁验证（同 bev_lsd）
///
/// # 与 bev_lsd 的区别
///
/// - bev_lsd：全图种子排序 + BFS 区域生长，对弱边缘更鲁棒
/// - bev_edlines：锚点检测 + 链式边缘追踪，速度更快、边缘定位更精确
///
/// # 流程
///
/// 1. **BEV 栅格化**：XY 投影 → 密度编码 → log 归一化
/// 2. **梯度计算**：Sobel 3×3 → 幅值 + 梯度方向
/// 3. **锚点检测**：沿梯度方向找局部幅值极大值
/// 4. **边缘绘制**：从锚点双向沿边缘方向追踪，连接高梯度像素
/// 5. **线段拟合**：PCA 或最小二乘 → 曲率分裂
/// 6. **墙壁验证**：长宽比 + 3D 点 Z 跨度 + 沿墙跨度
/// 7. **墙体点分类**：对每条候选直线，收集距离 < `distance` 的 3D 点
///
/// 聚类策略：edline（EDLines 线段检测）— 图像域链式边缘追踪
/// 几何检测：l2（2D 线拟合）— 点到直线距离分类
/// 空间索引：bev（鸟瞰图栅格）— XY 平面栅格化
use super::WallPickStrategy;

// 梯度二进制方向（EDLines 核心：用 |gx| vs |gy| 判断边缘方向，避免三角函数）
const EDGE_VERTICAL: u8 = 1;   // |gx| >= |gy|: 梯度垂直 → 边缘水平 → 左右追踪
const EDGE_HORIZONTAL: u8 = 2; // |gy| > |gx|: 梯度水平 → 边缘垂直 → 上下追踪

// 边缘绘制步进方向
const LEFT: u8 = 3;
const RIGHT: u8 = 4;
const UP: u8 = 5;
const DOWN: u8 = 6;

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
    /// 锚点检测阈值比例（相对最大梯度），低于此不产生锚点。
    /// NMS 检查：`grad_mag[i] >= grad_mag[neighbor] + max_mag * anchor_threshold`
    anchor_threshold: f32,
    /// 边缘链最少像素数
    min_chain_len: usize,
    /// 线段拟合最大曲率误差（像素），超过则分裂线段
    max_curvature_error: f32,
    /// 矩形最小长宽比
    min_length_ratio: f32,
    /// BEV 图高斯模糊 σ（像素），0 = 不模糊，推荐 0.8~1.5
    gaussian_sigma: f32,
    /// 线段拟合最大 RMS 误差（像素），0 = 不校验，推荐 0.5~1.0
    max_fit_error: f32,
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
            anchor_threshold: 0.0,
            min_chain_len: 15,
            max_curvature_error: 2.0,
            min_length_ratio: 2.5,
            gaussian_sigma: 0.0,
            max_fit_error: 0.0,
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

    pub fn with_anchor_threshold(mut self, t: f32) -> Self {
        self.anchor_threshold = t;
        self
    }

    pub fn with_chain_min_length(mut self, len: usize) -> Self {
        self.min_chain_len = len;
        self
    }

    pub fn with_gaussian_blur(mut self, sigma: f32) -> Self {
        self.gaussian_sigma = sigma;
        self
    }

    pub fn with_fit_error_threshold(mut self, err: f32) -> Self {
        self.max_fit_error = err;
        self
    }

    /// 启用所有优化：高斯模糊 0.8 + 锚点阈值 0.04 + 拟合误差校验 0.5
    pub fn with_optimizations(mut self) -> Self {
        self.gaussian_sigma = 0.8;
        self.anchor_threshold = 0.04;
        self.max_fit_error = 0.5;
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

        // ── 可选高斯模糊 ──
        let img = if self.gaussian_sigma > 0.0 {
            gaussian_blur(&img, size, size, self.gaussian_sigma)
        } else {
            img
        };

        // ── 2. Sobel 梯度 + 二进制方向（无三角函数）──
        let (grad_mag, grad_dir) = sobel_gradient(&img, size, size);

        let max_mag = grad_mag.iter().fold(0.0f32, |a, &b| a.max(b));
        if max_mag < 1e-6 { return (0, Vec::new()); }
        let mag_threshold = max_mag * self.grad_threshold;
        let anchor_mag_threshold = max_mag * self.anchor_threshold;

        // ── 3. 锚点检测（NMS + 二进制方向）──
        // 沿梯度方向检查是否为局部极大值
        // 当 anchor_threshold > 0 时增加阈值偏移，抑制弱边缘锚点
        let mut is_anchor = vec![false; size * size];
        for y in 2..size - 2 {
            for x in 2..size - 2 {
                let i = y * size + x;
                if grad_mag[i] < mag_threshold { continue; }

                match grad_dir[i] {
                    EDGE_VERTICAL => {
                        // |gx|>=|gy|: 梯度水平 → 检查左右邻域极大值
                        if grad_mag[i] >= grad_mag[i - 1] + anchor_mag_threshold
                            && grad_mag[i] >= grad_mag[i + 1] + anchor_mag_threshold
                        {
                            is_anchor[i] = true;
                        }
                    }
                    EDGE_HORIZONTAL => {
                        // |gy|>|gx|: 梯度垂直 → 检查上下邻域极大值
                        if grad_mag[i] >= grad_mag[i - size] + anchor_mag_threshold
                            && grad_mag[i] >= grad_mag[i + size] + anchor_mag_threshold
                        {
                            is_anchor[i] = true;
                        }
                    }
                    _ => {}
                }
            }
        }

        // ── 4. 边缘绘制（Edge Drawing）──
        let mut edges = vec![f32::NEG_INFINITY; size * size];
        let mut chains: Vec<Vec<(usize, usize)>> = Vec::new();

        // 所有锚点按梯度幅值降序
        let mut anchor_list: Vec<(usize, usize, f32)> = Vec::new();
        for y in 1..size - 1 {
            for x in 1..size - 1 {
                if is_anchor[y * size + x] {
                    anchor_list.push((x, y, grad_mag[y * size + x]));
                }
            }
        }
        anchor_list.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

        for &(ax, ay, _) in &anchor_list {
            if edges[ay * size + ax].is_finite() { continue; }

            let mut chain = Vec::new();
            // 从二进制方向确定边缘追踪方向
            // EDGE_VERTICAL(|gx|>=|gy|): 梯度水平 → 边缘垂直 → 上下追踪
            // EDGE_HORIZONTAL(|gy|>|gx|): 梯度垂直 → 边缘水平 → 左右追踪
            let (d1, d2) = match grad_dir[ay * size + ax] {
                EDGE_VERTICAL => (UP, DOWN),
                EDGE_HORIZONTAL => (LEFT, RIGHT),
                _ => continue,
            };

            walk_edge_chain(&grad_mag, &grad_dir, size, size, mag_threshold, ax, ay, d1, &mut edges, &mut chain);
            walk_edge_chain(&grad_mag, &grad_dir, size, size, mag_threshold, ax, ay, d2, &mut edges, &mut chain);

            if chain.len() < self.min_chain_len {
                for &(cx, cy) in &chain {
                    edges[cy * size + cx] = f32::NEG_INFINITY;
                }
                continue;
            }

            let chain_id = chains.len() as f32;
            for &(cx, cy) in &chain {
                edges[cy * size + cx] = chain_id;
            }
            chains.push(chain);
        }

        // ── 5. 线段拟合 + 曲率分裂 ──
        let mut line_segments: Vec<(f32, f32, f32, f32, f32)> = Vec::new();
        for chain in &chains {
            let sub_segments = split_chain_by_curvature(chain, self.max_curvature_error);
            for seg in &sub_segments {
                if seg.len() < self.min_chain_len { continue; }
                let (cx, cy, length, width, angle) = fit_rectangle(seg);
                if length < 3.0 || width < 0.5 { continue; }
                if length / width < self.min_length_ratio { continue; }
                // 可选拟合误差校验：过滤曲线段或锯齿链
                if self.max_fit_error > 0.0 {
                    let rms = fit_rms_error(seg, cx, cy, angle);
                    if rms > self.max_fit_error { continue; }
                }
                line_segments.push((cx, cy, length, width, angle));
            }
        }

        if line_segments.is_empty() { return (0, Vec::new()); }

        // 按长度降序
        line_segments.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

        log::debug!("bev_edlines: size={} anchors={} chains={} segments={}",
            size, anchor_list.len(), chains.len(), line_segments.len());

        // ── 6. 墙体点分类 ──
        let mut total_wall = 0usize;
        let mut planes = Vec::new();
        let wall_end = n;

        for &(cxp, cyp, length, _width, angle) in line_segments.iter().take(self.max_walls * 2) {
            if total_wall >= wall_end { break; }

            let half = length / 2.0;
            let cos_a = angle.cos();
            let sin_a = angle.sin();
            let px1 = cxp - half * cos_a;
            let py1 = cyp - half * sin_a;
            let px2 = cxp + half * cos_a;
            let py2 = cyp + half * sin_a;

            let x1 = px1 * self.resolution - self.max_range;
            let y1 = py1 * self.resolution - self.max_range;
            let x2 = px2 * self.resolution - self.max_range;
            let y2 = py2 * self.resolution - self.max_range;

            let dx = x2 - x1;
            let dy = y2 - y1;
            let len_m = (dx * dx + dy * dy).sqrt();
            if len_m < 1e-6 { continue; }

            let rnx = -dy / len_m;
            let rny = dx / len_m;
            let rd = -(rnx * x1 + rny * y1);

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

            let line_dir_x = -rny;
            let line_dir_y = rnx;
            let (mut t_min, mut t_max) = (f32::MAX, f32::MIN);
            for &rel_idx in &inlier_rel {
                let t = remaining[rel_idx][0] * line_dir_x + remaining[rel_idx][1] * line_dir_y;
                if t < t_min { t_min = t; }
                if t > t_max { t_max = t; }
            }
            if t_max - t_min < self.min_extent { continue; }

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

// ─── 边缘绘制（Edge Drawing）核心 ──────────────────────

/// 沿指定方向从起点追踪边缘链。
///
/// `walk_edge_chain` 实现 EDLines 的链式追踪：
/// 从起点 (sx, sy) 出发，沿边缘方向 (ed_x, ed_y) 前进，
/// 每一步从 3 个候选像素中选梯度幅值最高的作为下一步。
fn walk_edge_chain(
    grad_mag: &[f32],
    grad_dir: &[u8],
    w: usize,
    h: usize,
    mag_threshold: f32,
    sx: usize,
    sy: usize,
    dir: u8,
    edges: &mut [f32],
    chain: &mut Vec<(usize, usize)>,
) {
    if chain.is_empty() {
        chain.push((sx, sy));
        edges[sy * w + sx] = -1.0;
    }

    let (mut x, mut y) = (sx as i32, sy as i32);
    let (step_x, step_y) = match dir {
        LEFT => (-1, 0),
        RIGHT => (1, 0),
        UP => (0, -1),
        DOWN => (0, 1),
        _ => return,
    };
    let (fw_x, fw_y) = (step_x, step_y);
    let (ro_x, ro_y) = (step_x + step_y, step_y - step_x);
    let (lo_x, lo_y) = (step_x - step_y, step_y + step_x);

    loop {
        let mut best_i = None;
        let mut best_mag = mag_threshold;

        // 3 个候选像素：正前方 + 两侧偏，选梯度幅值最高且未使用
        let cands = [
            (x + fw_x, y + fw_y),
            (x + ro_x, y + ro_y),
            (x + lo_x, y + lo_y),
        ];
        for &(cx, cy) in &cands {
            if cx >= 1 && cx < w as i32 - 1 && cy >= 1 && cy < h as i32 - 1 {
                let ci = cy as usize * w + cx as usize;
                if !edges[ci].is_finite() {
                    let mag = grad_mag[ci];
                    if mag > best_mag { best_mag = mag; best_i = Some((cx, cy)); }
                }
            }
        }

        match best_i {
            Some((nx, ny)) => {
                let ni = ny as usize * w + nx as usize;
                // 边缘方向一致性检验：边缘类型必须匹配步进方向
                match dir {
                    LEFT | RIGHT => if grad_dir[ni] != EDGE_HORIZONTAL { break; },
                    UP | DOWN => if grad_dir[ni] != EDGE_VERTICAL { break; },
                    _ => {}
                }
                chain.push((nx as usize, ny as usize));
                edges[ni] = -1.0;
                x = nx;
                y = ny;
            }
            None => break,
        }
    }
}

// ─── 曲率分裂 ──────────────────────────────────────────

/// 对边缘链按曲率分裂为多个近似直线的线段。
///
/// 算法：递归分裂
/// 1. 拟合链端点到直线
/// 2. 找距直线最远的点
/// 3. 若距离 > max_error，在改点处分裂为两段，递归处理
fn split_chain_by_curvature(chain: &[(usize, usize)], max_error: f32) -> Vec<Vec<(usize, usize)>> {
    if chain.len() < 4 {
        return vec![chain.to_vec()];
    }

    let mut segments = Vec::new();
    split_recursive(chain, 0, chain.len() - 1, max_error, &mut segments);
    segments
}

fn split_recursive(
    chain: &[(usize, usize)],
    start: usize,
    end: usize,
    max_error: f32,
    segments: &mut Vec<Vec<(usize, usize)>>,
) {
    if end - start < 3 {
        segments.push(chain[start..=end].to_vec());
        return;
    }

    // 端点直线
    let (x1, y1) = (chain[start].0 as f32, chain[start].1 as f32);
    let (x2, y2) = (chain[end].0 as f32, chain[end].1 as f32);
    let dx = x2 - x1;
    let dy = y2 - y1;
    let len2 = dx * dx + dy * dy;
    if len2 < 1e-6 {
        segments.push(chain[start..=end].to_vec());
        return;
    }

    // 找距直线最远的点
    let mut max_dist = 0.0f32;
    let mut split_idx = start;
    for i in (start + 1)..end {
        let (px, py) = (chain[i].0 as f32, chain[i].1 as f32);
        let dist = ((py - y1) * dx - (px - x1) * dy).abs() / len2.sqrt();
        if dist > max_dist {
            max_dist = dist;
            split_idx = i;
        }
    }

    if max_dist > max_error && split_idx > start && split_idx < end {
        split_recursive(chain, start, split_idx, max_error, segments);
        split_recursive(chain, split_idx, end, max_error, segments);
    } else {
        segments.push(chain[start..=end].to_vec());
    }
}

// ─── 图像处理辅助 ──────────────────────────────────────

/// 高斯模糊（可分离 1D 卷积，边界 clamp）
fn gaussian_blur(src: &[u8], w: usize, h: usize, sigma: f32) -> Vec<u8> {
    let radius = (sigma * 2.5).ceil() as i32;
    let size = (2 * radius + 1) as usize;
    let mut kernel = vec![0.0f32; size];
    let mut sum = 0.0f32;
    for i in 0..size {
        let x = (i as i32 - radius) as f32;
        let g = (-0.5 * x * x / (sigma * sigma)).exp();
        kernel[i] = g;
        sum += g;
    }
    for k in &mut kernel { *k /= sum; }

    let mut tmp = vec![0.0f32; w * h];

    // 水平方向模糊
    for y in 0..h {
        for x in 0..w {
            let mut val = 0.0f32;
            for ki in 0..size {
                let sx = (x as i32 + ki as i32 - radius).clamp(0, w as i32 - 1) as usize;
                val += src[y * w + sx] as f32 * kernel[ki];
            }
            tmp[y * w + x] = val;
        }
    }

    // 垂直方向模糊
    let mut out = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut val = 0.0f32;
            for ki in 0..size {
                let sy = (y as i32 + ki as i32 - radius).clamp(0, h as i32 - 1) as usize;
                val += tmp[sy * w + x] * kernel[ki];
            }
            out[y * w + x] = val.round().clamp(0.0, 255.0) as u8;
        }
    }
    out
}

/// 单次 Sobel 计算，返回梯度幅值（|gx|+|gy|）和二进制方向（避免 atan2/cos/sin）
fn sobel_gradient(src: &[u8], w: usize, h: usize) -> (Vec<f32>, Vec<u8>) {
    let n = src.len();
    let mut mag = vec![0.0f32; n];
    let mut dir = vec![0u8; n];
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let i = y * w + x;
            let gx = -1i32 * src[i - w - 1] as i32 + 1 * src[i - w + 1] as i32
                     -2 * src[i - 1] as i32     + 2 * src[i + 1] as i32
                     -1 * src[i + w - 1] as i32 + 1 * src[i + w + 1] as i32;
            let gy = -1i32 * src[i - w - 1] as i32 - 2 * src[i - w] as i32 - 1 * src[i - w + 1] as i32
                     +1 * src[i + w - 1] as i32 + 2 * src[i + w] as i32 + 1 * src[i + w + 1] as i32;

            let gx_abs = gx.abs() as f32;
            let gy_abs = gy.abs() as f32;
            mag[i] = gx_abs + gy_abs;

            if gx_abs >= gy_abs {
                dir[i] = EDGE_VERTICAL;
            } else {
                dir[i] = EDGE_HORIZONTAL;
            }
        }
    }
    (mag, dir)
}

/// 计算边缘链像素到拟合直线的垂直 RMS 距离（像素）
///
/// line 由 (cx, cy, angle) 定义，方向向量为 (cos_a, sin_a)。
/// 垂直距离 = |-(x-cx)*sin_a + (y-cy)*cos_a|
fn fit_rms_error(region: &[(usize, usize)], cx: f32, cy: f32, angle: f32) -> f32 {
    let sin_a = angle.sin();
    let cos_a = angle.cos();
    let mut sum_sq = 0.0f32;
    for &(x, y) in region {
        let dx = x as f32 - cx;
        let dy = y as f32 - cy;
        let perp = -(dx * sin_a) + (dy * cos_a);
        sum_sq += perp * perp;
    }
    (sum_sq / region.len() as f32).sqrt()
}

/// PCA 矩形拟合（同 bev_lsd）
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

    let angle = if xy.abs() > 1e-6 {
        let trace = xx + yy;
        let det = xx * yy - xy * xy;
        let sqrt_term = ((trace * trace / 4.0 - det).max(0.0)).sqrt();
        let lambda1 = trace / 2.0 + sqrt_term;
        (lambda1 - xx).atan2(xy)
    } else {
        0.0
    };

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
