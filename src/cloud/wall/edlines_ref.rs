/// 原版 EDLines 参考实现（Akinlar & Topal, 2011）
///
/// 与 BevEdLines 的关键区别：
/// 1. 完整 atan2 梯度角度 → 4 扇区量化 NMS（而非 2 方向 NMS）
/// 2. Helmholtz NFA 线段验证（而非启发式 max_fit_error）
///
/// 共享设计（对比公平）：
/// - BEV 栅格化、高斯模糊 = 相同
/// - 二进制方向边缘绘制 = 相同（已验证稳定）
/// - PCA 拟合、曲率分裂、墙壁验证 = 相同
///
/// 算法步骤：BEV → 高斯模糊 → 4扇区梯度+NMS锚点 → 边缘绘制(二进制) →
///           曲率分裂 → NFA验证 → PCA拟合 → 墙壁验证
use super::WallPickStrategy;

// 梯度方向扇区（atan2 量化到 4 个方向）
const GRAD_S0: u8 = 0;   // [-22.5°, 22.5°) ∪ [157.5°, 202.5°) — 梯度近水平
const GRAD_S45: u8 = 1;  // [22.5°, 67.5°) ∪ [202.5°, 247.5°) — 梯度 45°
const GRAD_S90: u8 = 2;  // [67.5°, 112.5°) ∪ [247.5°, 292.5°) — 梯度近垂直
const GRAD_S135: u8 = 3; // [112.5°, 157.5°) ∪ [292.5°, 337.5°) — 梯度 135°

// 二进制梯度方向（与 BevEdLines 一致，用于边缘绘制）
const EDGE_VERTICAL: u8 = 1;   // |gx| >= |gy|
const EDGE_HORIZONTAL: u8 = 2; // |gy| > |gx|

// 边缘绘制步进方向
const LEFT: u8 = 3;
const RIGHT: u8 = 4;
const UP: u8 = 5;
const DOWN: u8 = 6;

pub struct EdLinesRef {
    resolution: f32,
    max_range: f32,
    distance: f32,
    min_wall_pts: usize,
    max_walls: usize,
    min_z_span: f32,
    min_extent: f32,
    grad_threshold: f32,
    anchor_threshold: f32,
    min_chain_len: usize,
    max_curvature_error: f32,
    gaussian_sigma: f32,
    nfa_epsilon: f32,
    use_nfa: bool,
}

impl EdLinesRef {
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
            min_chain_len: 10,
            max_curvature_error: 1.5,
            gaussian_sigma: 0.0,
            nfa_epsilon: 1.0,
            use_nfa: true,
        }
    }

    pub fn with_params(distance: f32, min_wall_pts: usize) -> Self {
        Self { distance, min_wall_pts, ..Self::new() }
    }

    pub fn with_min_extent(mut self, extent: f32) -> Self {
        self.min_extent = extent;
        self
    }

    pub fn with_gaussian_blur(mut self, sigma: f32) -> Self {
        self.gaussian_sigma = sigma;
        self
    }

    pub fn with_nfa(mut self, enable: bool) -> Self {
        self.use_nfa = enable;
        self
    }

    pub fn with_nfa_epsilon(mut self, eps: f32) -> Self {
        self.nfa_epsilon = eps;
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

    pub fn with_min_chain_len(mut self, len: usize) -> Self {
        self.min_chain_len = len;
        self
    }
}

impl WallPickStrategy for EdLinesRef {
    fn strategy_name(&self) -> &'static str { "edlines_ref" }

    fn pick(&mut self, cloud: &mut [[f32; 3]]) -> (usize, Vec<[f32; 4]>) {
        let n = cloud.len();
        if n < self.min_wall_pts { return (0, Vec::new()); }

        let size = (2.0 * self.max_range / self.resolution) as usize;

        // ── 1. BEV 密度编码（同 BevEdLines）──
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

        let img = if self.gaussian_sigma > 0.0 {
            gaussian_blur(&img, size, size, self.gaussian_sigma)
        } else {
            img
        };

        // ── 2. 4 扇区梯度计算 ──
        // 返回：幅值、4扇区(用于NMS)、级线角(用于NFA)、二进制方向(用于边缘绘制)
        let (grad_mag, grad_sector, ll_angle, grad_bin) = sobel_gradient_full(&img, size, size);

        let max_mag = grad_mag.iter().fold(0.0f32, |a, &b| a.max(b));
        if max_mag < 1e-6 { return (0, Vec::new()); }
        let mag_threshold = max_mag * self.grad_threshold;
        let anchor_mag_threshold = max_mag * self.anchor_threshold;

        // ── 3. 4 扇区锚点检测（原版 EDLines NMS）──
        // 沿梯度方向做 NMS，梯度方向分 4 个扇区
        let mut is_anchor = vec![false; size * size];
        for y in 2..size - 2 {
            for x in 2..size - 2 {
                let i = y * size + x;
                if grad_mag[i] < mag_threshold { continue; }

                let nms_pass = match grad_sector[i] {
                    GRAD_S0 => {
                        grad_mag[i] >= grad_mag[i - 1] + anchor_mag_threshold
                            && grad_mag[i] >= grad_mag[i + 1] + anchor_mag_threshold
                    }
                    GRAD_S45 => {
                        grad_mag[i] >= grad_mag[i - size + 1] + anchor_mag_threshold
                            && grad_mag[i] >= grad_mag[i + size - 1] + anchor_mag_threshold
                    }
                    GRAD_S90 => {
                        grad_mag[i] >= grad_mag[i - size] + anchor_mag_threshold
                            && grad_mag[i] >= grad_mag[i + size] + anchor_mag_threshold
                    }
                    GRAD_S135 => {
                        grad_mag[i] >= grad_mag[i - size - 1] + anchor_mag_threshold
                            && grad_mag[i] >= grad_mag[i + size + 1] + anchor_mag_threshold
                    }
                    _ => false,
                };
                if nms_pass { is_anchor[i] = true; }
            }
        }

        // ── 4. 边缘绘制（用二进制方向，与 BevEdLines 一致）──
        let mut edges = vec![f32::NEG_INFINITY; size * size];
        let mut chains: Vec<Vec<(usize, usize)>> = Vec::new();

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
            let (d1, d2) = match grad_bin[ay * size + ax] {
                EDGE_VERTICAL => (UP, DOWN),
                EDGE_HORIZONTAL => (LEFT, RIGHT),
                _ => continue,
            };

            walk_edge_chain(&grad_mag, &grad_bin, size, size, mag_threshold, ax, ay, d1, &mut edges, &mut chain);
            walk_edge_chain(&grad_mag, &grad_bin, size, size, mag_threshold, ax, ay, d2, &mut edges, &mut chain);

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

        // ── 5. 线段拟合 + 曲率分裂 + NFA 验证 ──
        let mut line_segments: Vec<(f32, f32, f32, f32, f32, Vec<(usize, usize)>)> = Vec::new();
        for chain in &chains {
            let sub_segments = split_chain_by_curvature(chain, self.max_curvature_error);
            for seg in &sub_segments {
                if seg.len() < self.min_chain_len { continue; }
                let (cx, cy, length, width, angle) = fit_rectangle(seg);
                if length < 3.0 || width < 0.5 || length / width < 1.5 { continue; }

                // Helmholtz NFA 验证（原版 EDLines 核心差异）
                if self.use_nfa && !nfa_validate(seg, &ll_angle, size, angle, self.nfa_epsilon) {
                    continue;
                }

                line_segments.push((cx, cy, length, width, angle, seg.clone()));
            }
        }

        if line_segments.is_empty() { return (0, Vec::new()); }
        line_segments.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

        log::debug!("edlines_ref: size={} anchors={} chains={} segments={}",
            size, anchor_list.len(), chains.len(), line_segments.len());

        // ── 6. 墙壁点分类（同 BevEdLines）──
        let mut total_wall = 0usize;
        let mut planes = Vec::new();
        let wall_end = n;

        for &(cxp, cyp, length, _width, angle, _) in line_segments.iter().take(self.max_walls * 2) {
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

// ─── 梯度计算 ──────────────────────────────────────────

/// Sobel 3×3 梯度计算，返回四种信息
///
/// 返回值：(幅值, 4扇区, 级线角, 二进制方向)
/// - 幅值：|gx| + |gy|（与 BevEdLines 一致）
/// - 4扇区：atan2 量化为 S0/S45/S90/S135（用于 NMS）
/// - 级线角：[0,π)（用于 NFA 验证）
/// - 二进制方向：EDGE_VERTICAL/EDGE_HORIZONTAL（用于边缘绘制）
fn sobel_gradient_full(src: &[u8], w: usize, h: usize) -> (Vec<f32>, Vec<u8>, Vec<f32>, Vec<u8>) {
    let n = src.len();
    let mut mag = vec![0.0f32; n];
    let mut sector = vec![0u8; n];
    let mut ll_angle = vec![0.0f32; n];
    let mut bin_dir = vec![0u8; n];

    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let i = y * w + x;
            let gx = -1i32 * src[i - w - 1] as i32 + 1 * src[i - w + 1] as i32
                     -2 * src[i - 1] as i32     + 2 * src[i + 1] as i32
                     -1 * src[i + w - 1] as i32 + 1 * src[i + w + 1] as i32;
            let gy = -1i32 * src[i - w - 1] as i32 - 2 * src[i - w] as i32 - 1 * src[i - w + 1] as i32
                     +1 * src[i + w - 1] as i32 + 2 * src[i + w] as i32 + 1 * src[i + w + 1] as i32;

            let gx_f = gx as f32;
            let gy_f = gy as f32;
            let gx_abs = gx_f.abs();
            let gy_abs = gy_f.abs();
            mag[i] = gx_abs + gy_abs;

            // 二进制方向（用于边缘绘制，同 BevEdLines）
            bin_dir[i] = if gx_abs >= gy_abs { EDGE_VERTICAL } else { EDGE_HORIZONTAL };

            // 4 扇区量化（用于 NMS）
            let angle = gy_f.atan2(gx_f);
            let angle_pos = if angle < 0.0 { angle + 2.0 * std::f32::consts::PI } else { angle };
            let sec = (angle_pos / (std::f32::consts::PI / 4.0)) as u8 % 8;
            sector[i] = match sec {
                0 | 4 => GRAD_S0,
                1 | 5 => GRAD_S45,
                2 | 6 => GRAD_S90,
                3 | 7 => GRAD_S135,
                _ => GRAD_S0,
            };

            // 级线角 level-line angle（用于 NFA）
            // LSD 定义：atan2(gx, -gy) 规范到 [0, π)
            let ll = if gy != 0 || gx != 0 {
                let a = (gx_f).atan2(-gy_f);
                if a < 0.0 { a + std::f32::consts::PI } else if a >= std::f32::consts::PI { a - std::f32::consts::PI } else { a }
            } else {
                0.0
            };
            ll_angle[i] = ll;
        }
    }
    (mag, sector, ll_angle, bin_dir)
}

// ─── Helmholtz NFA 验证 ──────────────────────────────────

/// Helmholtz NFA 验证
///
/// NFA = N_tests × P_binomial(n, k, p)
/// 其中 n = 像素数, k = 对齐像素数, p = π/8/π = 1/8
/// N_tests = (w×h)^(5/2)
/// 若 NFA < ε → 线段有效
fn nfa_validate(
    chain: &[(usize, usize)],
    ll_angle: &[f32],
    img_w: usize,
    line_angle: f32,
    epsilon: f32,
) -> bool {
    let n = chain.len();
    if n < 4 { return false; }

    let tau = std::f32::consts::PI / 8.0; // 22.5°
    let p = 1.0 / 8.0;

    // 统计与线段方向对齐的像素数
    let mut k = 0u32;
    for &(x, y) in chain {
        let px_angle = ll_angle[y * img_w + x];
        let diff = (px_angle - line_angle).abs();
        if diff.min(std::f32::consts::PI - diff) < tau {
            k += 1;
        }
    }

    if k == 0 { return false; }

    // log(NFA) = log((w*h)^(5/2)) + log(BinomialTail)
    let log_n_tests = 2.5 * (img_w as f64 * img_w as f64).ln();
    let log_tail = log_binomial_tail(n as u64, k as u64, p as f64);
    let log_nfa = log_n_tests + log_tail;

    log_nfa.exp() < epsilon as f64
}

/// log(P(X >= k)) 其中 X ~ Binomial(n, p)
/// 用 log-sum-exp + 递推避免溢出
fn log_binomial_tail(n: u64, k: u64, p: f64) -> f64 {
    if k > n { return f64::NEG_INFINITY; }
    if k == 0 { return 0.0; }
    if k == n { return n as f64 * p.ln(); }

    let log_p = p.ln();
    let log_1mp = (1.0 - p).ln();
    let log_ratio = log_p - log_1mp;

    // 第一轮：找最大值
    let mut log_term = log_comb(n, k) + k as f64 * log_p + (n - k) as f64 * log_1mp;
    let mut max_log = log_term;
    for i in k + 1..=n {
        log_term = log_term + ((n - i + 1) as f64).ln() - (i as f64).ln() + log_ratio;
        if log_term > max_log { max_log = log_term; }
    }

    // 第二轮：log-sum-exp
    log_term = log_comb(n, k) + k as f64 * log_p + (n - k) as f64 * log_1mp;
    let mut sum = 1.0; // exp(0) for the first term (relative to max)
    for i in k + 1..=n {
        log_term = log_term + ((n - i + 1) as f64).ln() - (i as f64).ln() + log_ratio;
        sum += (log_term - max_log).exp();
    }

    max_log + sum.ln()
}

/// log(C(n, k))
fn log_comb(n: u64, k: u64) -> f64 {
    if k == 0 || k == n { return 0.0; }
    log_gamma(n as f64 + 1.0) - log_gamma(k as f64 + 1.0) - log_gamma((n - k) as f64 + 1.0)
}

/// lnΓ(x) — Lanczos 近似
fn log_gamma(x: f64) -> f64 {
    let p: [f64; 7] = [
        0.99999999999980993, 676.5203681218851, -1259.1392167224028,
        771.32342877765313, -176.61502916214059, 12.507343278686905,
        -0.13857109526572012,
    ];
    let g = 7.0;
    if x < 0.5 {
        std::f64::consts::PI.ln() - (std::f64::consts::PI * x).sin().ln() - log_gamma(1.0 - x)
    } else {
        let xm1 = x - 1.0;
        let t = xm1 + g + 0.5;
        let ser = p.iter().enumerate().fold(0.0, |s, (i, &c)| s + c / (xm1 + i as f64));
        0.5 * (2.0 * std::f64::consts::PI).ln() + (xm1 + 0.5) * t.ln() - t + ser.ln()
    }
}

// ─── 边缘绘制（同 BevEdLines） ──────────────────────────

fn walk_edge_chain(
    grad_mag: &[f32],
    grad_bin: &[u8],
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

    loop {
        let mut best_i = None;
        let mut best_mag = mag_threshold;

        let cands = [
            (x + step_x, y + step_y),
            (x + step_x + step_y, y + step_y - step_x),
            (x + step_x - step_y, y + step_y + step_x),
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
                // 二进制方向一致性检查（同 BevEdLines）
                match dir {
                    LEFT | RIGHT => if grad_bin[ni] != EDGE_HORIZONTAL { break; },
                    UP | DOWN => if grad_bin[ni] != EDGE_VERTICAL { break; },
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

// ─── 曲率分裂（同 BevEdLines） ──────────────────────────

fn split_chain_by_curvature(chain: &[(usize, usize)], max_error: f32) -> Vec<Vec<(usize, usize)>> {
    if chain.len() < 4 { return vec![chain.to_vec()]; }
    let mut segments = Vec::new();
    split_recursive(chain, 0, chain.len() - 1, max_error, &mut segments);
    segments
}

fn split_recursive(
    chain: &[(usize, usize)], start: usize, end: usize,
    max_error: f32, segments: &mut Vec<Vec<(usize, usize)>>,
) {
    if end - start < 3 {
        segments.push(chain[start..=end].to_vec());
        return;
    }

    let (x1, y1) = (chain[start].0 as f32, chain[start].1 as f32);
    let (x2, y2) = (chain[end].0 as f32, chain[end].1 as f32);
    let dx = x2 - x1;
    let dy = y2 - y1;
    let len2 = dx * dx + dy * dy;
    if len2 < 1e-6 {
        segments.push(chain[start..=end].to_vec());
        return;
    }

    let (mut max_dist, mut split_idx) = (0.0f32, start);
    for i in (start + 1)..end {
        let (px, py) = (chain[i].0 as f32, chain[i].1 as f32);
        let dist = ((py - y1) * dx - (px - x1) * dy).abs() / len2.sqrt();
        if dist > max_dist { max_dist = dist; split_idx = i; }
    }

    if max_dist > max_error && split_idx > start && split_idx < end {
        split_recursive(chain, start, split_idx, max_error, segments);
        split_recursive(chain, split_idx, end, max_error, segments);
    } else {
        segments.push(chain[start..=end].to_vec());
    }
}

// ─── 辅助 ────────────────────────────────────────────────

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

fn fit_rectangle(region: &[(usize, usize)]) -> (f32, f32, f32, f32, f32) {
    let n = region.len() as f32;
    let (mut cx, mut cy) = (0.0, 0.0);
    for &(x, y) in region { cx += x as f32; cy += y as f32; }
    cx /= n; cy /= n;

    let (mut xx, mut xy, mut yy) = (0.0, 0.0, 0.0);
    for &(x, y) in region {
        let dx = x as f32 - cx;
        let dy = y as f32 - cy;
        xx += dx * dx; xy += dx * dy; yy += dy * dy;
    }

    let angle = if xy.abs() > 1e-6 {
        let trace = xx + yy;
        let det = xx * yy - xy * xy;
        let sqrt_term = ((trace * trace / 4.0 - det).max(0.0)).sqrt();
        (trace / 2.0 + sqrt_term - xx).atan2(xy)
    } else { 0.0 };

    let (cos_a, sin_a) = (angle.cos(), angle.sin());
    let (mut min_p, mut max_p, mut min_q, mut max_q) = (f32::MAX, f32::MIN, f32::MAX, f32::MIN);
    for &(x, y) in region {
        let dx = x as f32 - cx;
        let dy = y as f32 - cy;
        let p = dx * cos_a + dy * sin_a;
        let q = -dx * sin_a + dy * cos_a;
        min_p = min_p.min(p); max_p = max_p.max(p);
        min_q = min_q.min(q); max_q = max_q.max(q);
    }

    (cx, cy, max_p - min_p, max_q - min_q, angle)
}
