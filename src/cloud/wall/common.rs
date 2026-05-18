//! 墙体策略共用代码
//!
//! 被 `bev_lsd`、`bev_edlines`、`edlines_ref` 共享的图像处理与几何验证函数。

// ─── 梯度二进制方向与步进方向常量（EDLines 共用） ─────────

/// |gx| >= |gy|: 梯度垂直 → 边缘水平 → 左右追踪
pub const EDGE_VERTICAL: u8 = 1;
/// |gy| > |gx|: 梯度水平 → 边缘垂直 → 上下追踪
pub const EDGE_HORIZONTAL: u8 = 2;

pub const LEFT: u8 = 3;
pub const RIGHT: u8 = 4;
pub const UP: u8 = 5;
pub const DOWN: u8 = 6;

// ─── BEV 密度编码 ────────────────────────────────────────

/// 将点云投影到 BEV 栅格，log1p 归一化到 [0, 255]。
pub fn bev_encode(cloud: &[[f32; 3]], size: usize, max_range: f32, resolution: f32) -> Vec<u8> {
    let mut bev = vec![0u32; size * size];
    for p in cloud.iter() {
        if p[0].abs() >= max_range || p[1].abs() >= max_range { continue; }
        let x = ((p[0] + max_range) / resolution) as isize;
        let y = ((p[1] + max_range) / resolution) as isize;
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
    img
}

// ─── 高斯模糊（可分离 1D 卷积，边界 clamp） ──────────────

pub fn gaussian_blur(src: &[u8], w: usize, h: usize, sigma: f32) -> Vec<u8> {
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

// ─── 边缘绘制（EDLines 链式追踪） ────────────────────────

/// 从锚点沿指定方向追踪边缘链。
pub fn walk_edge_chain(
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

// ─── 曲率分裂 ────────────────────────────────────────────

/// 对边缘链按曲率递归分裂为多个近似直线线段。
pub fn split_chain_by_curvature(chain: &[(usize, usize)], max_error: f32) -> Vec<Vec<(usize, usize)>> {
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

    let (x1, y1) = (chain[start].0 as f32, chain[start].1 as f32);
    let (x2, y2) = (chain[end].0 as f32, chain[end].1 as f32);
    let dx = x2 - x1;
    let dy = y2 - y1;
    let len2 = dx * dx + dy * dy;
    if len2 < 1e-6 {
        segments.push(chain[start..=end].to_vec());
        return;
    }

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

// ─── PCA 矩形拟合 ────────────────────────────────────────

/// PCA 最小外接矩形拟合。
pub fn fit_rectangle(region: &[(usize, usize)]) -> (f32, f32, f32, f32, f32) {
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

// ─── 墙体点分类与验证 ────────────────────────────────────

/// 对一条候选线段执行墙体点分类与几何验证。
///
/// 计算线段在物理空间中的位置，收集距离在 `distance` 内的点，校验
/// Z 跨度和沿墙跨度。通过验证后将内点 swap 到 `cloud` 前端 `total_wall` 处。
///
/// 返回墙面平面方程 `[nx, ny, 0.0, d]`（法向量水平，垂直分量恒为 0）。
pub fn classify_wall_points(
    cloud: &mut [[f32; 3]],
    total_wall: &mut usize,
    cxp: f32, cyp: f32, length: f32, angle: f32,
    resolution: f32, max_range: f32,
    distance: f32, min_wall_pts: usize,
    min_z_span: f32, min_extent: f32,
) -> Option<[f32; 4]> {
    let half = length / 2.0;
    let cos_a = angle.cos();
    let sin_a = angle.sin();
    let px1 = cxp - half * cos_a;
    let py1 = cyp - half * sin_a;
    let px2 = cxp + half * cos_a;
    let py2 = cyp + half * sin_a;

    let x1 = px1 * resolution - max_range;
    let y1 = py1 * resolution - max_range;
    let x2 = px2 * resolution - max_range;
    let y2 = py2 * resolution - max_range;

    let dx = x2 - x1;
    let dy = y2 - y1;
    let len_m = (dx * dx + dy * dy).sqrt();
    if len_m < 1e-6 { return None; }

    let rnx = -dy / len_m;
    let rny = dx / len_m;
    let rd = -(rnx * x1 + rny * y1);

    let remaining = &cloud[*total_wall..];
    let mut inlier_rel = Vec::new();
    let mut z_min = f32::MAX;
    let mut z_max = f32::MIN;

    for (i, p) in remaining.iter().enumerate() {
        let dist = (rnx * p[0] + rny * p[1] + rd).abs();
        if dist < distance {
            inlier_rel.push(i);
            if p[2] < z_min { z_min = p[2]; }
            if p[2] > z_max { z_max = p[2]; }
        }
    }

    if inlier_rel.len() < min_wall_pts { return None; }
    if z_max - z_min < min_z_span { return None; }

    let line_dir_x = -rny;
    let line_dir_y = rnx;
    let (mut t_min, mut t_max) = (f32::MAX, f32::MIN);
    for &rel_idx in &inlier_rel {
        let t = remaining[rel_idx][0] * line_dir_x + remaining[rel_idx][1] * line_dir_y;
        if t < t_min { t_min = t; }
        if t > t_max { t_max = t; }
    }
    if t_max - t_min < min_extent { return None; }

    let mut write = *total_wall;
    for &rel_idx in &inlier_rel {
        cloud.swap(write, *total_wall + rel_idx);
        write += 1;
    }
    *total_wall = write;
    Some([rnx, rny, 0.0, rd])
}
