/// XY 平面 RANSAC 线拟合：提取投影直线的最佳法线方向。
///
/// 返回 `(nx, ny, d, 内点数)` 满足 nx·x + ny·y + d = 0。
pub(crate) fn best_xy_line(
    points: &[[f32; 3]],
    distance: f32,
    iterations: usize,
    rng_seed: Option<u64>,
) -> Option<(f32, f32, f32, usize)> {
    let n = points.len();
    if n < 2 { return None; }

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

    let (refined_nx, refined_ny, refined_d, refined_count) =
        refine_line_ls(points, best_nx, best_ny, best_d, distance);

    if refined_count == 0 {
        Some((best_nx, best_ny, best_d, best_count))
    } else {
        Some((refined_nx, refined_ny, refined_d, refined_count))
    }
}

/// TLS 最小二乘精化直线参数：收集内点 → 协方差特征分解 → 重估法线。
pub(crate) fn refine_line_ls(
    points: &[[f32; 3]],
    init_nx: f32, init_ny: f32, init_d: f32,
    distance: f32,
) -> (f32, f32, f32, usize) {
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

    let mut count = 0usize;
    for p in points {
        let dist = (nx * p[0] + ny * p[1] + d).abs();
        if dist < distance { count += 1; }
    }

    (nx, ny, d, count)
}

/// 确定性伪随机选择：用 SplitMix64 从 [start, end) 中选 `count` 个不重复索引。
pub(crate) fn select_some_seeded(start: usize, end: usize, count: usize, seed: Option<u64>) -> Vec<usize> {
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
