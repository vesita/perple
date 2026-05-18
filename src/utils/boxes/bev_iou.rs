/// 计算 2D 凸多边形面积（Shoelace 公式）
pub(crate) fn polygon_area_2d(poly: &[(f32, f32)]) -> f32 {
    let n = poly.len();
    if n < 3 {
        return 0.0;
    }
    let mut area = 0.0;
    for i in 0..n {
        let j = (i + 1) % n;
        area += poly[i].0 * poly[j].1;
        area -= poly[j].0 * poly[i].1;
    }
    area.abs() / 2.0
}

/// 2D 线段交点（参数 t，沿 ab 方向）
fn intersect_2d(
    a: (f32, f32), b: (f32, f32),
    c: (f32, f32), d: (f32, f32),
) -> (f32, f32) {
    let denom = (b.0 - a.0) * (d.1 - c.1) - (b.1 - a.1) * (d.0 - c.0);
    if denom.abs() < 1e-12 {
        return ((a.0 + b.0) / 2.0, (a.1 + b.1) / 2.0);
    }
    let t = ((c.0 - a.0) * (d.1 - c.1) - (c.1 - a.1) * (d.0 - c.0)) / denom;
    let t = t.clamp(0.0, 1.0);
    (a.0 + t * (b.0 - a.0), a.1 + t * (b.1 - a.1))
}

/// 2D Sutherland-Hodgman：用 clipping 多边形裁剪 subject 多边形（均为凸多边形，CCW）
pub(crate) fn clip_polygon_2d(subject: &[(f32, f32)], clipping: &[(f32, f32)]) -> Vec<(f32, f32)> {
    let mut output = subject.to_vec();
    if output.is_empty() {
        return output;
    }

    let n = clipping.len();
    for i in 0..n {
        if output.is_empty() {
            return output;
        }
        let input = output;
        output = Vec::new();

        let p1 = clipping[i];
        let p2 = clipping[(i + 1) % n];
        let edge_x = p2.0 - p1.0;
        let edge_y = p2.1 - p1.1;

        for j in 0..input.len() {
            let curr = input[j];
            let prev = input[(j + input.len() - 1) % input.len()];

            let curr_inside = edge_x * (curr.1 - p1.1) - edge_y * (curr.0 - p1.0) >= 0.0;
            let prev_inside = edge_x * (prev.1 - p1.1) - edge_y * (prev.0 - p1.0) >= 0.0;

            if curr_inside {
                if !prev_inside {
                    output.push(intersect_2d(prev, curr, p1, p2));
                }
                output.push(curr);
            } else if prev_inside {
                output.push(intersect_2d(prev, curr, p1, p2));
            }
        }
    }
    output
}

/// 2D 凸包（Monotone Chain / Andrew 算法）
pub(crate) fn convex_hull_2d(points: &[(f32, f32)]) -> Vec<(f32, f32)> {
    if points.len() <= 1 {
        return points.to_vec();
    }

    let mut pts = points.to_vec();
    pts.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    });

    let mut lower: Vec<(f32, f32)> = Vec::new();
    for &p in &pts {
        while lower.len() >= 2 {
            let a = lower[lower.len() - 2];
            let b = lower[lower.len() - 1];
            let cross = (b.0 - a.0) * (p.1 - a.1) - (b.1 - a.1) * (p.0 - a.0);
            if cross <= 0.0 {
                lower.pop();
            } else {
                break;
            }
        }
        lower.push(p);
    }

    let mut upper: Vec<(f32, f32)> = Vec::new();
    for &p in pts.iter().rev() {
        while upper.len() >= 2 {
            let a = upper[upper.len() - 2];
            let b = upper[upper.len() - 1];
            let cross = (b.0 - a.0) * (p.1 - a.1) - (b.1 - a.1) * (p.0 - a.0);
            if cross <= 0.0 {
                upper.pop();
            } else {
                break;
            }
        }
        upper.push(p);
    }

    lower.pop();
    upper.pop();
    lower.extend(upper);
    lower
}
