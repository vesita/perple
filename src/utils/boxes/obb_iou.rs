use nalgebra::{Point3, Vector3};

/// 用一个半平面裁剪一个三角形，输出 0/1/2 个新三角形。
///
/// 半平面定义为 `normal·p + d ≥ 0`（内侧），normal 为向内法线。
pub(crate) fn clip_triangle_by_plane(
    tri: [Point3<f32>; 3],
    normal: &Vector3<f32>,
    d: f32,
    out: &mut Vec<[Point3<f32>; 3]>,
) {
    let dists = [
        normal.dot(&tri[0].coords) + d,
        normal.dot(&tri[1].coords) + d,
        normal.dot(&tri[2].coords) + d,
    ];

    let inside = [dists[0] >= -1e-9, dists[1] >= -1e-9, dists[2] >= -1e-9];
    let n_inside = inside.iter().filter(|&&x| x).count();

    match n_inside {
        3 => out.push(tri),
        0 => {}
        1 => {
            let i = inside.iter().position(|&x| x).unwrap();
            let i1 = (i + 1) % 3;
            let i2 = (i + 2) % 3;
            let p1 = intersect_edge(tri[i], tri[i1], dists[i], dists[i1]);
            let p2 = intersect_edge(tri[i], tri[i2], dists[i], dists[i2]);
            out.push([tri[i], p1, p2]);
        }
        2 => {
            let o = inside.iter().position(|&x| !x).unwrap();
            let i1 = (o + 1) % 3;
            let i2 = (o + 2) % 3;
            let p1 = intersect_edge(tri[i1], tri[o], dists[i1], dists[o]);
            let p2 = intersect_edge(tri[i2], tri[o], dists[i2], dists[o]);
            out.push([tri[i1], tri[i2], p1]);
            out.push([tri[i2], p2, p1]);
        }
        _ => unreachable!(),
    }
}

/// 计算线段上两点与半平面交点（内→外的插值参数）。
fn intersect_edge(
    inside: Point3<f32>,
    outside: Point3<f32>,
    d_inside: f32,
    d_outside: f32,
) -> Point3<f32> {
    let t = (d_inside / (d_inside - d_outside)).clamp(0.0, 1.0);
    inside + (outside - inside) * t
}

/// 计算封闭三角形网格体积（散度定理）。
pub(crate) fn triangle_mesh_volume(triangles: &[[Point3<f32>; 3]]) -> f32 {
    let mut volume = 0.0;
    for tri in triangles {
        let v0 = tri[0].coords;
        let v1 = tri[1].coords;
        let v2 = tri[2].coords;
        volume += v0.dot(&v1.cross(&v2));
    }
    (volume / 6.0).abs()
}
