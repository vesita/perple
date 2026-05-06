use nalgebra::{Matrix3, SVD};
use crate::config::fixif;
use super::{GroundResult, GroundStrategy};

/// GPF (Ground Plane Fitting) 地面提取
///
/// 1. 取最低 n_lpr 个点的平均高度作为种子阈值
/// 2. 迭代：SVD 拟合平面 → 扩展内点 → 重复直到收敛
pub struct GpfGround;

impl GpfGround {
    pub fn new() -> Self { Self }
}

impl GroundStrategy for GpfGround {
    fn strategy_name(&self) -> &'static str { "gpf" }

    fn extract(&mut self, cloud: &mut [[f32; 3]]) -> GroundResult {
        let n = cloud.len();
        if n < 10 {
            return GroundResult { n_ground: 0, ground_mask: vec![false; n], plane_eq: None };
        }

        let cfg = fixif();
        let upside_down = cfg.upside_down;
        let n_lpr = 100usize.min(n);
        let th_seed = 0.5f32;
        let th_dist = cfg.ground_ransac_distance;

        let mut indexed: Vec<(usize, [f32; 3])> = (0..n).zip(cloud.iter().copied()).collect();
        if upside_down { for p in cloud.iter_mut() { p[2] = -p[2]; } }
        indexed.sort_by(|a, b| a.1[2].partial_cmp(&b.1[2]).unwrap());
        for (i, (_, p)) in indexed.iter().enumerate() { cloud[i] = *p; }

        // LPR: 最低 n_lpr 个点的平均 Z
        let lpr: f32 = cloud[..n_lpr].iter().map(|p| p[2]).sum::<f32>() / n_lpr as f32;

        let mut mask = vec![false; n];
        let mut seed_count = 0;
        for (i, p) in cloud.iter().enumerate() {
            if p[2] < lpr + th_seed {
                mask[i] = true;
                seed_count += 1;
            }
        }

        if seed_count < 3 {
            if upside_down { for p in cloud.iter_mut() { p[2] = -p[2]; } }
            return GroundResult { n_ground: 0, ground_mask: vec![false; n], plane_eq: None };
        }

        // 迭代拟合
        let mut ground_count = seed_count;
        #[allow(unused_assignments)]
        let mut plane_normal = [0.0f32; 3];
        #[allow(unused_assignments)]
        let mut plane_d = 0.0f32;
        loop {
            let gp: Vec<[f32; 3]> = cloud.iter().enumerate()
                .filter(|(i, _)| mask[*i]).map(|(_, p)| *p).collect();
            let (normal, d) = fit_plane_svd(&gp);
            plane_normal = normal;
            plane_d = d;

            let mut new_count = 0;
            let mut new_mask = vec![false; n];
            for (i, p) in cloud.iter().enumerate() {
                if (normal[0]*p[0] + normal[1]*p[1] + normal[2]*p[2] + d).abs() < th_dist {
                    new_mask[i] = true;
                    new_count += 1;
                }
            }

            if new_count <= ground_count { break; }
            mask = new_mask;
            ground_count = new_count;
        }

        // 转换为原始索引顺序
        let mut ground_mask = vec![false; n];
        for (sorted_i, &is_ground) in mask.iter().enumerate() {
            if is_ground { ground_mask[indexed[sorted_i].0] = true; }
        }

        let n_ground = ground_mask.iter().filter(|&&m| m).count();
        let plane_eq = if n_ground > 0 {
            Some([plane_normal[0], plane_normal[1], plane_normal[2], plane_d])
        } else {
            None
        };

        if upside_down { for p in cloud.iter_mut() { p[2] = -p[2]; } }

        GroundResult { n_ground, ground_mask, plane_eq }
    }
}

fn fit_plane_svd(points: &[[f32; 3]]) -> ([f32; 3], f32) {
    let n = points.len();
    if n < 3 { return ([0.0, 0.0, 1.0], 0.0); }
    let n_f = n as f32;
    let mut mx = 0.0; let mut my = 0.0; let mut mz = 0.0;
    for p in points { mx += p[0]; my += p[1]; mz += p[2]; }
    mx /= n_f; my /= n_f; mz /= n_f;

    let mut cov = Matrix3::zeros();
    for p in points {
        let x = p[0]-mx; let y = p[1]-my; let z = p[2]-mz;
        cov[(0,0)] += x*x; cov[(0,1)] += x*y; cov[(0,2)] += x*z;
        cov[(1,0)] += y*x; cov[(1,1)] += y*y; cov[(1,2)] += y*z;
        cov[(2,0)] += z*x; cov[(2,1)] += z*y; cov[(2,2)] += z*z;
    }
    cov /= n_f;

    let svd = SVD::new(cov, true, false);
    let norm = match svd.v_t {
        Some(vt) => {
            let v = vt.transpose();
            let col = v.column(2);
            [col[0], col[1], col[2]]
        }
        None => [0.0, 0.0, 1.0],
    };
    let d = -(norm[0]*mx + norm[1]*my + norm[2]*mz);
    (norm, d)
}
