use nalgebra::{Matrix3, SVD};
use crate::config::fixif;
use crate::utils::boxes::Box3D;
use super::GroundPickStrategy;
use super::super::CldBud;

/// GPF (Ground Plane Fitting) 地面提取
pub struct GpfGround {
    n_lpr: Option<usize>,
    th_seed: Option<f32>,
    th_dist: Option<f32>,
}

impl GpfGround {
    pub fn new() -> Self { Self { n_lpr: None, th_seed: None, th_dist: None } }
    pub fn with_params(n_lpr: usize, th_seed: f32, th_dist: f32) -> Self {
        Self { n_lpr: Some(n_lpr), th_seed: Some(th_seed), th_dist: Some(th_dist) }
    }
}

impl GroundPickStrategy for GpfGround {
    fn strategy_name(&self) -> &'static str { "gpf" }

    fn pick(&mut self, cloud: &mut [[f32; 3]]) -> (usize, Vec<CldBud>, Option<[f32; 4]>) {
        let n = cloud.len();
        if n < 10 {
            return (0, Vec::new(), None);
        }

        let cfg = fixif();
        let upside_down = cfg.upside_down;
        let n_lpr = self.n_lpr.unwrap_or(100).min(n);
        let th_seed = self.th_seed.unwrap_or(0.5);
        let th_dist = self.th_dist.unwrap_or(cfg.ground_ransac_distance);

        let mut indexed: Vec<(usize, [f32; 3])> = (0..n).zip(cloud.iter().copied()).collect();
        if upside_down { for p in cloud.iter_mut() { p[2] = -p[2]; } }
        indexed.sort_by(|a, b| a.1[2].partial_cmp(&b.1[2]).unwrap());
        for (i, (_, p)) in indexed.iter().enumerate() { cloud[i] = *p; }

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
            return (0, Vec::new(), None);
        }

        let mut ground_count = seed_count;
        let mut plane_normal = [0.0f32; 3];
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

        let mut ground_mask = vec![false; n];
        for (sorted_i, &is_ground) in mask.iter().enumerate() {
            if is_ground { ground_mask[indexed[sorted_i].0] = true; }
        }

        let n_ground = ground_mask.iter().filter(|&&m| m).count();

        // 重排地面点到前部
        let mut write = 0;
        for read in 0..n {
            if ground_mask[read] { cloud.swap(read, write); write += 1; }
        }

        let plane_eq = if n_ground > 0 {
            Some([plane_normal[0], plane_normal[1], plane_normal[2], plane_d])
        } else {
            None
        };

        if upside_down { for p in cloud.iter_mut() { p[2] = -p[2]; } }

        let mut ground_box = Box3D::empty_box();
        ground_box.cloud2box(&cloud[..n_ground].to_vec());
        let bud = CldBud::new(ground_box, 0, "ground".to_string(), 1.0);

        (n_ground, vec![bud], plane_eq)
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
