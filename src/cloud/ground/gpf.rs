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

        if upside_down { for p in cloud.iter_mut() { p[2] = -p[2]; } }
        cloud.sort_unstable_by(|a, b| a[2].partial_cmp(&b[2]).unwrap());

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
        let gp: Vec<[f32; 3]> = cloud.iter().enumerate()
            .filter(|(i, _)| mask[*i]).map(|(_, p)| *p).collect();
        let (mut plane_normal, mut plane_d) = fit_plane_svd(&gp);
        loop {
            let mut new_count = 0;
            let mut new_mask = vec![false; n];
            for (i, p) in cloud.iter().enumerate() {
                if (plane_normal[0]*p[0] + plane_normal[1]*p[1] + plane_normal[2]*p[2] + plane_d).abs() < th_dist {
                    new_mask[i] = true;
                    new_count += 1;
                }
            }

            if new_count <= ground_count { break; }
            mask = new_mask;
            ground_count = new_count;

            let gp: Vec<[f32; 3]> = cloud.iter().enumerate()
                .filter(|(i, _)| mask[*i]).map(|(_, p)| *p).collect();
            let (normal, d) = fit_plane_svd(&gp);
            plane_normal = normal;
            plane_d = d;
        }

        // mask 基于排序后的 cloud 索引，直接用 mask 分区
        let n_ground = mask.iter().filter(|&&m| m).count();

        // 重排地面点到前部（cloud 已排序，mask 索引对应正确）
        let mut write = 0;
        for read in 0..n {
            if mask[read] { cloud.swap(read, write); write += 1; }
        }

        let plane_eq = if n_ground > 0 {
            // plane 在翻转坐标系下拟合，转回原始坐标系
            if upside_down {
                Some([plane_normal[0], plane_normal[1], -plane_normal[2], plane_d])
            } else {
                Some([plane_normal[0], plane_normal[1], plane_normal[2], plane_d])
            }
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
    let mut norm = match svd.v_t {
        Some(vt) => {
            let v = vt.transpose();
            let col = v.column(2);
            [col[0], col[1], col[2]]
        }
        None => [0.0, 0.0, 1.0],
    };
    let mut d = -(norm[0]*mx + norm[1]*my + norm[2]*mz);

    // 统一法向量方向：约定 n_z >= 0，确保帧间 plane_eq 符号一致
    if norm[2] < 0.0 {
        norm = [-norm[0], -norm[1], -norm[2]];
        d = -d;
    }

    (norm, d)
}
