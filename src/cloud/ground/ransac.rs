use crate::config::fixif;
use crate::utils::random::select_some;
use crate::utils::boxes::Box3D;
use super::GroundPickStrategy;
use super::super::CldBud;

/// RANSAC 地面检测
pub struct RansacGround {
    distance: Option<f32>,
    iterations: Option<usize>,
}

impl RansacGround {
    pub fn new() -> Self { Self { distance: None, iterations: None } }
    pub fn with_params(distance: f32, iterations: usize) -> Self {
        Self { distance: Some(distance), iterations: Some(iterations) }
    }
}

impl GroundPickStrategy for RansacGround {
    fn strategy_name(&self) -> &'static str { "ransac" }

    fn pick(&mut self, cloud: &mut [[f32; 3]]) -> (usize, Vec<CldBud>, Option<[f32; 4]>) {
        let n = cloud.len();
        if n < 10 {
            return (0, Vec::new(), None);
        }

        let cfg = fixif();
        let distance_threshold = self.distance.unwrap_or(cfg.ground_ransac_distance);
        let iterations = self.iterations.unwrap_or(cfg.ground_ransac_iterations);
        let upside_down = cfg.upside_down;

        if upside_down { for p in cloud.iter_mut() { p[2] = -p[2]; } }

        let mut best_count = 0usize;
        let mut best_plane = ([0.0f32; 3], [0.0f32; 3]);

        for _ in 0..iterations {
            let idx = select_some(0, n, 3);
            if idx.len() < 3 { continue; }
            let (p1, p2, p3) = (&cloud[idx[0]], &cloud[idx[1]], &cloud[idx[2]]);

            let v1 = [p2[0]-p1[0], p2[1]-p1[1], p2[2]-p1[2]];
            let v2 = [p3[0]-p1[0], p3[1]-p1[1], p3[2]-p1[2]];
            let nx = v1[1]*v2[2] - v1[2]*v2[1];
            let ny = v1[2]*v2[0] - v1[0]*v2[2];
            let nz = v1[0]*v2[1] - v1[1]*v2[0];
            let len = (nx*nx + ny*ny + nz*nz).sqrt();
            if len < 1e-6 { continue; }
            let (nx, ny, nz) = (nx/len, ny/len, nz/len);
            if nz.abs() < 0.7 { continue; }

            let count = cloud.iter().filter(|p| {
                let dx = p[0]-p1[0]; let dy = p[1]-p1[1]; let dz = p[2]-p1[2];
                (nx*dx + ny*dy + nz*dz).abs() < distance_threshold
            }).count();

            if count > best_count {
                best_count = count;
                best_plane = ([p1[0], p1[1], p1[2]], [nx, ny, nz]);
            }
        }

        let (pp, norm) = &best_plane;
        let inlier_mask: Vec<bool> = cloud.iter().map(|p| {
            let dx = p[0]-pp[0]; let dy = p[1]-pp[1]; let dz = p[2]-pp[2];
            (norm[0]*dx + norm[1]*dy + norm[2]*dz).abs() < distance_threshold
        }).collect();

        let n_ground = inlier_mask.iter().filter(|&&m| m).count();
        let plane_eq = if best_count > 0 {
            Some([norm[0], norm[1], norm[2], -(norm[0]*pp[0] + norm[1]*pp[1] + norm[2]*pp[2])])
        } else {
            None
        };

        // 重排：地面点移到前部
        let mut write = 0;
        for read in 0..n {
            if inlier_mask[read] { cloud.swap(read, write); write += 1; }
        }

        if upside_down { for p in cloud.iter_mut() { p[2] = -p[2]; } }

        let mut ground_box = Box3D::empty_box();
        ground_box.cloud2box(&cloud[..n_ground].to_vec());
        let bud = CldBud::new(ground_box, 0, "ground".to_string(), 1.0);

        (n_ground, vec![bud], plane_eq)
    }
}
