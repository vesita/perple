use crate::config::fixif;
use crate::utils::random::select_some;
use crate::utils::boxes::Box3D;
use super::GroundPickStrategy;
use super::super::CldBud;

/// 直方图种子 + RANSAC 平面拟合生长
pub struct HistoseedPlane {
    expand: Option<f32>,
    distance: Option<f32>,
    iterations: Option<usize>,
}

impl HistoseedPlane {
    pub fn new() -> Self { Self { expand: None, distance: None, iterations: None } }
    pub fn with_params(expand: f32, distance: f32, iterations: usize) -> Self {
        Self { expand: Some(expand), distance: Some(distance), iterations: Some(iterations) }
    }
}

impl GroundPickStrategy for HistoseedPlane {
    fn strategy_name(&self) -> &'static str { "histoseed" }

    fn pick(&mut self, cloud: &mut [[f32; 3]]) -> (usize, Vec<CldBud>, Option<[f32; 4]>) {
        let n = cloud.len();
        if n < 10 {
            return (0, Vec::new(), None);
        }

        let cfg = fixif();
        let upside_down = cfg.upside_down;
        let expand = self.expand.unwrap_or(cfg.ground_expand);
        let distance = self.distance.unwrap_or(cfg.ground_ransac_distance);
        let iterations = self.iterations.unwrap_or(cfg.ground_ransac_iterations);

        if upside_down { for p in cloud.iter_mut() { p[2] = -p[2]; } }

        cloud.sort_by(|a, b| a[2].partial_cmp(&b[2]).unwrap());
        let z_min = cloud[0][2];
        let z_max = cloud[n - 1][2];
        let z_range = z_max - z_min;

        if z_range < 1e-6 {
            if upside_down { for p in cloud.iter_mut() { p[2] = -p[2]; } }
            return (0, Vec::new(), None);
        }

        let num_bins = 128usize;
        let bin_w = z_range / num_bins as f32;
        let mut bins = vec![0usize; num_bins];
        for p in cloud.iter() {
            let mut b = ((p[2] - z_min) / bin_w) as usize;
            b = b.min(num_bins - 1);
            bins[b] += 1;
        }

        let peak = if upside_down {
            let max_count = *bins.iter().max().unwrap_or(&1);
            let threshold = (max_count / 10).max(1);
            bins.iter().enumerate()
                .find(|(_, c)| **c >= threshold)
                .map(|(i, _)| i)
                .unwrap_or(0)
        } else {
            bins.iter().enumerate()
                .max_by_key(|(_, c)| *c)
                .map(|(i, _)| i)
                .unwrap_or(0)
        };
        let peak_z = z_min + (peak as f32 + 0.5) * bin_w;
        let z_low = peak_z - expand;
        let z_high = peak_z + expand;

        let mut seed_start = 0;
        for (i, p) in cloud.iter().enumerate() {
            if p[2] >= z_low { seed_start = i; break; }
        }
        let mut seed_end = n;
        for (i, p) in cloud.iter().enumerate().rev() {
            if p[2] <= z_high { seed_end = i + 1; break; }
        }
        let n_seed = seed_end - seed_start;

        let (n_ground, plane_eq) = if n_seed >= 10 {
            ransac_and_grow(cloud, n, seed_start, n_seed, distance, iterations)
        } else {
            for i in 0..n_seed { cloud.swap(seed_start + i, i); }
            (n_seed, None)
        };

        if upside_down { for p in cloud.iter_mut() { p[2] = -p[2]; } }

        let mut ground_box = Box3D::empty_box();
        ground_box.cloud2box(&cloud[..n_ground].to_vec());
        let bud = CldBud::new(ground_box, 0, "ground".to_string(), 1.0);

        (n_ground, vec![bud], plane_eq)
    }
}

fn ransac_and_grow(
    cloud: &mut [[f32; 3]], n: usize,
    seed_start: usize, n_seed: usize,
    distance: f32, iterations: usize,
) -> (usize, Option<[f32; 4]>) {
    let seed_cloud: Vec<[f32; 3]> = cloud[seed_start..seed_start + n_seed].to_vec();

    let mut best_plane = ([0.0f32; 3], [0.0f32; 3]);
    let mut best_count = 0usize;

    for _ in 0..iterations {
        let idx = select_some(0, n_seed, 3);
        if idx.len() < 3 { continue; }
        let (p1, p2, p3) = (&seed_cloud[idx[0]], &seed_cloud[idx[1]], &seed_cloud[idx[2]]);

        let v1 = [p2[0]-p1[0], p2[1]-p1[1], p2[2]-p1[2]];
        let v2 = [p3[0]-p1[0], p3[1]-p1[1], p3[2]-p1[2]];
        let nx = v1[1]*v2[2] - v1[2]*v2[1];
        let ny = v1[2]*v2[0] - v1[0]*v2[2];
        let nz = v1[0]*v2[1] - v1[1]*v2[0];
        let len = (nx*nx + ny*ny + nz*nz).sqrt();
        if len < 1e-6 { continue; }
        let (nx, ny, nz) = (nx/len, ny/len, nz/len);

        let count = seed_cloud.iter().filter(|p| {
            let dx = p[0] - p1[0]; let dy = p[1] - p1[1]; let dz = p[2] - p1[2];
            (nx*dx + ny*dy + nz*dz).abs() < distance
        }).count();

        if count > best_count {
            best_count = count;
            best_plane = (*p1, [nx, ny, nz]);
        }
    }

    if best_count < 3 {
        for i in 0..n_seed { cloud.swap(seed_start + i, i); }
        return (n_seed, None);
    }

    let (pp, norm) = &best_plane;
    let inlier_mask: Vec<bool> = cloud.iter().map(|p| {
        let dx = p[0]-pp[0]; let dy = p[1]-pp[1]; let dz = p[2]-pp[2];
        (norm[0]*dx + norm[1]*dy + norm[2]*dz).abs() < distance
    }).collect();

    let mut write = 0;
    for read in 0..n {
        if inlier_mask[read] { cloud.swap(read, write); write += 1; }
    }

    let plane_eq = [norm[0], norm[1], norm[2], -(norm[0]*pp[0] + norm[1]*pp[1] + norm[2]*pp[2])];
    (write, Some(plane_eq))
}
