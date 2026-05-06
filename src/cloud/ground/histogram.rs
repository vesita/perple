use crate::config::fixif;
use crate::utils::boxes::Box3D;
use super::GroundPickStrategy;
use super::super::CldBud;

/// Z-直方图 + expand 地面提取
///
/// 最简单的地面策略：按 Z 坐标建直方图，找峰值，峰值 ± expand 范围内为地面。
pub struct HistogramExpand {
    expand: Option<f32>,
}

impl HistogramExpand {
    pub fn new() -> Self { Self { expand: None } }
    pub fn with_expand(expand: f32) -> Self { Self { expand: Some(expand) } }
}

impl GroundPickStrategy for HistogramExpand {
    fn strategy_name(&self) -> &'static str { "histogram" }

    fn pick(&mut self, cloud: &mut [[f32; 3]]) -> (usize, Vec<CldBud>, Option<[f32; 4]>) {
        let n = cloud.len();
        if n < 10 {
            return (0, Vec::new(), None);
        }

        let cfg = fixif();
        let upside_down = cfg.upside_down;
        let expand = self.expand.unwrap_or(cfg.ground_expand);

        if upside_down { for p in cloud.iter_mut() { p[2] = -p[2]; } }

        cloud.sort_unstable_by(|a, b| a[2].partial_cmp(&b[2]).unwrap());

        let z_min = cloud[0][2];
        let z_max = cloud[n - 1][2];
        let z_range = z_max - z_min;
        if z_range < 1e-6 {
            if upside_down { for p in cloud.iter_mut() { p[2] = -p[2]; } }
            return (0, Vec::new(), None);
        }

        let num_bins = 128;
        let bin_w = z_range / num_bins as f32;
        let mut bins = vec![0usize; num_bins];
        for p in cloud.iter() {
            let mut b = ((p[2] - z_min) / bin_w) as usize;
            b = b.min(num_bins - 1);
            bins[b] += 1;
        }

        let peak = find_peak_bin(&bins, upside_down);
        let peak_z = z_min + (peak as f32 + 0.5) * bin_w;
        let z_low = peak_z - expand;
        let z_high = peak_z + expand;

        let start = cloud.partition_point(|p| p[2] < z_low);
        let end = cloud.partition_point(|p| p[2] <= z_high);
        let n_ground = end - start;

        if n_ground == 0 {
            if upside_down { for p in cloud.iter_mut() { p[2] = -p[2]; } }
            return (0, Vec::new(), None);
        }

        // 重排：地面点移到 [0..n_ground)，其余在 [n_ground..)
        let ground_pts: Vec<[f32; 3]> = cloud[start..end].to_vec();
        let front: Vec<[f32; 3]> = cloud[..start].to_vec();
        let back: Vec<[f32; 3]> = cloud[end..].to_vec();
        for (i, p) in ground_pts.iter().enumerate() {
            cloud[i] = *p;
        }
        let mut idx = n_ground;
        for p in &front {
            cloud[idx] = *p;
            idx += 1;
        }
        for p in &back {
            cloud[idx] = *p;
            idx += 1;
        }

        if upside_down { for p in cloud.iter_mut() { p[2] = -p[2]; } }

        // 构造地面 CldBud
        let mut ground_box = Box3D::empty_box();
        ground_box.cloud2box(&cloud[..n_ground].to_vec());
        let bud = CldBud::new(ground_box, 0, "ground".to_string(), 1.0);

        (n_ground, vec![bud], None)
    }
}

fn find_peak_bin(bins: &[usize], upside_down: bool) -> usize {
    if upside_down {
        let avg = bins.iter().sum::<usize>() / bins.len().max(1);
        bins.iter().enumerate()
            .find(|(_, c)| **c > avg)
            .map(|(i, _)| i)
            .unwrap_or(0)
    } else {
        bins.iter().enumerate()
            .max_by_key(|(_, c)| *c)
            .map(|(i, _)| i)
            .unwrap_or(0)
    }
}
