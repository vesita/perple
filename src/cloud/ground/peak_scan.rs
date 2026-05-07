use crate::config::fixif;
use crate::utils::boxes::Box3D;
use super::GroundPickStrategy;
use super::super::CldBud;

/// 峰扫描地面提取：从 Z 直方图峰值向下扫描找地面下界，向上扩展找地面上界。
pub struct PeakScan {
    threshold: Option<f32>,
    expand: Option<f32>,
}

impl PeakScan {
    pub fn new() -> Self { Self { threshold: None, expand: None } }
    pub fn with_params(threshold: f32, expand: f32) -> Self {
        Self { threshold: Some(threshold), expand: Some(expand) }
    }
}

impl GroundPickStrategy for PeakScan {
    fn strategy_name(&self) -> &'static str { "peak_scan" }

    fn pick(&mut self, cloud: &mut [[f32; 3]]) -> (usize, Vec<CldBud>, Option<[f32; 4]>) {
        let n = cloud.len();
        if n < 10 {
            return (0, Vec::new(), None);
        }

        let cfg = fixif();
        let upside_down = cfg.upside_down;
        let expand = self.expand.unwrap_or(cfg.ground_expand);
        let threshold = self.threshold.unwrap_or(0.15);
        let num_bins = 128;

        if upside_down { for p in cloud.iter_mut() { p[2] = -p[2]; } }
        cloud.sort_unstable_by(|a, b| a[2].partial_cmp(&b[2]).unwrap());

        let z_min = cloud[0][2];
        let z_max = cloud[n - 1][2];
        let z_range = z_max - z_min;
        if z_range < 1e-6 {
            if upside_down { for p in cloud.iter_mut() { p[2] = -p[2]; } }
            return (0, Vec::new(), None);
        }

        let bin_w = z_range / num_bins as f32;
        let mut bins = vec![0usize; num_bins];
        for p in cloud.iter() {
            let mut b = ((p[2] - z_min) / bin_w) as usize;
            b = b.min(num_bins - 1);
            bins[b] += 1;
        }

        let peak = find_peak_bin(&bins, upside_down);
        let peak_count = bins[peak];
        let peak_z = z_min + (peak as f32 + 0.5) * bin_w;
        let t = (peak_count as f32 * threshold).max(1.0) as usize;

        let mut ground_start_bin = 0;
        for i in (0..peak).rev() {
            if bins[i] < t { ground_start_bin = i + 1; break; }
        }
        let z_lower = z_min + ground_start_bin as f32 * bin_w;
        let z_upper = peak_z + expand;

        let mut ground_start = 0;
        for (i, p) in cloud.iter().enumerate() {
            if p[2] >= z_lower { ground_start = i; break; }
        }
        let mut ground_end = n;
        for (i, p) in cloud.iter().enumerate().rev() {
            if p[2] <= z_upper { ground_end = i + 1; break; }
        }

        let n_ground = if ground_end > ground_start { ground_end - ground_start } else { 0 };

        // 重排地面点到前部
        let ground_pts: Vec<[f32; 3]> = cloud[ground_start..ground_end].to_vec();
        let front: Vec<[f32; 3]> = cloud[..ground_start].to_vec();
        let back: Vec<[f32; 3]> = cloud[ground_end..].to_vec();
        for (i, p) in ground_pts.iter().enumerate() { cloud[i] = *p; }
        let mut idx = n_ground;
        for p in &front { cloud[idx] = *p; idx += 1; }
        for p in &back { cloud[idx] = *p; idx += 1; }

        if upside_down { for p in cloud.iter_mut() { p[2] = -p[2]; } }

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
