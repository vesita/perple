use crate::config::fixif;
use super::{GroundResult, GroundStrategy};

/// 峰下扫 + 上扩 地面提取
///
/// 从直方图峰值向下扫描到密度低于阈值处作为地面下界，
/// 再从峰值向上扩展 expand 范围作为地面上界。
pub struct PeakDownExpandUp;

impl PeakDownExpandUp {
    pub fn new() -> Self { Self }
}

impl GroundStrategy for PeakDownExpandUp {
    fn strategy_name(&self) -> &'static str { "peak_down" }

    fn extract(&mut self, cloud: &mut [[f32; 3]]) -> GroundResult {
        let n = cloud.len();
        if n < 10 {
            return GroundResult { n_ground: 0, ground_mask: vec![false; n], plane_eq: None };
        }

        let cfg = fixif();
        let upside_down = cfg.upside_down;
        let expand = cfg.ground_expand;
        let threshold = 0.10; // 密度阈值比例
        let num_bins = 128;

        let mut indexed: Vec<(usize, [f32; 3])> = (0..n).zip(cloud.iter().copied()).collect();
        if upside_down { for p in cloud.iter_mut() { p[2] = -p[2]; } }
        indexed.sort_by(|a, b| a.1[2].partial_cmp(&b.1[2]).unwrap());
        for (i, (_, p)) in indexed.iter().enumerate() { cloud[i] = *p; }

        let z_min = cloud[0][2];
        let z_max = cloud[n - 1][2];
        let z_range = z_max - z_min;
        if z_range < 1e-6 {
            if upside_down { for p in cloud.iter_mut() { p[2] = -p[2]; } }
            return GroundResult { n_ground: 0, ground_mask: vec![false; n], plane_eq: None };
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
        let mut ground_mask = vec![false; n];
        for i in ground_start..ground_end { ground_mask[indexed[i].0] = true; }

        if upside_down { for p in cloud.iter_mut() { p[2] = -p[2]; } }

        GroundResult { n_ground, ground_mask, plane_eq: None }
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
