use crate::config::fixif;
use super::{GroundResult, GroundStrategy};

/// Z-直方图 + expand 地面提取
///
/// 最简单的地面策略：按 Z 坐标建直方图，找峰值，峰值 ± expand 范围内为地面。
pub struct HistogramExpand;

impl HistogramExpand {
    pub fn new() -> Self { Self }
}

impl GroundStrategy for HistogramExpand {
    fn strategy_name(&self) -> &'static str { "histogram" }

    fn extract(&mut self, cloud: &mut [[f32; 3]]) -> GroundResult {
        let n = cloud.len();
        if n < 10 {
            return GroundResult { n_ground: 0, ground_mask: vec![false; n], plane_eq: None };
        }

        let cfg = fixif();
        let upside_down = cfg.upside_down;
        let expand = cfg.ground_expand;

        if upside_down { for p in cloud.iter_mut() { p[2] = -p[2]; } }

        let mut indexed: Vec<(usize, [f32; 3])> = (0..n).zip(cloud.iter().copied()).collect();
        indexed.sort_by(|a, b| a.1[2].partial_cmp(&b.1[2]).unwrap());
        for (i, (_, p)) in indexed.iter().enumerate() { cloud[i] = *p; }

        let z_min = cloud[0][2];
        let z_max = cloud[n - 1][2];
        let z_range = z_max - z_min;
        if z_range < 1e-6 {
            if upside_down { for p in cloud.iter_mut() { p[2] = -p[2]; } }
            return GroundResult { n_ground: 0, ground_mask: vec![false; n], plane_eq: None };
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

        let mut start = 0;
        for (i, p) in cloud.iter().enumerate() {
            if p[2] >= z_low { start = i; break; }
        }
        let mut end = n;
        for (i, p) in cloud.iter().enumerate().rev() {
            if p[2] <= z_high { end = i + 1; break; }
        }

        let n_ground = end - start;
        let mut ground_mask = vec![false; n];
        for i in start..end { ground_mask[indexed[i].0] = true; }

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
