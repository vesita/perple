//! 地面提取模块
//!
//! 默认使用 Z-直方图 + expand 算法，可通过 [`extract_ground`] 指定 expand 参数。
//! 可通过 [`GroundPickStrategy`] trait 扩展不同的地面提取策略。

use super::CldBud;
use crate::utils::boxes::Box3D;

/// 提取地面点。
///
/// 返回 `(地面点数, ground_mask)`，其中 `ground_mask[i]` 对应输入 `cloud` 的第 i 个点。
///
/// 默认 expand=0.20 效果较好，可通过 `expand` 参数调整。
pub fn extract_ground(cloud: &mut [[f32; 3]], expand: f32, upside_down: bool) -> (usize, Vec<bool>) {
    histogram_expand(cloud, expand, upside_down)
}

/// Z-直方图 + expand 地面提取
fn histogram_expand(cloud: &mut [[f32; 3]], expand: f32, upside_down: bool) -> (usize, Vec<bool>) {
    let n = cloud.len();
    if n < 10 { return (0, vec![false; n]); }
    if upside_down { for p in cloud.iter_mut() { p[2] = -p[2]; } }
    let mut indexed: Vec<(usize, [f32; 3])> = (0..n).zip(cloud.iter().copied()).collect();
    indexed.sort_by(|a, b| a.1[2].partial_cmp(&b.1[2]).unwrap());
    for (i, (_, p)) in indexed.iter().enumerate() { cloud[i] = *p; }

    let z_min = cloud[0][2];
    let z_max = cloud[n - 1][2];
    let z_range = z_max - z_min;
    if z_range < 1e-6 {
        if upside_down { for p in cloud.iter_mut() { p[2] = -p[2]; } }
        return (0, vec![false; n]);
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
    for i in start..end {
        ground_mask[indexed[i].0] = true;
    }

    if upside_down { for p in cloud.iter_mut() { p[2] = -p[2]; } }
    (n_ground, ground_mask)
}

fn find_peak_bin(bins: &[usize], upside_down: bool) -> usize {
    if upside_down {
        let avg = bins.iter().sum::<usize>() / bins.len().max(1);
        bins.iter()
            .enumerate()
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

/// 地面提取策略 trait — 新增策略只需 impl 此 trait + 在工厂注册一行
pub trait GroundPickStrategy: Send {
    /// 执行地面提取，原地修改点云。
    ///
    /// 返回 `(地面点数, 地面 CldBud 列表, 平面方程)`。
    /// 调用后 `cloud[..n_ground]` 为地面点，`cloud[n_ground..]` 为非地面点。
    fn pick(&mut self, cloud: &mut [[f32; 3]]) -> (usize, Vec<CldBud>, Option<[f32; 4]>);

    /// 策略名称（用于日志/可视化）
    fn strategy_name(&self) -> &'static str {
        "unknown"
    }
}

/// Z-直方图 + expand 地面提取策略（默认, expand=0.20）
///
/// 算法：
/// 1. 按 Z 排序点云
/// 2. 建 128-bin Z 直方图，找峰值 bin
/// 3. 取 peak_z ± expand 范围内所有点为地面
/// 4. 地面点前移，非地面点后置
///
/// 对应配置：`ground_expand` 和 `upside_down`
pub struct HistogramExpandStrategy {
    expand: f32,
    upside_down: bool,
}

impl HistogramExpandStrategy {
    /// 从全局配置创建
    pub fn new() -> Self {
        let cfg = crate::config::fixif();
        Self {
            expand: cfg.ground_expand,
            upside_down: cfg.upside_down,
        }
    }

    /// 直接参数化（用于基准测试/手动配置）
    pub fn with_params(expand: f32, upside_down: bool) -> Self {
        Self { expand, upside_down }
    }
}

impl GroundPickStrategy for HistogramExpandStrategy {
    fn pick(&mut self, cloud: &mut [[f32; 3]]) -> (usize, Vec<CldBud>, Option<[f32; 4]>) {
        let n = cloud.len();
        if n < 10 {
            return (0, Vec::new(), None);
        }

        // 倒置 LiDAR：取反 Z 使地面回到 LOW Z 端
        if self.upside_down {
            for p in cloud.iter_mut() {
                p[2] = -p[2];
            }
        }

        cloud.sort_unstable_by(|a, b| a[2].partial_cmp(&b[2]).unwrap());

        let z_min = cloud[0][2];
        let z_max = cloud[n - 1][2];
        let z_range = z_max - z_min;
        if z_range < 1e-6 {
            if self.upside_down {
                for p in cloud.iter_mut() { p[2] = -p[2]; }
            }
            return (0, Vec::new(), None);
        }

        // Z 直方图
        let num_bins = 128;
        let bin_w = z_range / num_bins as f32;
        let mut bins = vec![0usize; num_bins];
        for p in cloud.iter() {
            let mut b = ((p[2] - z_min) / bin_w) as usize;
            b = b.min(num_bins - 1);
            bins[b] += 1;
        }

        // 找地面峰值
        let peak = if self.upside_down {
            // 倒置：从底部向上扫描找第一个显著 bin
            let max_count = *bins.iter().max().unwrap_or(&1);
            let threshold = (max_count / 10).max(1);
            bins.iter()
                .enumerate()
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
        let z_low = peak_z - self.expand;
        let z_high = peak_z + self.expand;

        // 地面点范围（在已排序的点云中连续）
        let start = cloud.partition_point(|p| p[2] < z_low);
        let end = cloud.partition_point(|p| p[2] <= z_high);
        let n_ground = end - start;

        if n_ground == 0 {
            if self.upside_down {
                for p in cloud.iter_mut() { p[2] = -p[2]; }
            }
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

        // 恢复 Z
        if self.upside_down {
            for p in cloud.iter_mut() { p[2] = -p[2]; }
        }

        // 构造地面 CldBud
        let mut ground_box = Box3D::empty_box();
        ground_box.cloud2box(&cloud[..n_ground].to_vec());
        let bud = CldBud::new(ground_box, 0, "ground".to_string(), 1.0);

        (n_ground, vec![bud], None)
    }

    fn strategy_name(&self) -> &'static str {
        "expand0.20"
    }
}

/// 创建默认地面提取策略（HistogramExpand, expand=0.20）
pub fn create_ground_strategy() -> Box<dyn GroundPickStrategy> {
    Box::new(HistogramExpandStrategy::new())
}
