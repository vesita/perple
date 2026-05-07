mod histogram;
mod histoseed;
mod ransac;
mod peak_scan;
mod gpf;

pub use histogram::HistogramExpand;
pub use histogram::HistogramExpand as HistogramExpandStrategy;
pub use histoseed::HistoseedPlane;
pub use ransac::RansacGround;
pub use peak_scan::PeakScan;
pub use gpf::GpfGround;

use super::CldBud;
use crate::config::fixif;

/// 地面提取策略 trait
///
/// 所有策略统一接收 `&mut [[f32; 3]]` 点云，返回 `(地面点数, 地面 CldBud, 平面方程)`。
/// 调用后 `cloud[..n_ground]` 为地面点，`cloud[n_ground..]` 为非地面点。
pub trait GroundPickStrategy: Send {
    fn pick(&mut self, cloud: &mut [[f32; 3]]) -> (usize, Vec<CldBud>, Option<[f32; 4]>);
    fn strategy_name(&self) -> &'static str { "unknown" }
}

/// 创建地面提取策略（从配置读取 ground_strategy 分发）
pub fn create_ground_strategy() -> Box<dyn GroundPickStrategy> {
    let cfg = fixif();
    match cfg.ground_strategy.as_str() {
        "peak_scan" => Box::new(PeakScan::new()),
        "histoseed" => Box::new(HistoseedPlane::new()),
        "ransac" => Box::new(RansacGround::new()),
        "gpf" => Box::new(GpfGround::new()),
        _ => Box::new(HistogramExpand::new()),
    }
}
