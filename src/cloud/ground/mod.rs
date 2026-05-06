mod histogram;
mod histoseed;
mod ransac;
mod peak_down;
mod gpf;

pub use histogram::HistogramExpand;
pub use histoseed::HistoseedPlane;
pub use ransac::RansacGround;
pub use peak_down::PeakDownExpandUp;
pub use gpf::GpfGround;

use crate::config::fixif;

/// 地面提取结果
pub struct GroundResult {
    /// 地面点数量（cloud[0..n_ground] 已原地交换为地面点）
    pub n_ground: usize,
    /// 每个点是否为地面的掩码（原始顺序）
    pub ground_mask: Vec<bool>,
    /// 平面方程 [a, b, c, d]，可选
    pub plane_eq: Option<[f32; 4]>,
}

/// 地面提取策略 trait
///
/// 所有策略统一接收 `&mut [[f32; 3]]` 点云，返回 `GroundResult`。
/// 策略内部处理 upside_down 翻转，调用方无需关心。
pub trait GroundStrategy: Send {
    /// 执行地面提取，cloud 会被原地修改（排序/翻转等）
    fn extract(&mut self, cloud: &mut [[f32; 3]]) -> GroundResult;

    /// 策略名称（用于日志/配置）
    fn strategy_name(&self) -> &'static str;
}

/// 策略工厂 — 根据配置创建对应的地面提取策略
pub fn create_ground_strategy() -> Box<dyn GroundStrategy> {
    let cfg = fixif();
    match cfg.ground_strategy.as_str() {
        "histogram" => {
            log::info!("地面策略: histogram (expand={})", cfg.ground_expand);
            Box::new(HistogramExpand::new())
        }
        "histoseed" => {
            log::info!("地面策略: histoseed (expand={}, ransac_dist={}, iter={})",
                cfg.ground_expand, cfg.ground_ransac_distance, cfg.ground_ransac_iterations);
            Box::new(HistoseedPlane::new())
        }
        "ransac" => {
            log::info!("地面策略: ransac (dist={}, iter={})",
                cfg.ground_ransac_distance, cfg.ground_ransac_iterations);
            Box::new(RansacGround::new())
        }
        "peak_down" => {
            log::info!("地面策略: peak_down");
            Box::new(PeakDownExpandUp::new())
        }
        "gpf" => {
            log::info!("地面策略: gpf");
            Box::new(GpfGround::new())
        }
        _ => {
            log::warn!("未知地面策略 '{}'，使用默认 histoseed", cfg.ground_strategy);
            Box::new(HistoseedPlane::new())
        }
    }
}
