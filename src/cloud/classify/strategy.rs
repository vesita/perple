mod dbscan;
mod range_image;
mod xy_grid_dbscan;
mod lvdot_cluster;

pub use dbscan::DbscanStrategy;
pub use range_image::RangeImageStrategy;
pub use xy_grid_dbscan::XYGridDBSCAN;
pub use lvdot_cluster::LvdotClusterStrategy;

use crate::config::fixif;

/// 聚类策略 trait — 新增策略只需 impl 此 trait + 在工厂注册一行
pub trait ClusteringStrategy: Send {
    /// 执行聚类，返回 (处理后的点集, 簇索引列表)
    fn run(&mut self, points: &[[f32; 3]]) -> (Vec<[f32; 3]>, Vec<Vec<usize>>);

    /// 策略名称（用于日志/可视化）
    fn strategy_name(&self) -> &'static str {
        "unknown"
    }
}

/// 策略工厂 — 根据配置创建对应的策略实例
pub fn create_strategy() -> Box<dyn ClusteringStrategy> {
    let cfg = fixif();
    match cfg.claster.strategy.as_str() {
        "range_image" => {
            log::info!("聚类策略: range_image");
            Box::new(RangeImageStrategy::new())
        }
        "xy_grid_dbscan" => {
            log::info!("聚类策略: xy_grid_dbscan");
            Box::new(XYGridDBSCAN::new())
        }
        "lvdot" => {
            log::info!("聚类策略: lvdot (体素{:.2}m 占用>={})", cfg.claster.voxel_size, 3);
            Box::new(LvdotClusterStrategy::new())
        }
        "dbscan_light" => {
            log::info!("聚类策略: dbscan_light (无内部下采样)");
            Box::new(DbscanStrategy::new_light())
        }
        "dbscan" | "dbscan_adaptive" => {
            log::info!("聚类策略: dbscan (eps_slope={})", cfg.claster.eps_slope);
            Box::new(DbscanStrategy::new())
        }
        _ => {
            log::warn!("未知聚类策略 '{}'，使用默认 dbscan_light", cfg.claster.strategy);
            Box::new(DbscanStrategy::new_light())
        }
    }
}
