mod cc_cluster;
mod ransac_cluster;
mod seq_cluster;
mod dbscan;
mod dbscan_grid;
mod range_image;
mod xy_grid_dbscan;
mod lvdot_cluster;
mod lvdot_qt;

pub use cc_cluster::CcCluster;
pub use ransac_cluster::RansacCluster;
pub use seq_cluster::SeqCluster;
pub use dbscan::DbscanStrategy;
pub use dbscan_grid::DbscanGrid;
pub use range_image::RangeImageStrategy;
pub use xy_grid_dbscan::XYGridDBSCAN;
pub use lvdot_cluster::LvdotClusterStrategy;
pub use lvdot_qt::LvdotQt;

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
    match cfg.cluster.strategy.as_str() {
        "cc_grid" | "cc" => {
            log::info!("聚类策略: cc_grid (连通域)");
            Box::new(CcCluster::new(cfg.cluster.voxel_size.max(0.10), 3))
        }
        "ransac" => {
            log::info!("聚类策略: ransac (RANSAC 线聚类)");
            Box::new(RansacCluster::new(cfg.cluster.cluster_threshold.max(0.05), 50, 10))
        }
        "seq" => {
            log::info!("聚类策略: seq (顺序 SVD 平面聚类)");
            Box::new(SeqCluster::new(cfg.cluster.cluster_threshold.max(0.10), 10))
        }
        "range_image" => {
            log::info!("聚类策略: range_image");
            Box::new(RangeImageStrategy::new())
        }
        "xy_grid_dbscan" | "xy_grid_dbscan_grid" => {
            log::info!("聚类策略: xy_grid_dbscan_grid");
            Box::new(XYGridDBSCAN::new())
        }
        "lvdot_grid" | "lvdot" => {
            log::info!("聚类策略: lvdot_grid (体素{:.2}m 占用>={})", cfg.cluster.voxel_size, 3);
            Box::new(LvdotClusterStrategy::new())
        }
        "lvdot_qt" => {
            log::info!("聚类策略: lvdot_qt (叶节点占用>={})", 3);
            Box::new(LvdotQt::new())
        }
        "dbscan_light" => {
            log::info!("聚类策略: dbscan_light (无内部下采样)");
            Box::new(DbscanStrategy::new_light())
        }
        "dbscan_qt" | "dbscan" | "dbscan_adaptive" => {
            let min_pts = cfg.cluster.min_points_per_cluster.unwrap_or(3) as usize;
            log::info!("聚类策略: dbscan_qt (eps_slope={}, min_pts={})", cfg.cluster.eps_slope, min_pts);
            // 框架层 RadiusOutlierRemoval 已做离群点剔除, 此处 min_points 控制 DBSCAN 核心点密度门槛
            Box::new(DbscanStrategy::with_params(
                cfg.cluster.merge_patience,
                cfg.cluster.eps_slope,
                min_pts,
                cfg.cluster.max_points_per_node.unwrap_or(20),
                cfg.cluster.max_tree_depth.unwrap_or(10),
                cfg.cluster.voxel_size,
            ))
        }
        "dbscan_grid" => {
            log::info!("聚类策略: dbscan_grid");
            Box::new(DbscanGrid::new(cfg.cluster.voxel_size.max(0.15), 5))
        }
        _ => {
            log::warn!("未知聚类策略 '{}'，使用默认 dbscan_qt", cfg.cluster.strategy);
            Box::new(DbscanStrategy::new_light())
        }
    }
}
