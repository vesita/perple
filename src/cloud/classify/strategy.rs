mod cc_cluster;
mod ransac_cluster;
mod seq_cluster;
mod dbscan;
mod dbscan_grid;
mod range_image;
mod xy_grid_dbscan;
mod lvdot_cluster;
mod prune_qt;

pub use cc_cluster::CcCluster;
pub use ransac_cluster::RansacCluster;
pub use seq_cluster::SeqCluster;
pub use dbscan::DbscanStrategy;
pub use dbscan_grid::DbscanGrid;
pub use range_image::RangeImageStrategy;
pub use xy_grid_dbscan::XYGridDBSCAN;
pub use lvdot_cluster::LvdotClusterStrategy;
pub use prune_qt::PruneQt;

use crate::config::fixif;
use crate::cloud::wall::BevLsd;

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
pub fn create_strategy(pre_extracted_wall: bool) -> Box<dyn ClusteringStrategy> {
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
        "xy_grid_dbscan" => {
            let cell = cfg.cluster.voxel_size.max(0.05);
            let eps = cfg.cluster.merge_patience;
            let min_pts = cfg.cluster.min_points_per_cluster.unwrap_or(3) as usize;
            log::info!("聚类策略: xy_grid_dbscan (cell={:.2}, eps={}, min_pts={})", cell, eps, min_pts);
            let wall = Box::new(BevLsd::with_params(cfg.wall_distance, 20));
            Box::new(XYGridDBSCAN::with_params(wall, cell, min_pts, cfg.max_range, eps, min_pts)
                .with_pre_extracted_wall())
        }
        "xy_grid_dbscan_grid" => {
            log::info!("聚类策略: xy_grid_dbscan_grid");
            Box::new(XYGridDBSCAN::new())
        }
        "lvdot_grid" | "lvdot" => {
            let min_pts = cfg.cluster.min_points_per_cluster.unwrap_or(5) as usize;
            log::info!("聚类策略: lvdot_grid (体素{:.2}m 占用>={}, eps={}, min_pts={})",
                cfg.cluster.voxel_size, cfg.cluster.min_occ, cfg.cluster.merge_patience, min_pts);
            let s = LvdotClusterStrategy::new()
                .with_voxel(cfg.cluster.voxel_size, cfg.cluster.min_occ)
                .with_dbscan(cfg.cluster.merge_patience, min_pts);
            Box::new(if pre_extracted_wall { s.with_pre_extracted_wall() } else { s })
        }
        "prune_qt" | "lvdot_qt" => {
            let min_pts = cfg.cluster.min_points_per_cluster.unwrap_or(5) as usize;
            let mut s = PruneQt::new()
                .with_params(cfg.cluster.min_occ, cfg.cluster.merge_patience, min_pts);
            if cfg.cluster.adaptive_depth {
                let c = &cfg.cluster;
                log::info!("  adaptive_depth: res0={}, r0={}, beta={}, max_depth={}",
                    c.adaptive_res0, c.adaptive_r0, c.adaptive_beta, c.adaptive_global_max_depth);
                s = s.with_adaptive_depth(
                    c.adaptive_res0, c.adaptive_r0,
                    c.adaptive_beta, c.adaptive_global_max_depth);
            }
            Box::new(if pre_extracted_wall { s.with_pre_extracted_wall() } else { s })
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
