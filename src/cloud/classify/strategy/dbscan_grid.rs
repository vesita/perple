use super::ClusteringStrategy;
use crate::cloud::wall::xy_dbscan;

/// XY 网格 DBSCAN 策略（dbscan_grid）。
///
/// 使用 XYGrid 加速邻域查询的 DBSCAN 聚类。
/// 与 dbscan_qt 的区别：使用 HashMap 网格索引而非四叉树。
///
/// 聚类策略：dbscan
/// 空间索引：grid（XYGrid）
pub struct DbscanGrid {
    eps: f32,
    min_pts: usize,
}

impl DbscanGrid {
    pub fn new(eps: f32, min_pts: usize) -> Self {
        Self { eps, min_pts }
    }
}

impl ClusteringStrategy for DbscanGrid {
    fn run(&mut self, points: &[[f32; 3]]) -> (Vec<[f32; 3]>, Vec<Vec<usize>>) {
        let clusters = xy_dbscan(points, self.eps, self.min_pts);
        (points.to_vec(), clusters)
    }

    fn strategy_name(&self) -> &'static str {
        "dbscan_grid"
    }
}
