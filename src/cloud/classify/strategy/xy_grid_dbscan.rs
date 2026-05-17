use super::ClusteringStrategy;
use crate::cloud::wall::{WallPickStrategy, BevLsd, XYGrid, cluster_obstacles_with_indices, xy_dbscan};

/// XY 网格预过滤 + DBSCAN 聚类策略。
///
/// 管线（墙提前后行为一致）：
/// 1. 网格连通域聚类生成 AABB（`cluster_obstacles_with_indices`）
/// 2. 按 box 中心 XY 距离过滤（12m 范围）
/// 3. XYGrid 下采样
/// 4. xy_dbscan 精化聚类
pub struct XYGridDBSCAN {
    wall: Box<dyn WallPickStrategy>,
    skip_wall: bool,
    cell_size: f32,
    min_pts: usize,
    min_edge: f32,
    max_range: f32,
    dbscan_eps: f32,
    dbscan_min_pts: usize,
    max_target_pts: usize,
}

impl XYGridDBSCAN {
    pub fn new() -> Self {
        Self {
            wall: Box::new(BevLsd::with_params(0.05, 20).with_min_extent(0.0)),
            skip_wall: false,
            cell_size: 0.30,
            min_pts: 3,
            min_edge: 0.05,
            max_range: 12.0,
            dbscan_eps: 0.30,
            dbscan_min_pts: 5,
            max_target_pts: 4000,
        }
    }

    pub fn with_params(
        wall: Box<dyn WallPickStrategy>,
        cell_size: f32,
        min_pts: usize,
        max_range: f32,
        dbscan_eps: f32,
        dbscan_min_pts: usize,
    ) -> Self {
        Self {
            wall,
            skip_wall: false,
            cell_size,
            min_pts,
            min_edge: 0.05,
            max_range,
            dbscan_eps,
            dbscan_min_pts,
            max_target_pts: 4000,
        }
    }

    pub fn with_pre_extracted_wall(mut self) -> Self {
        self.skip_wall = true;
        self
    }

    pub fn with_max_pts(mut self, n: usize) -> Self {
        self.max_target_pts = n;
        self
    }
}

impl ClusteringStrategy for XYGridDBSCAN {
    fn run(&mut self, non_ground: &[[f32; 3]]) -> (Vec<[f32; 3]>, Vec<Vec<usize>>) {
        let n = non_ground.len();
        if n == 0 { return (Vec::new(), Vec::new()); }

        let owned: Vec<[f32; 3]> = if self.skip_wall {
            non_ground.to_vec()
        } else {
            let mut wall_buf = non_ground.to_vec();
            let (n_wall, _planes) = self.wall.pick(&mut wall_buf);
            if n_wall >= n { return (non_ground.to_vec(), Vec::new()); }
            wall_buf[n_wall..].to_vec()
        };

        if owned.is_empty() {
            return (non_ground.to_vec(), Vec::new());
        }

        let (_boxes, box_indices) = cluster_obstacles_with_indices(
            &owned, self.cell_size, self.min_pts, self.min_edge, self.max_range,
        );

        let mut in_range_pts: Vec<[f32; 3]> = Vec::new();
        for indices in &box_indices {
            for &idx in indices {
                in_range_pts.push(owned[idx]);
            }
        }

        if in_range_pts.is_empty() {
            return (non_ground.to_vec(), Vec::new());
        }

        let (sampled, _map) = XYGrid::grid_downsample(&in_range_pts, self.max_target_pts);
        let clusters = xy_dbscan(&sampled, self.dbscan_eps, self.dbscan_min_pts);

        (sampled, clusters)
    }

    fn strategy_name(&self) -> &'static str {
        "xy_grid_dbscan"
    }
}
