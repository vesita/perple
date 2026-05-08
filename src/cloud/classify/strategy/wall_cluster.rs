use std::collections::VecDeque;

use super::ClusteringStrategy;
use crate::cloud::wall::{WallPickStrategy, XYRansacWall, XYGrid, cluster_obstacles_with_indices};

/// Wall → Box 过滤 → 下采样 + DBSCAN 聚类策略。
///
/// 管线：
/// 1. 墙面提取（XYRansacWall）分离墙面/非墙面点
/// 2. 网格连通域聚类生成 AABB
/// 3. 按 box 中心 XY 距离过滤（12m 范围）
/// 4. 对 12m 内的点下采样 + DBSCAN 精化聚类
pub struct WallClusterStrategy {
    wall: Box<dyn WallPickStrategy>,
    // 上游已提取墙体时跳过内部墙提
    skip_wall: bool,
    // cluster_obstacles 参数
    cell_size: f32,
    min_pts: usize,
    min_edge: f32,
    max_range: f32,
    // DBSCAN 精化参数
    dbscan_eps: f32,
    dbscan_min_pts: usize,
    max_target_pts: usize,
}

impl WallClusterStrategy {
    pub fn new() -> Self {
        Self {
            wall: Box::new(XYRansacWall::with_params(0.05, 50, 30).with_seed(42)),
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

    /// 上游已提取墙体，跳过内部墙提直接做 box+DBSCAN。
    pub fn with_pre_extracted_wall(mut self) -> Self {
        self.skip_wall = true;
        self
    }

    pub fn with_max_pts(mut self, n: usize) -> Self {
        self.max_target_pts = n;
        self
    }
}

/// XY 平面 DBSCAN，使用 XYGrid 空间索引。
///
/// 返回簇索引列表，每个簇是采样点集中的索引。
fn xy_dbscan(points: &[[f32; 3]], eps: f32, min_pts: usize) -> Vec<Vec<usize>> {
    let n = points.len();
    if n == 0 { return Vec::new(); }

    let grid = XYGrid::new(points, eps);
    let mut visited = vec![false; n];
    let mut clusters = Vec::new();
    let mut nbr_buf = Vec::new();

    for i in 0..n {
        if visited[i] { continue; }
        visited[i] = true;

        nbr_buf.clear();
        grid.query_neighbors(points, points[i][0], points[i][1], eps, &mut nbr_buf);
        if nbr_buf.len() < min_pts { continue; }

        let mut cluster = vec![i];
        let mut queue: VecDeque<usize> = VecDeque::new();
        let seed_nbrs: Vec<usize> = nbr_buf.drain(..).collect();
        for &j in &seed_nbrs {
            if !visited[j] {
                visited[j] = true;
                queue.push_back(j);
            }
        }

        while let Some(cur) = queue.pop_front() {
            cluster.push(cur);
            nbr_buf.clear();
            grid.query_neighbors(points, points[cur][0], points[cur][1], eps, &mut nbr_buf);
            if nbr_buf.len() >= min_pts {
                for &j in &nbr_buf {
                    if !visited[j] {
                        visited[j] = true;
                        queue.push_back(j);
                    }
                }
            }
        }
        clusters.push(cluster);
    }

    clusters
}

impl ClusteringStrategy for WallClusterStrategy {
    fn run(&mut self, non_ground: &[[f32; 3]]) -> (Vec<[f32; 3]>, Vec<Vec<usize>>) {
        let n = non_ground.len();
        if n == 0 { return (Vec::new(), Vec::new()); }

        // 1. 墙面提取（上游已提取时可跳过）
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

        // 2. 网格连通域聚类 → box + 点索引
        let (_boxes, box_indices) = cluster_obstacles_with_indices(
            &owned, self.cell_size, self.min_pts, self.min_edge, self.max_range,
        );

        // 3. 收集 12m 内 box 对应的所有点
        let mut in_range_pts: Vec<[f32; 3]> = Vec::new();
        for indices in &box_indices {
            for &idx in indices {
                in_range_pts.push(owned[idx]);
            }
        }

        if in_range_pts.is_empty() {
            return (non_ground.to_vec(), Vec::new());
        }

        // 4. 下采样
        let (sampled, _map) = XYGrid::grid_downsample(&in_range_pts, self.max_target_pts);

        // 5. DBSCAN 精化
        let clusters = xy_dbscan(&sampled, self.dbscan_eps, self.dbscan_min_pts);

        (sampled, clusters)
    }

    fn strategy_name(&self) -> &'static str {
        "wall_cluster"
    }
}
