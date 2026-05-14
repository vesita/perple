use super::ClusteringStrategy;
use crate::cloud::wall::{WallPickStrategy, BevEdLines, XYGrid, cluster_obstacles_with_indices, xy_dbscan};

/// LV-DOT 风格聚类策略（lvdot_grid）：墙体提取 → 体素占用下采样 → DBSCAN。
///
/// 对应 LV-DOT 原版的视觉深度管线：
/// 1. 墙面提取（XYRansacWall）分离墙面/非墙面点
/// 2. [可选] 网格连通域粗聚类生成 AABB，过滤远距/小 box
/// 3. LV-DOT 体素占用下采样（≥min_occ 点 → 输出质心，否则丢弃）
/// 4. XY 平面 DBSCAN 精化聚类
///
/// 与 WallClusterStrategy 的区别：
/// - 使用 voxel_occupancy_downsample（占位过滤+压缩合一）替代 grid_downsample
/// - LV-DOT 原版的 voxelFilter 即此思路：够密的格子输出一个代表点
pub struct LvdotClusterStrategy {
    wall: Box<dyn WallPickStrategy>,
    // 上游已提取墙体时跳过内部墙提
    skip_wall: bool,
    // LV-DOT voxel filter
    voxel_size: f32,
    min_occ: usize,
    // box 预聚类（可选）
    use_box_filter: bool,
    box_cell_size: f32,
    box_min_pts: usize,
    box_max_range: f32,
    // DBSCAN
    dbscan_eps: f32,
    dbscan_min_pts: usize,
}

impl LvdotClusterStrategy {
    pub fn new() -> Self {
        Self {
            wall: Box::new(BevEdLines::with_params(0.05, 20).with_min_extent(0.0)),
            skip_wall: false,
            voxel_size: 0.10,
            min_occ: 3,
            use_box_filter: false,
            box_cell_size: 0.30,
            box_min_pts: 3,
            box_max_range: 12.0,
            dbscan_eps: 0.30,
            dbscan_min_pts: 5,
        }
    }

    /// 上游已提取墙体，跳过内部墙提直接做体素过滤+DBSCAN。
    pub fn with_pre_extracted_wall(mut self) -> Self {
        self.skip_wall = true;
        self
    }

    /// LV-DOT 直连模式：墙体提取 → 体素下采样 → DBSCAN（无 box 预聚类）
    pub fn direct(voxel_size: f32, min_occ: usize, dbscan_eps: f32, dbscan_min_pts: usize) -> Self {
        Self {
            voxel_size,
            min_occ,
            dbscan_eps,
            dbscan_min_pts,
            use_box_filter: false,
            ..Self::new()
        }
    }

    /// LV-DOT box 模式：墙体提取 → box 预聚类过滤 → 体素下采样 → DBSCAN
    pub fn with_box_filter(
        mut self,
        box_cell_size: f32,
        box_min_pts: usize,
        box_max_range: f32,
    ) -> Self {
        self.use_box_filter = true;
        self.box_cell_size = box_cell_size;
        self.box_min_pts = box_min_pts;
        self.box_max_range = box_max_range;
        self
    }

    pub fn with_voxel(mut self, voxel_size: f32, min_occ: usize) -> Self {
        self.voxel_size = voxel_size;
        self.min_occ = min_occ;
        self
    }

    pub fn with_dbscan(mut self, eps: f32, min_pts: usize) -> Self {
        self.dbscan_eps = eps;
        self.dbscan_min_pts = min_pts;
        self
    }
}

impl ClusteringStrategy for LvdotClusterStrategy {
    fn run(&mut self, points: &[[f32; 3]]) -> (Vec<[f32; 3]>, Vec<Vec<usize>>) {
        let n = points.len();
        if n == 0 { return (Vec::new(), Vec::new()); }

        // 1. 墙面提取（上游已提取时可跳过）
        let non_wall: Vec<[f32; 3]> = if self.skip_wall {
            points.to_vec()
        } else {
            let mut buf = points.to_vec();
            let (n_wall, _planes) = self.wall.pick(&mut buf);
            if n_wall >= n { return (points.to_vec(), Vec::new()); }
            buf[n_wall..].to_vec()
        };

        let cluster_input: Vec<[f32; 3]> = if self.use_box_filter && !non_wall.is_empty() {
            // 2a. box 预聚类 → 过滤远距/小 box
            let (_boxes, box_indices) = cluster_obstacles_with_indices(
                &non_wall, self.box_cell_size, self.box_min_pts, 0.05, self.box_max_range,
            );
            let mut pts = Vec::new();
            for indices in &box_indices {
                for &idx in indices {
                    pts.push(non_wall[idx]);
                }
            }
            if pts.is_empty() { return (points.to_vec(), Vec::new()); }
            pts
        } else {
            non_wall
        };

        // 3. LV-DOT 体素占用下采样
        let (sampled, _map) = XYGrid::voxel_occupancy_downsample(
            &cluster_input, self.voxel_size, self.min_occ,
        );

        if sampled.is_empty() {
            return (points.to_vec(), Vec::new());
        }

        // 4. XY DBSCAN 精化
        let clusters = xy_dbscan(&sampled, self.dbscan_eps, self.dbscan_min_pts);

        (sampled, clusters)
    }

    fn strategy_name(&self) -> &'static str {
        "lvdot_grid"
    }
}
