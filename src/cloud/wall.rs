pub mod l2_util;
mod xy_grid;
mod cluster_utils;

mod common;
mod bev_hough;
mod bev_lsd;
mod bev_edlines;
mod edlines_ref;

pub use bev_lsd::BevLsd;
pub use bev_edlines::BevEdLines;
pub use bev_hough::BevHough;
pub use edlines_ref::EdLinesRef;

pub(crate) use l2_util::best_xy_line;
pub use xy_grid::XYGrid;
pub use cluster_utils::{cluster_obstacles, cluster_obstacles_with_indices, xy_dbscan};

/// 墙体检测策略 trait
///
/// 输入非地面点云，原地分区：`cloud[..n_wall]` 为墙面点，`cloud[n_wall..]` 为非墙面点。
/// 返回 `(墙面点数, 各墙面平面方程 Vec<[nx, ny, nz, d]>)`。
pub trait WallPickStrategy: Send {
    fn pick(&mut self, cloud: &mut [[f32; 3]]) -> (usize, Vec<[f32; 4]>);
    fn strategy_name(&self) -> &'static str { "unknown" }
}
