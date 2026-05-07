mod seq_fit;
mod top_down;
mod xy_ransac;
mod xy_dbscan;

pub use seq_fit::SequentialFit;
pub use top_down::TopDownCluster;
pub use xy_ransac::XYRansacWall;
pub use xy_dbscan::XYDBSCANWall;

/// 墙体提取策略 trait
///
/// 输入非地面点云，原地分区：`cloud[..n_wall]` 为墙面点，`cloud[n_wall..]` 为非墙面点。
/// 返回 `(墙面点数, 各墙面平面方程 Vec<[nx, ny, nz, d]>)`。
pub trait WallPickStrategy: Send {
    fn pick(&mut self, cloud: &mut [[f32; 3]]) -> (usize, Vec<[f32; 4]>);
    fn strategy_name(&self) -> &'static str { "unknown" }
}
