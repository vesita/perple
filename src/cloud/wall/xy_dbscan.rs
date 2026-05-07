use std::collections::VecDeque;
use super::WallPickStrategy;
use crate::cloud::classify::quadtree::QuadTreeNode;

/// XY DBSCAN 聚类墙体提取（实验版）。
///
/// 基于四叉树的 XY DBSCAN 聚类，用 Z 跨度区分墙面和物体：
/// - 墙面：Z 跨度大（从地面到天花板）
/// - 物体：Z 跨度小（桌子、椅子等）
pub struct XYDBSCANWall {
    /// DBSCAN 邻域半径 (m)
    eps: f32,
    /// 核心点最小邻居数
    min_pts: usize,
    /// 最小墙面点数
    min_wall_pts: usize,
    /// 最大墙面数
    max_walls: usize,
    /// 墙面最小 Z 跨度 (m)
    min_z_span: f32,
    /// 四叉树叶节点最大点数
    max_points_per_node: usize,
    /// 四叉树最大深度
    max_tree_depth: usize,
}

impl XYDBSCANWall {
    pub fn new() -> Self {
        Self {
            eps: 0.15,
            min_pts: 5,
            min_wall_pts: 30,
            max_walls: 8,
            min_z_span: 1.5,
            max_points_per_node: 50,
            max_tree_depth: 10,
        }
    }

    pub fn with_params(eps: f32, min_pts: usize, min_z_span: f32) -> Self {
        Self { eps, min_pts, min_z_span, ..Self::new() }
    }
}

fn build_quadtree(points: &[[f32; 3]], max_pts: usize, max_depth: usize) -> QuadTreeNode {
    let (mut x_min, mut x_max, mut y_min, mut y_max) = (f32::MAX, f32::MIN, f32::MAX, f32::MIN);
    for p in points {
        if p[0] < x_min { x_min = p[0]; }
        if p[0] > x_max { x_max = p[0]; }
        if p[1] < y_min { y_min = p[1]; }
        if p[1] > y_max { y_max = p[1]; }
    }
    let pad = 0.1;
    let vec_points: Vec<[f32; 3]> = points.to_vec();
    let mut root = QuadTreeNode::new(x_min - pad, x_max + pad, y_min - pad, y_max + pad);
    for i in 0..vec_points.len() {
        root.insert_point(i, &vec_points, max_pts, max_depth, 0);
    }
    root
}

impl WallPickStrategy for XYDBSCANWall {
    fn strategy_name(&self) -> &'static str { "xy_dbscan" }

    fn pick(&mut self, cloud: &mut [[f32; 3]]) -> (usize, Vec<[f32; 4]>) {
        let n = cloud.len();
        if n < self.min_wall_pts { return (0, Vec::new()); }

        // 1. 构建四叉树（query_range 需要 &Vec）
        let vec_cloud: Vec<[f32; 3]> = cloud.to_vec();
        let qt = build_quadtree(cloud, self.max_points_per_node, self.max_tree_depth);

        // 2. DBSCAN — 按需查询邻居
        let mut visited = vec![false; n];
        let mut clusters: Vec<Vec<usize>> = Vec::new();
        let mut nbr_buf = Vec::new();

        for i in 0..n {
            if visited[i] { continue; }
            visited[i] = true;

            nbr_buf.clear();
            qt.query_range(vec_cloud[i][0], vec_cloud[i][1], self.eps, &vec_cloud, &mut nbr_buf);
            if nbr_buf.len() < self.min_pts { continue; } // 非核心点 → 噪声

            // 核心点 → BFS 展开簇
            let mut cluster = vec![i];
            let mut queue: VecDeque<usize> = VecDeque::new();
            for &j in &nbr_buf {
                if !visited[j] {
                    visited[j] = true;
                    queue.push_back(j);
                }
            }

            while let Some(cur) = queue.pop_front() {
                cluster.push(cur);
                nbr_buf.clear();
                qt.query_range(vec_cloud[cur][0], vec_cloud[cur][1], self.eps, &vec_cloud, &mut nbr_buf);
                if nbr_buf.len() >= self.min_pts {
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

        // 3. 每簇计算 Z 跨度，Z 跨度大 → 墙面
        let mut walls: Vec<(Vec<usize>, [f32; 4])> = Vec::new();

        for cluster in &clusters {
            if cluster.len() < self.min_wall_pts { continue; }

            let mut z_min = f32::MAX;
            let mut z_max = f32::MIN;
            let mut cx = 0.0f32;
            let mut cy = 0.0f32;
            for &i in cluster {
                if cloud[i][2] < z_min { z_min = cloud[i][2]; }
                if cloud[i][2] > z_max { z_max = cloud[i][2]; }
                cx += cloud[i][0];
                cy += cloud[i][1];
            }

            if z_max - z_min < self.min_z_span { continue; }

            let nf = cluster.len() as f32;
            cx /= nf;
            cy /= nf;

            // 简易法线：用 XY 2D PCA
            let mut cxx = 0.0f32; let mut cxy = 0.0f32; let mut cyy = 0.0f32;
            for &i in cluster {
                let dx = cloud[i][0] - cx;
                let dy = cloud[i][1] - cy;
                cxx += dx * dx; cxy += dx * dy; cyy += dy * dy;
            }
            cxx /= nf; cxy /= nf; cyy /= nf;

            let trace = cxx + cyy;
            let det = cxx * cyy - cxy * cxy;
            let disc = (trace * trace - 4.0 * det).max(0.0).sqrt();
            let lambda_min = (trace - disc) * 0.5;

            let nx = cxy;
            let ny = lambda_min - cxx;
            let len = (nx * nx + ny * ny).sqrt();
            let (nx, ny) = if len > 1e-8 { (nx / len, ny / len) } else { (1.0, 0.0) };
            let d = -(nx * cx + ny * cy);

            walls.push((cluster.clone(), [nx, ny, 0.0, d]));
        }

        walls.sort_by(|a, b| b.0.len().cmp(&a.0.len()));
        walls.truncate(self.max_walls);

        // 4. 原地重排
        let mut wall_set = std::collections::HashSet::new();
        for (cluster, _) in &walls {
            for &i in cluster { wall_set.insert(i); }
        }
        let mut write = 0usize;
        for read in 0..n {
            if wall_set.contains(&read) {
                cloud.swap(read, write);
                write += 1;
            }
        }

        let planes: Vec<[f32; 4]> = walls.into_iter().map(|(_, p)| p).collect();
        (write, planes)
    }
}
