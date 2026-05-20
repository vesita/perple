use std::sync::Arc;

use super::ClusteringStrategy;
use crate::cloud::classify::quadtree::QuadTreeNode;
use crate::cloud::classify::split_policy::{AdaptiveDepthPolicy, FixedDepthPolicy, SplitPolicy};
use crate::cloud::wall::{WallPickStrategy, BevLsd, cluster_obstacles_with_indices};

/// 剪叶四叉树聚类策略（prune_qt）。
///
/// 与 `lvdot_cluster`（均匀体素过滤）的区别：使用四叉树叶节点过滤替代体素占用下采样，
/// 使用四叉树范围查询替代 XYGrid 网格 DBSCAN。
///
/// 管线：
/// 1. 墙体检测（可选）
/// 2. 网格连通域预聚类（可选）
/// 3. 四叉树构建 → 密集叶节点过滤（叶片点数 ≥ min_occ）→ 质心输出
/// 4. 四叉树加速 DBSCAN 精化聚类
/// 5. 边界点分配（可选）：稀疏叶中的点若在 DBSCAN 半径内，作为边界点加入簇
pub struct PruneQt {
    wall: Box<dyn WallPickStrategy>,
    skip_wall: bool,
    use_box_filter: bool,
    box_cell_size: f32,
    box_min_pts: usize,
    box_max_range: f32,
    /// 叶节点最小点数（≥ min_occ → 保留质心）
    min_occ: usize,
    /// 四叉树叶节点容量（越小叶片越精细）
    max_pts_per_node: usize,
    /// 分裂策略（固定深度 / 自适应分辨率）
    split_policy: Arc<dyn SplitPolicy>,
    /// DBSCAN 半径
    eps: f32,
    /// DBSCAN 核心点阈值
    min_pts: usize,
    /// 启用边界点：稀疏叶中的点若在 eps 内，作为边界点加入最近簇
    use_border_points: bool,
}

impl PruneQt {
    pub fn new() -> Self {
        Self {
            wall: Box::new(BevLsd::with_params(0.05, 20).with_min_extent(0.0)),
            skip_wall: false,
            use_box_filter: false,
            box_cell_size: 0.30,
            box_min_pts: 3,
            box_max_range: 12.0,
            min_occ: 4,
            max_pts_per_node: 20,
            split_policy: Arc::new(FixedDepthPolicy::new(10)),
            eps: 0.20,
            min_pts: 5,
            use_border_points: false,
        }
    }

    pub fn with_border_points(mut self) -> Self {
        self.use_border_points = true;
        self
    }

    pub fn with_pre_extracted_wall(mut self) -> Self {
        self.skip_wall = true;
        self
    }

    pub fn with_box_filter(mut self, cell_size: f32, min_pts: usize, max_range: f32) -> Self {
        self.use_box_filter = true;
        self.box_cell_size = cell_size;
        self.box_min_pts = min_pts;
        self.box_max_range = max_range;
        self
    }

    pub fn with_params(mut self, min_occ: usize, eps: f32, min_pts: usize) -> Self {
        self.min_occ = min_occ;
        self.eps = eps;
        self.min_pts = min_pts;
        self
    }

    /// 使用自适应分辨率策略（距离越远叶子越粗）
    pub fn with_adaptive_depth(mut self, res0: f32, r0: f32, k: f32, global_max_depth: usize) -> Self {
        self.split_policy = Arc::new(AdaptiveDepthPolicy { global_max_depth, res0, r0, k });
        self
    }
}

impl ClusteringStrategy for PruneQt {
    fn run(&mut self, points: &[[f32; 3]]) -> (Vec<[f32; 3]>, Vec<Vec<usize>>) {
        let n = points.len();
        if n == 0 { return (Vec::new(), Vec::new()); }

        // 1. 墙面检测
        let non_wall: Vec<[f32; 3]> = if self.skip_wall {
            points.to_vec()
        } else {
            let mut buf = points.to_vec();
            let (n_wall, _) = self.wall.pick(&mut buf);
            if n_wall >= n { return (points.to_vec(), Vec::new()); }
            buf[n_wall..].to_vec()
        };

        // 2. 预聚类过滤（可选）
        let cluster_input: Vec<[f32; 3]> = if self.use_box_filter && !non_wall.is_empty() {
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

        // 3. 四叉树叶节点过滤 + 分离密集/稀疏叶
        let (x_min, x_max, y_min, y_max) = compute_bounds_xy(&cluster_input);
        let mut qt = QuadTreeNode::new(x_min, x_max, y_min, y_max)
            .with_max_pts_per_node(self.max_pts_per_node)
            .with_policy(Arc::clone(&self.split_policy));
        for i in 0..cluster_input.len() {
            qt.insert_point(i, &cluster_input);
        }
        let leaves = qt.collect_leaves();

        // 分离密集叶（≥ min_occ → 质心）和稀疏叶（< min_occ → 边界点候选）
        let mut dense_idx = Vec::new();     // leaves 中密集叶的索引
        let mut sparse_idx = Vec::new();    // leaves 中稀疏叶的索引
        for (li, leaf) in leaves.iter().enumerate() {
            if leaf.points.len() >= self.min_occ {
                dense_idx.push(li);
            } else if self.use_border_points {
                sparse_idx.push(li);
            }
        }

        let mut sampled = Vec::new();
        for &li in &dense_idx {
            let leaf = &leaves[li];
            let (sx, sy): (f32, f32) = leaf.points.iter()
                .map(|&i| (cluster_input[i][0], cluster_input[i][1]))
                .fold((0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1));
            let nf = leaf.points.len() as f32;
            let zi = cluster_input[leaf.points[0]][2];
            sampled.push([sx / nf, sy / nf, zi]);
        }

        if sampled.is_empty() {
            return (points.to_vec(), Vec::new());
        }

        // 4. 四叉树加速 DBSCAN
        let mut clusters = {
            let (x_min, x_max, y_min, y_max) = compute_bounds_xy(&sampled);
            let mut db_qt = QuadTreeNode::new(x_min, x_max, y_min, y_max)
                .with_max_pts_per_node(self.max_pts_per_node)
                .with_policy(Arc::clone(&self.split_policy));
            for i in 0..sampled.len() {
                db_qt.insert_point(i, &sampled);
            }

            let ns = sampled.len();
            let mut visited = vec![false; ns];
            let mut labels = vec![-1i32; ns];
            let mut cluster_id = 0i32;
            let mut objects = Vec::new();

            for i in 0..ns {
                if visited[i] { continue; }
                visited[i] = true;

                let mut neighbors = Vec::new();
                db_qt.query_range(sampled[i][0], sampled[i][1], self.eps, &sampled, &mut neighbors);
                if neighbors.len() < self.min_pts { continue; }

                labels[i] = cluster_id;
                let mut cluster = vec![i];
                let mut nvec: Vec<usize> = neighbors;
                let mut k = 0;
                while k < nvec.len() {
                    let ni = nvec[k];
                    if !visited[ni] {
                        visited[ni] = true;
                        let mut more = Vec::new();
                        db_qt.query_range(sampled[ni][0], sampled[ni][1], self.eps, &sampled, &mut more);
                        if more.len() >= self.min_pts {
                            nvec.extend(more);
                        }
                    }
                    if labels[ni] == -1 {
                        labels[ni] = cluster_id;
                        cluster.push(ni);
                    }
                    k += 1;
                }
                objects.push(cluster);
                cluster_id += 1;
            }
            objects
        };

        // 5. 边界点分配：稀疏叶中的点若在 eps 内，加入最近簇
        if self.use_border_points && !sparse_idx.is_empty() {
            let eps2 = self.eps * self.eps;
            for &li in &sparse_idx {
                let leaf = &leaves[li];
                for &pt_idx in &leaf.points {
                    let p = cluster_input[pt_idx];
                    let mut assigned = None;
                    for (ci, cluster) in clusters.iter().enumerate() {
                        for &si in cluster {
                            let d2 = (p[0] - sampled[si][0]).powi(2)
                                   + (p[1] - sampled[si][1]).powi(2);
                            if d2 <= eps2 {
                                assigned = Some(ci);
                                break;
                            }
                        }
                        if assigned.is_some() { break; }
                    }
                    if let Some(ci) = assigned {
                        let new_idx = sampled.len();
                        sampled.push(p);
                        clusters[ci].push(new_idx);
                    }
                }
            }
        }

        (sampled, clusters)
    }

    fn strategy_name(&self) -> &'static str {
        "prune_qt"
    }
}

fn compute_bounds_xy(cloud: &[[f32; 3]]) -> (f32, f32, f32, f32) {
    let mut x_min = f32::MAX;
    let mut x_max = f32::MIN;
    let mut y_min = f32::MAX;
    let mut y_max = f32::MIN;
    for p in cloud {
        if p[0] < x_min { x_min = p[0]; }
        if p[0] > x_max { x_max = p[0]; }
        if p[1] < y_min { y_min = p[1]; }
        if p[1] > y_max { y_max = p[1]; }
    }
    let pad = 0.1;
    (x_min - pad, x_max + pad, y_min - pad, y_max + pad)
}
