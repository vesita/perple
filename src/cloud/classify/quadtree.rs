use std::sync::Arc;
use std::vec::Vec;

use super::split_policy::{FixedDepthPolicy, SplitPolicy};

/// 四叉树叶节点信息（collect_leaves 产出）
pub struct QuadLeaf {
    pub x_min: f32,
    pub x_max: f32,
    pub y_min: f32,
    pub y_max: f32,
    pub points: Vec<usize>,
}

/// 四叉树节点定义
pub struct QuadTreeNode {
    x_min: f32,
    x_max: f32,
    y_min: f32,
    y_max: f32,
    points: Vec<usize>,
    children: Option<[Box<QuadTreeNode>; 4]>,
    is_leaf: bool,
    /// 每个节点最大点数（超过则细分）
    max_pts_per_node: usize,
    /// 分裂策略（控制最大深度 / 自适应分辨率）
    policy: Arc<dyn SplitPolicy>,
}

impl QuadTreeNode {
    /// 创建根节点，使用默认参数（max_pts_per_node=50, FixedDepthPolicy=10）。
    pub fn new(x_min: f32, x_max: f32, y_min: f32, y_max: f32) -> Self {
        Self {
            x_min,
            x_max,
            y_min,
            y_max,
            points: Vec::new(),
            children: None,
            is_leaf: true,
            max_pts_per_node: 50,
            policy: Arc::new(FixedDepthPolicy::default()),
        }
    }

    pub fn with_max_pts_per_node(mut self, n: usize) -> Self {
        self.max_pts_per_node = n;
        self
    }

    pub fn with_policy(mut self, policy: Arc<dyn SplitPolicy>) -> Self {
        self.policy = policy;
        self
    }

    fn subdivide(&mut self) {
        let x_mid = (self.x_min + self.x_max) / 2.0;
        let y_mid = (self.y_min + self.y_max) / 2.0;

        let ch = [
            Box::new(Self::new(self.x_min, x_mid, self.y_min, y_mid)
                .with_max_pts_per_node(self.max_pts_per_node)
                .with_policy(Arc::clone(&self.policy))),
            Box::new(Self::new(x_mid, self.x_max, self.y_min, y_mid)
                .with_max_pts_per_node(self.max_pts_per_node)
                .with_policy(Arc::clone(&self.policy))),
            Box::new(Self::new(self.x_min, x_mid, y_mid, self.y_max)
                .with_max_pts_per_node(self.max_pts_per_node)
                .with_policy(Arc::clone(&self.policy))),
            Box::new(Self::new(x_mid, self.x_max, y_mid, self.y_max)
                .with_max_pts_per_node(self.max_pts_per_node)
                .with_policy(Arc::clone(&self.policy))),
        ];
        self.children = Some(ch);
        self.is_leaf = false;
    }

    /// 插入点索引。max_pts_per_node 和 max_depth 使用构造时的设定。
    pub fn insert_point(&mut self, point_idx: usize, points: &[[f32; 3]]) {
        let point = &points[point_idx];
        if point[0] < self.x_min
            || point[0] >= self.x_max
            || point[1] < self.y_min
            || point[1] >= self.y_max
        {
            return;
        }

        self.insert_point_rec(point_idx, points, 0);
    }

    fn insert_point_rec(&mut self, point_idx: usize, points: &[[f32; 3]], depth: usize) {
        if self.is_leaf {
            let cx = (self.x_min + self.x_max) / 2.0;
            let cy = (self.y_min + self.y_max) / 2.0;
            let should_split = self.policy.should_split(depth, cx, cy, self.diagonal());
            if self.points.len() < self.max_pts_per_node || !should_split {
                self.points.push(point_idx);
                return;
            }
            self.subdivide();
            let mut existing = std::mem::take(&mut self.points);
            existing.push(point_idx);
            for idx in existing {
                self.insert_into_child(idx, points, depth + 1);
            }
        } else {
            self.insert_into_child(point_idx, points, depth + 1);
        }
    }

    fn insert_into_child(&mut self, idx: usize, points: &[[f32; 3]], depth: usize) {
        let p = &points[idx];
        if let Some(children) = &mut self.children {
            let x_mid = (self.x_min + self.x_max) / 2.0;
            let y_mid = (self.y_min + self.y_max) / 2.0;
            let ci = match (p[0] < x_mid, p[1] < y_mid) {
                (true, true) => 0,
                (false, true) => 1,
                (true, false) => 2,
                (false, false) => 3,
            };
            children[ci].insert_point_rec(idx, points, depth);
        }
    }

    /// 收集所有叶节点的边界与点索引。
    pub fn collect_leaves(&self) -> Vec<QuadLeaf> {
        let mut leaves = Vec::new();
        self.collect_leaves_rec(&mut leaves);
        leaves
    }

    fn collect_leaves_rec(&self, out: &mut Vec<QuadLeaf>) {
        if self.is_leaf {
            out.push(QuadLeaf {
                x_min: self.x_min,
                x_max: self.x_max,
                y_min: self.y_min,
                y_max: self.y_max,
                points: self.points.clone(),
            });
        } else if let Some(children) = &self.children {
            for child in children.iter() {
                child.collect_leaves_rec(out);
            }
        }
    }

    /// 计算节点对角线长度。
    pub fn diagonal(&self) -> f32 {
        let dx = self.x_max - self.x_min;
        let dy = self.y_max - self.y_min;
        (dx * dx + dy * dy).sqrt()
    }

    /// 外部强制分裂叶节点（不考虑 max_pts_per_node）。
    /// 子节点继承父节点 1/4 的 max_pts_per_node。
    /// 只对叶节点有效；已分裂节点或无点节点无操作。
    pub fn force_split(&mut self, points: &[[f32; 3]], depth: usize) {
        if !self.is_leaf || depth >= self.policy.global_max_depth() || self.points.is_empty() {
            return;
        }
        self.subdivide();
        let child_max = (self.max_pts_per_node / 4).max(1);
        if let Some(children) = &mut self.children {
            for child in children.iter_mut() {
                child.max_pts_per_node = child_max;
            }
        }
        let existing = std::mem::take(&mut self.points);
        for idx in existing {
            self.insert_into_child(idx, points, depth + 1);
        }
    }

    /// 递归分裂对角线超过 max_diag 的叶节点，
    /// 直至所有叶节点满足条件或达到 max_depth。
    pub fn split_large_leaves(&mut self, max_diag: f32, points: &[[f32; 3]], depth: usize) {
        if self.is_leaf {
            if self.diagonal() > max_diag && depth < self.policy.global_max_depth() && !self.points.is_empty() {
                self.force_split(points, depth);
                for child in self.children.as_mut().unwrap().iter_mut() {
                    child.split_large_leaves(max_diag, points, depth + 1);
                }
            }
        } else if let Some(children) = &mut self.children {
            for child in children.iter_mut() {
                child.split_large_leaves(max_diag, points, depth + 1);
            }
        }
    }

    /// 查询某个点周围的点
    pub fn query_range(
        &self,
        x: f32,
        y: f32,
        radius: f32,
        points: &[[f32; 3]],
        result: &mut Vec<usize>,
    ) {
        let node_half_width = (self.x_max - self.x_min) / 2.0;
        let node_half_height = (self.y_max - self.y_min) / 2.0;
        let dx = (x - (self.x_min + self.x_max) / 2.0).abs();
        let dy = (y - (self.y_min + self.y_max) / 2.0).abs();

        if dx > (node_half_width + radius) || dy > (node_half_height + radius) {
            return;
        }

        if self.is_leaf {
            for &point_idx in &self.points {
                let point = &points[point_idx];
                let distance = (point[0] - x).powi(2) + (point[1] - y).powi(2);
                if distance <= radius * radius {
                    result.push(point_idx);
                }
            }
        } else if let Some(children) = &self.children {
            for child in children.iter() {
                child.query_range(x, y, radius, points, result);
            }
        }
    }
}
