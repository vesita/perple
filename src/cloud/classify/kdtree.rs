use std::cmp::Ordering;

/// 一个简单的KD-Tree实现，用于3D点的最近邻搜索
pub struct KdNode {
    point: [f32; 3],
    index: usize,
    left: Option<Box<KdNode>>,
    right: Option<Box<KdNode>>,
}

pub struct KdTree {
    root: Option<Box<KdNode>>,
}

impl KdNode {
    /// 创建新的节点
    fn new(point: [f32; 3], index: usize) -> Self {
        KdNode {
            point,
            index,
            left: None,
            right: None,
        }
    }
}

impl KdTree {
    /// 创建新的空KD-Tree
    pub fn new() -> Self {
        KdTree { root: None }
    }

    /// 插入一个点到树中
    pub fn insert(&mut self, point: [f32; 3], index: usize) {
        self.root = Self::insert_recursive(self.root.take(), point, index, 0);
    }

    /// 递归插入函数
    fn insert_recursive(
        node: Option<Box<KdNode>>,
        point: [f32; 3],
        index: usize,
        depth: usize,
    ) -> Option<Box<KdNode>> {
        match node {
            None => Some(Box::new(KdNode::new(point, index))),
            Some(mut node) => {
                // 根据深度决定比较哪个维度
                let dim = depth % 3;
                match point[dim].partial_cmp(&node.point[dim]).unwrap() {
                    Ordering::Less => {
                        node.left = Self::insert_recursive(node.left.take(), point, index, depth + 1);
                    }
                    _ => {
                        node.right = Self::insert_recursive(node.right.take(), point, index, depth + 1);
                    }
                }
                Some(node)
            }
        }
    }

    /// 查找最近邻
    pub fn nearest_neighbor(&self, target: &[f32; 3]) -> Option<(usize, f32)> {
        match &self.root {
            None => None,
            Some(_) => {
                let mut best: Option<(usize, f32)> = None;
                Self::nearest_neighbor_recursive(&self.root, target, 0, &mut best);
                best
            }
        }
    }

    /// 递归查找最近邻
    fn nearest_neighbor_recursive(
        node: &Option<Box<KdNode>>,
        target: &[f32; 3],
        depth: usize,
        best: &mut Option<(usize, f32)>,
    ) {
        if let Some(node) = node {
            // 计算当前点到目标点的距离
            let distance = squared_distance(&node.point, target);
            
            // 更新最佳匹配
            match best {
                None => *best = Some((node.index, distance)),
                Some((_, best_dist)) => {
                    if distance < *best_dist {
                        *best = Some((node.index, distance));
                    }
                }
            }

            // 根据深度决定比较哪个维度
            let dim = depth % 3;
            let (next_branch, opposite_branch) = if target[dim] < node.point[dim] {
                (&node.left, &node.right)
            } else {
                (&node.right, &node.left)
            };

            // 递归搜索靠近的那一侧
            Self::nearest_neighbor_recursive(next_branch, target, depth + 1, best);

            // 检查是否需要搜索另一侧
            let axis_distance = (target[dim] - node.point[dim]).powi(2);
            if let Some((_, best_dist)) = best {
                if axis_distance < *best_dist {
                    Self::nearest_neighbor_recursive(opposite_branch, target, depth + 1, best);
                }
            }
        }
    }
}

/// 计算两点之间的平方距离
fn squared_distance(a: &[f32; 3], b: &[f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    dx * dx + dy * dy + dz * dz
}