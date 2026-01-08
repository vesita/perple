use std::vec::Vec;

/// 四叉树节点定义
pub struct QuadTreeNode {
    x_min: f32,
    x_max: f32,
    y_min: f32,
    y_max: f32,
    points: Vec<usize>, // 存储点的索引
    children: Option<[Box<QuadTreeNode>; 4]>,
    is_leaf: bool,
}

impl QuadTreeNode {
    pub fn new(x_min: f32, x_max: f32, y_min: f32, y_max: f32) -> Self {
        Self {
            x_min,
            x_max,
            y_min,
            y_max,
            points: Vec::new(),
            children: None,
            is_leaf: true,
        }
    }

    fn subdivide(&mut self) {
        let x_mid = (self.x_min + self.x_max) / 2.0;
        let y_mid = (self.y_min + self.y_max) / 2.0;

        let mut children = Vec::with_capacity(4);
        // 创建4个子节点
        children.push(Box::new(QuadTreeNode::new(
            self.x_min, x_mid, self.y_min, y_mid,
        )));
        children.push(Box::new(QuadTreeNode::new(
            x_mid, self.x_max, self.y_min, y_mid,
        )));
        children.push(Box::new(QuadTreeNode::new(
            self.x_min, x_mid, y_mid, self.y_max,
        )));
        children.push(Box::new(QuadTreeNode::new(
            x_mid, self.x_max, y_mid, self.y_max,
        )));

        self.children = Some([
            std::mem::replace(
                &mut children[0],
                Box::new(QuadTreeNode::new(0.0, 0.0, 0.0, 0.0)),
            ),
            std::mem::replace(
                &mut children[1],
                Box::new(QuadTreeNode::new(0.0, 0.0, 0.0, 0.0)),
            ),
            std::mem::replace(
                &mut children[2],
                Box::new(QuadTreeNode::new(0.0, 0.0, 0.0, 0.0)),
            ),
            std::mem::replace(
                &mut children[3],
                Box::new(QuadTreeNode::new(0.0, 0.0, 0.0, 0.0)),
            ),
        ]);
        self.is_leaf = false;
    }

    pub fn insert_point(
        &mut self,
        point_idx: usize,
        points: &Vec<[f32; 3]>,
        max_points_per_node: usize,
        max_depth: usize,
        depth: usize,
    ) {
        let point = &points[point_idx];
        // 检查点是否在当前节点范围内
        if point[0] < self.x_min
            || point[0] >= self.x_max
            || point[1] < self.y_min
            || point[1] >= self.y_max
        {
            return;
        }

        // 如果是叶节点且点数未超过限制，直接添加
        if self.is_leaf {
            if self.points.len() < max_points_per_node || depth >= max_depth {
                self.points.push(point_idx);
                return;
            }

            // 否则进行细分
            self.subdivide();

            // 重新分配现有的点
            let mut existing_points = std::mem::take(&mut self.points);
            existing_points.push(point_idx);

            for idx in existing_points {
                let p = &points[idx];
                let px = p[0];
                let py = p[1];

                if let Some(children) = &mut self.children {
                    if px < (self.x_min + self.x_max) / 2.0 && py < (self.y_min + self.y_max) / 2.0
                    {
                        children[0].insert_point(
                            idx,
                            points,
                            max_points_per_node,
                            max_depth,
                            depth + 1,
                        );
                    } else if px >= (self.x_min + self.x_max) / 2.0
                        && py < (self.y_min + self.y_max) / 2.0
                    {
                        children[1].insert_point(
                            idx,
                            points,
                            max_points_per_node,
                            max_depth,
                            depth + 1,
                        );
                    } else if px < (self.x_min + self.x_max) / 2.0
                        && py >= (self.y_min + self.y_max) / 2.0
                    {
                        children[2].insert_point(
                            idx,
                            points,
                            max_points_per_node,
                            max_depth,
                            depth + 1,
                        );
                    } else {
                        children[3].insert_point(
                            idx,
                            points,
                            max_points_per_node,
                            max_depth,
                            depth + 1,
                        );
                    }
                }
            }
        } else {
            // 插入到适当的子节点中
            let px = point[0];
            let py = point[1];

            if let Some(children) = &mut self.children {
                if px < (self.x_min + self.x_max) / 2.0 && py < (self.y_min + self.y_max) / 2.0 {
                    children[0].insert_point(
                        point_idx,
                        points,
                        max_points_per_node,
                        max_depth,
                        depth + 1,
                    );
                } else if px >= (self.x_min + self.x_max) / 2.0
                    && py < (self.y_min + self.y_max) / 2.0
                {
                    children[1].insert_point(
                        point_idx,
                        points,
                        max_points_per_node,
                        max_depth,
                        depth + 1,
                    );
                } else if px < (self.x_min + self.x_max) / 2.0
                    && py >= (self.y_min + self.y_max) / 2.0
                {
                    children[2].insert_point(
                        point_idx,
                        points,
                        max_points_per_node,
                        max_depth,
                        depth + 1,
                    );
                } else {
                    children[3].insert_point(
                        point_idx,
                        points,
                        max_points_per_node,
                        max_depth,
                        depth + 1,
                    );
                }
            }
        }
    }

    // 查询某个点周围的点
    pub fn query_range(
        &self,
        x: f32,
        y: f32,
        radius: f32,
        points: &Vec<[f32; 3]>,
        result: &mut Vec<usize>,
    ) {
        // 检查当前节点是否与查询范围相交
        let node_center_x = (self.x_min + self.x_max) / 2.0;
        let node_center_y = (self.y_min + self.y_max) / 2.0;
        let node_half_width = (self.x_max - self.x_min) / 2.0;
        let node_half_height = (self.y_max - self.y_min) / 2.0;

        let dx = (x - node_center_x).abs();
        let dy = (y - node_center_y).abs();

        if dx > (node_half_width + radius) || dy > (node_half_height + radius) {
            return;
        }

        if self.is_leaf {
            // 检查叶子节点中的所有点
            for &point_idx in &self.points {
                let point = &points[point_idx];
                let distance = ((point[0] - x).powi(2) + (point[1] - y).powi(2)).sqrt();
                if distance <= radius {
                    result.push(point_idx);
                }
            }
        } else {
            // 递归查询子节点
            if let Some(children) = &self.children {
                for child in children.iter() {
                    child.query_range(x, y, radius, points, result);
                }
            }
        }
    }
}
