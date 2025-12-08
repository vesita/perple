use crate::{cloud::{CldBud}, config::fixif, utils::Box3D};
use std::collections::{HashMap, HashSet};
use super::quadtree::QuadTreeNode;


pub struct Claster {
    objects: Vec<Vec<usize>>, // 存储属于每个簇的点的索引
    all_points: Vec<[f32; 3]>, // 存储所有的点
    // 缓存配置参数
    patience: f32,
    merge_threshold: f32,
    min_points_per_cluster: usize,
    max_points_per_node: usize,
    max_tree_depth: usize,
    // 下采样相关参数
    voxel_size: f32,
    // 四叉树根节点
    quad_tree: Option<QuadTreeNode>,
    // 空间范围
    x_min: f32,
    x_max: f32,
    y_min: f32,
    y_max: f32,
}

impl Claster { 
    pub fn new() -> Self {
        let config = fixif();
        Claster {
            objects: Vec::new(),
            all_points: Vec::new(),
            patience: config.claster.merge_patience,
            merge_threshold: config.claster.merge_threshold,
            min_points_per_cluster: config.claster.min_points_per_cluster.unwrap_or(3),
            max_points_per_node: config.claster.max_points_per_node.unwrap_or(50),
            max_tree_depth: config.claster.max_tree_depth.unwrap_or(10),
            voxel_size: config.claster.voxel_size,
            quad_tree: None,
            x_min: -100.0,
            x_max: 100.0,
            y_min: -100.0,
            y_max: 100.0,
        }
    }
    
    /// 清空聚类结果，重置状态
    pub fn clear(&mut self) {
        self.objects.clear();
        self.all_points.clear();
        self.quad_tree = None;
    }

    /// 获取聚类对象的不可变引用
    pub fn objects(&self) -> &Vec<Vec<usize>> {
        &self.objects
    }

    /// 计算点所属的体素键值
    fn get_voxel_key(&self, point: &[f32; 3]) -> [i32; 3] {
        [
            (point[0] / self.voxel_size).floor() as i32,
            (point[1] / self.voxel_size).floor() as i32,
            (point[2] / self.voxel_size).floor() as i32,
        ]
    }

    /// 构建四叉树
    fn build_quad_tree(&mut self) {
        let mut root = QuadTreeNode::new(self.x_min, self.x_max, self.y_min, self.y_max);
        
        // 插入所有点
        for i in 0..self.all_points.len() {
            root.insert_point(i, &self.all_points, self.max_points_per_node, self.max_tree_depth, 0);
        }
        
        self.quad_tree = Some(root);
    }

    /// DBSCAN风格的聚类算法
    fn dbscan_cluster(&mut self) {
        if self.all_points.is_empty() {
            return;
        }

        // 构建四叉树以加速邻域查询
        self.build_quad_tree();

        let n_points = self.all_points.len();
        let mut visited = vec![false; n_points];
        let mut cluster_labels = vec![-1i32; n_points]; // -1 表示噪声点
        let mut cluster_id = 0;

        for i in 0..n_points {
            if visited[i] {
                continue;
            }

            visited[i] = true;

            // 使用四叉树查询邻域内的点
            let mut neighbors = Vec::new();
            if let Some(quad_tree) = &self.quad_tree {
                quad_tree.query_range(
                    self.all_points[i][0],
                    self.all_points[i][1],
                    self.patience,
                    &self.all_points,
                    &mut neighbors,
                );
            }

            // 如果邻域内点数不足，则标记为噪声
            if neighbors.len() < self.min_points_per_cluster {
                continue;
            }

            // 创建新簇
            cluster_labels[i] = cluster_id;
            let mut cluster = vec![i];
            
            // 使用HashSet来提高contains操作的性能
            let mut neighbor_set: HashSet<usize> = neighbors.into_iter().collect();
            let mut neighbors_vec: Vec<usize> = neighbor_set.iter().cloned().collect();

            // 扩展簇
            let mut k = 0;
            while k < neighbors_vec.len() {
                let neighbor_idx = neighbors_vec[k];
                if !visited[neighbor_idx] {
                    visited[neighbor_idx] = true;
                    
                    // 查询该邻居的邻域
                    let mut more_neighbors = Vec::new();
                    if let Some(quad_tree) = &self.quad_tree {
                        quad_tree.query_range(
                            self.all_points[neighbor_idx][0],
                            self.all_points[neighbor_idx][1],
                            self.patience,
                            &self.all_points,
                            &mut more_neighbors,
                        );
                    }

                    // 如果邻居也是核心点，将其邻域加入
                    if more_neighbors.len() >= self.min_points_per_cluster {
                        for n_idx in more_neighbors {
                            if neighbor_set.insert(n_idx) {
                                neighbors_vec.push(n_idx);
                            }
                        }
                    }
                }

                if cluster_labels[neighbor_idx] == -1 {
                    cluster_labels[neighbor_idx] = cluster_id;
                    cluster.push(neighbor_idx);
                }

                k += 1;
            }

            self.objects.push(cluster);
            cluster_id += 1;
        }
    }

    /// 将索引簇转换为Box3D边界框
    pub fn clusters_to_boxes(&self) -> Vec<Box3D> {
        let mut boxes = Vec::new();
        
        for cluster in &self.objects {
            if cluster.is_empty() {
                continue;
            }
            
            let mut box3d = Box3D::empty_box();
            // 避免不必要的内存分配，直接传递引用
            let cluster_points: Vec<[f32; 3]> = cluster.iter()
                .map(|&idx| self.all_points[idx])
                .collect();
            
            box3d.cloud2box(&cluster_points);
            boxes.push(box3d);
        }
        
        boxes
    }

    /// 直接处理整个Vec<[f32; 3]>帧数据
    pub fn claster(&mut self, lifra: &Vec<[f32; 3]>) {
        println!("开始聚类，共{}个点", lifra.len());
        
        // 清空之前的聚类结果
        self.clear();
        
        // 直接借用数据而不是克隆
        self.all_points.clear();
        self.all_points.reserve(lifra.len());
        self.all_points.extend_from_slice(lifra);
        
        // 使用体素下采样预处理点云
        let mut seen_voxels = HashMap::new();
        let mut downsampled_indices = Vec::new();
        
        for (i, point) in self.all_points.iter().enumerate() {
            let voxel_key = self.get_voxel_key(point);
            if !seen_voxels.contains_key(&voxel_key) {
                seen_voxels.insert(voxel_key, true);
                downsampled_indices.push(i);
            }
        }
        
        println!("下采样后剩余 {} 个点", downsampled_indices.len());
        
        // 创建下采样后的点列表用于聚类
        let downsampled_points: Vec<[f32; 3]> = downsampled_indices
            .iter()
            .map(|&i| self.all_points[i])
            .collect();
        
        // 替换点集为下采样后的点集
        self.all_points = downsampled_points;
        
        // 执行DBSCAN风格的聚类
        self.dbscan_cluster();
        
        println!("聚类完成，共发现{}个聚类", self.objects.len());
    }
    
    // 将聚类结果转换为CldBud向量
    pub fn to_cldbuds(&self) -> Vec<CldBud> {
        let boxes = self.clusters_to_boxes();
        
        boxes.iter().enumerate().map(|(idx, box3d)| {
            CldBud::new(
                box3d.clone(), // 使用clone而不是解引用
                1, // 默认类别ID为1（非地面）
                format!("cluster_{}", idx),
                1.0, // 置信度
            )
        }).collect()
    }
    
    // 添加Box3D对象到聚类中
    pub fn add_box3d(&mut self, _box3d: Box3D) {
        // 此方法在新的实现中不再适用
    }
}