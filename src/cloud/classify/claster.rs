use crate::{cloud::CldBud, utils::Box3D};

use super::strategy::{ClusteringStrategy, create_strategy};

/// 聚类器 — 不依赖具体策略，通过策略 trait 扩展
pub struct Claster {
    objects: Vec<Vec<usize>>,  // 每个簇的点索引
    all_points: Vec<[f32; 3]>, // 聚类实际使用的点集
    strategy: Box<dyn ClusteringStrategy>,
}

impl Claster {
    pub fn new() -> Self {
        Claster {
            objects: Vec::new(),
            all_points: Vec::new(),
            strategy: create_strategy(),
        }
    }

    /// 清空结果
    pub fn clear(&mut self) {
        self.objects.clear();
        self.all_points.clear();
    }

    /// 聚类结果引用
    pub fn objects(&self) -> &Vec<Vec<usize>> {
        &self.objects
    }

    /// 主入口：对一帧点云执行聚类
    pub fn claster(&mut self, lifra: &[[f32; 3]]) {
        println!("开始聚类，共 {} 个点", lifra.len());
        self.clear();

        let (points, objects) = self.strategy.run(lifra);
        self.all_points = points;
        self.objects = objects;
    }

    /// 从聚类索引同时计算 Box3D 和质心
    fn cluster_box_and_centroid(&self, indices: &[usize]) -> (Box3D, [f32; 3]) {
        let pts: Vec<[f32; 3]> = indices.iter().map(|&idx| self.all_points[idx]).collect();
        let mut box3d = Box3D::empty_box();
        box3d.cloud2box(&pts);
        let centroid = [
            pts.iter().map(|p| p[0]).sum::<f32>() / pts.len() as f32,
            pts.iter().map(|p| p[1]).sum::<f32>() / pts.len() as f32,
            pts.iter().map(|p| p[2]).sum::<f32>() / pts.len() as f32,
        ];
        (box3d, centroid)
    }

    /// 输出为 CldBud 向量，过滤掉明显无效的簇
    pub fn to_cldbuds(&self) -> Vec<CldBud> {
        self.objects
            .iter()
            .filter(|cluster| !cluster.is_empty())
            .enumerate()
            .filter_map(|(idx, cluster)| {
                let (box3d, centroid) = self.cluster_box_and_centroid(cluster);
                let w = box3d.length.max(box3d.width);
                let h = box3d.height;
                if w <= 0.25 || h <= 0.5 {
                    return None;
                }
                Some(CldBud::with_centroid(box3d, 1, format!("cluster_{}", idx), 1.0, centroid))
            })
            .collect()
    }

    #[allow(unused)]
    pub fn add_box3d(&mut self, _box3d: Box3D) {}
}

impl Default for Claster {
    fn default() -> Self {
        Self::new()
    }
}
