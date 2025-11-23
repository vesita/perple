use crate::config::DETECTIONS_CAPACITY;

/// 3D边界框结构
/// 
/// 表示一个3D空间中的边界框，用于包围点云中的对象。
#[derive(Debug, Clone, Copy, Default)]
pub struct Box3D {
    pub x_min: f32,
    pub x_max: f32,
    pub y_min: f32,
    pub y_max: f32,
    pub z_min: f32,
    pub z_max: f32,
}

impl Box3D {
    /// 创建一个空的边界框
    pub fn empty_box() -> Self {
        Self {
            x_min: 0.0,
            x_max: 0.0,
            y_min: 0.0,
            y_max: 0.0,
            z_min: 0.0,
            z_max: 0.0,
        }
    }
    
    /// 创建一个新的边界框
    /// 
    /// # 参数
    /// * `x_min` - X轴最小值
    /// * `x_max` - X轴最大值
    /// * `y_min` - Y轴最小值
    /// * `y_max` - Y轴最大值
    /// * `z_min` - Z轴最小值
    /// * `z_max` - Z轴最大值
    pub fn new(x_min: f32, x_max: f32, y_min: f32, y_max: f32, z_min: f32, z_max: f32) -> Self {
        Self {
            x_min,
            x_max,
            y_min,
            y_max,
            z_min,
            z_max,
        }
    }
}

/// 固定容量的3D边界框容器
/// 
/// 这是一个类似于Vec的容器，但具有固定的最大容量，避免了动态分配内存的开销。
/// 它实现了常用的集合操作，如push、clear、len等，并支持迭代器。
#[derive(Clone)]
pub struct LidBud {
    bounds: Vec<Box3D>,
    len: usize,
}

impl LidBud {
    /// 创建一个新的空Bounds容器
    pub fn new() -> Self {
        Self {
            bounds: Vec::with_capacity(DETECTIONS_CAPACITY),
            len: 0,
        }
    }
    
    /// 向容器中添加一个新的检测结果
    pub fn push(&mut self, detection: Box3D) {
        if self.len < DETECTIONS_CAPACITY {
            self.bounds.push(detection);
            self.len += 1;
        }
    }
    
    /// 清空容器
    pub fn clear(&mut self) {
        self.bounds.clear();
        self.len = 0;
    }
    
    /// 获取容器中元素的数量
    pub fn len(&self) -> usize {
        self.len
    }
    
    /// 检查容器是否为空
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    
    /// 获取容器的切片引用
    pub fn as_slice(&self) -> &[Box3D] {
        &self.bounds
    }
    
    /// 获取容器的可变切片引用
    pub fn as_mut_slice(&mut self) -> &mut [Box3D] {
        &mut self.bounds
    }
    
    /// 提供只读引用迭代器
    pub fn iter(&self) -> std::slice::Iter<Box3D> {
        self.bounds.iter()
    }
    
    /// 提供可变引用迭代器
    pub fn iter_mut(&mut self) -> std::slice::IterMut<Box3D> {
        self.bounds.iter_mut()
    }
}

impl Default for LidBud {
    fn default() -> Self {
        Self::new()
    }
}

impl IntoIterator for LidBud {
    type Item = Box3D;
    type IntoIter = std::vec::IntoIter<Box3D>;
    
    fn into_iter(self) -> Self::IntoIter {
        self.bounds.into_iter()
    }
}

impl<'a> IntoIterator for &'a LidBud {
    type Item = &'a Box3D;
    type IntoIter = std::slice::Iter<'a, Box3D>;
    
    fn into_iter(self) -> Self::IntoIter {
        self.bounds.iter()
    }
}

impl<'a> IntoIterator for &'a mut LidBud {
    type Item = &'a mut Box3D;
    type IntoIter = std::slice::IterMut<'a, Box3D>;
    
    fn into_iter(self) -> Self::IntoIter {
        self.bounds.iter_mut()
    }
}

impl Box3D {
    pub fn contains(&self, point: [f32; 3]) -> bool {
        point[0] >= self.x_min && point[0] <= self.x_max &&
        point[1] >= self.y_min && point[1] <= self.y_max &&
        point[2] >= self.z_min && point[2] <= self.z_max
    }

    pub fn near(&self, point: &[f32; 3], distance: f32) -> bool {
        // 使用 clamp 找到点在边界框上的最近点
        let x = point[0].clamp(self.x_min, self.x_max);
        let y = point[1].clamp(self.y_min, self.y_max);
        let z = point[2].clamp(self.z_min, self.z_max);
        
        // 计算欧几里得距离的平方
        let dx = x - point[0];
        let dy = y - point[1];
        let dz = z - point[2];
        
        dx * dx + dy * dy + dz * dz <= distance * distance
    }

    pub fn expand(&mut self, point: &[f32; 3]) {
        self.x_min = self.x_min.min(point[0]);
        self.x_max = self.x_max.max(point[0]);
        self.y_min = self.y_min.min(point[1]);
        self.y_max = self.y_max.max(point[1]);
        self.z_min = self.z_min.min(point[2]);
        self.z_max = self.z_max.max(point[2]);
    }

    pub fn merge(&mut self, other: &Self) {
        self.x_min = self.x_min.min(other.x_min);
        self.x_max = self.x_max.max(other.x_max);
        self.y_min = self.y_min.min(other.y_min);
        self.y_max = self.y_max.max(other.y_max);
        self.z_min = self.z_min.min(other.z_min);
        self.z_max = self.z_max.max(other.z_max);
    }

    pub fn iou(&self, other: &Self) -> f32 {
        // 计算交集区域的边界
        let inter_x_min = self.x_min.max(other.x_min);
        let inter_x_max = self.x_max.min(other.x_max);
        let inter_y_min = self.y_min.max(other.y_min);
        let inter_y_max = self.y_max.min(other.y_max);
        let inter_z_min = self.z_min.max(other.z_min);
        let inter_z_max = self.z_max.min(other.z_max);

        // 检查是否有交集
        if inter_x_min >= inter_x_max || 
            inter_y_min >= inter_y_max || 
            inter_z_min >= inter_z_max {
            return 0.0;
        }

        // 计算交集体积
        let intersection_volume = (inter_x_max - inter_x_min) * 
                                (inter_y_max - inter_y_min) * 
                                (inter_z_max - inter_z_min);

        // 计算两个盒子的体积
        let self_volume = self.volume();
        let other_volume = other.volume();

        // 计算并集体积
        let union_volume = self_volume + other_volume - intersection_volume;

        // 返回交并比
        if union_volume == 0.0 {
            0.0
        } else {
            intersection_volume / union_volume
        }
    }

    pub fn volume(&self) -> f32 {
        (self.x_max - self.x_min) * 
        (self.y_max - self.y_min) * 
        (self.z_max - self.z_min)
    }
}