use crate::config::DETECTIONS_CAPACITY;

/// 边界框结构
/// 
/// 表示一个矩形边界框，用于包围检测到的目标。
#[derive(Debug, Clone, Default, Copy, PartialEq)]
pub struct Box2D {
    /// 左上角x坐标
    pub x1: f32,
    /// 左上角y坐标
    pub y1: f32,
    /// 右下角x坐标
    pub x2: f32,
    /// 右下角y坐标
    pub y2: f32,
}

impl Box2D {
    /// 创建一个新的边界框
    pub fn new(x1: f32, y1: f32, x2: f32, y2: f32) -> Self {
        Self { x1, y1, x2, y2 }
    }
    
    
    /// 计算边界框的宽度
    pub fn width(&self) -> f32 {
        (self.x2 - self.x1).abs()
    }
    
    /// 计算边界框的高度
    pub fn height(&self) -> f32 {
        (self.y2 - self.y1).abs()
    }
    
    /// 计算边界框的面积
    pub fn area(&self) -> f32 {
        self.width() * self.height()
    }
    
    /// 检查边界框是否有效（宽度和高度都大于0）
    pub fn is_valid(&self) -> bool {
        self.width() > 0.0 && self.height() > 0.0
    }
}


/// 目标检测结果
/// 
/// 表示一个检测到的对象，包括边界框、类别ID、类别名称和置信度。
#[derive(Debug, Clone, Default)]
pub struct Detection {
    pub bbox: Box2D,
    pub class_id: u32,
    pub class_name: String,
    pub confidence: f32,
}

impl Detection {
    /// 创建一个新的检测结果
    /// 
    /// # 参数
    /// * `bbox` - 边界框
    /// * `class_id` - 类别ID
    /// * `class_name` - 类别名称
    /// * `confidence` - 置信度
    pub fn new(bbox: Box2D, class_id: u32, class_name: String, confidence: f32) -> Self {
        Self { bbox, class_id, class_name, confidence }
    }
}

/// 固定容量的检测结果容器
/// 
/// 这是一个带有预分配容量的容器，用于存储检测到的对象。
/// 它实现了常用的集合操作，如push、clear、len等，并支持迭代器。
#[derive(Clone)]
pub struct ImgBud {
    bounds: Vec<Detection>,
}

impl ImgBud {
    /// 创建一个新的空检测结果容器
    /// 
    /// 容量被预设为配置文件中定义的 `DETECTIONS_CAPACITY`。
    pub fn new() -> Self {
        Self {
            bounds: Vec::with_capacity(DETECTIONS_CAPACITY),
        }
    }
    
    /// 向容器中添加一个新的检测结果
    /// 
    /// 如果容器已满 (达到 `DETECTIONS_CAPACITY`)，则不会添加新元素。
    pub fn push(&mut self, detection: Detection) {
        if self.bounds.len() < DETECTIONS_CAPACITY {
            self.bounds.push(detection);
        }
    }
    
    /// 清空容器中的所有检测结果
    pub fn clear(&mut self) {
        self.bounds.clear();
    }
    
    /// 获取容器中检测结果的数量
    pub fn len(&self) -> usize {
        self.bounds.len()
    }
    
    /// 检查容器是否为空
    pub fn is_empty(&self) -> bool {
        self.bounds.is_empty()
    }
    
    /// 获取容器的切片引用
    pub fn as_slice(&self) -> &[Detection] {
        &self.bounds
    }
    
    /// 获取容器的可变切片引用
    pub fn as_mut_slice(&mut self) -> &mut [Detection] {
        &mut self.bounds
    }
    
    /// 提供只读引用迭代器
    pub fn iter(&self) -> std::slice::Iter<Detection> {
        self.bounds.iter()
    }
    
    /// 提供可变引用迭代器
    pub fn iter_mut(&mut self) -> std::slice::IterMut<Detection> {
        self.bounds.iter_mut()
    }
}

impl Default for ImgBud {
    fn default() -> Self {
        Self::new()
    }
}

impl IntoIterator for ImgBud {
    type Item = Detection;
    type IntoIter = std::vec::IntoIter<Detection>;
    
    fn into_iter(self) -> Self::IntoIter {
        self.bounds.into_iter()
    }
}

impl<'a> IntoIterator for &'a ImgBud {
    type Item = &'a Detection;
    type IntoIter = std::slice::Iter<'a, Detection>;
    
    fn into_iter(self) -> Self::IntoIter {
        self.bounds.iter()
    }
}

impl<'a> IntoIterator for &'a mut ImgBud {
    type Item = &'a mut Detection;
    type IntoIter = std::slice::IterMut<'a, Detection>;
    
    fn into_iter(self) -> Self::IntoIter {
        self.bounds.iter_mut()
    }
}
