use crate::config::DETECTIONS_CAPACITY;

/// 边界框结构
/// 
/// 表示一个矩形边界框，用于包围检测到的目标。
#[derive(Debug, Clone, Default, Copy, PartialEq)]
pub struct BoundingBox {
    /// 左上角x坐标
    pub x1: f32,
    /// 左上角y坐标
    pub y1: f32,
    /// 右下角x坐标
    pub x2: f32,
    /// 右下角y坐标
    pub y2: f32,
}

impl BoundingBox {
    /// 创建一个新的边界框
    pub fn new(x1: f32, y1: f32, x2: f32, y2: f32) -> Self {
        Self { x1, y1, x2, y2 }
    }
    
    /// 创建一个默认的边界框
    pub fn default() -> Self {
        Self { x1: 0.0, y1: 0.0, x2: 0.0, y2: 0.0 }
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


/// 检测结果结构
/// 
/// 包含检测到的目标的完整信息。
#[derive(Debug, Clone, Default)]
pub struct Detection {
    /// 目标的边界框
    pub bbox: BoundingBox,
    /// 类别ID
    pub class_id: usize,
    /// 类别名称
    pub class_name: String,
    /// 置信度
    pub confidence: f32,
}

impl Detection {
    /// 创建一个新的检测结果
    pub fn new(bbox: BoundingBox, class_id: usize, class_name: String, confidence: f32) -> Self {
        Self { bbox, class_id, class_name, confidence }
    }
    
    /// 创建一个默认的检测结果
    pub fn default() -> Self {
        Self { 
            bbox: BoundingBox::default(), 
            class_id: 0, 
            class_name: String::new(), 
            confidence: 0.0 
        }
    }
}

/// 固定容量的检测结果容器
/// 
/// 这是一个类似于Vec的容器，但具有固定的最大容量，避免了动态分配内存的开销。
/// 它实现了常用的集合操作，如push、clear、len等，并支持迭代器。
pub struct Bounds {
    bounds: [Detection; DETECTIONS_CAPACITY],
    len: usize,
}

impl Bounds {
    /// 创建一个新的空Bounds容器
    pub fn new() -> Self {
        Self {
            bounds: std::array::from_fn(|_| Detection::default()),
            len: 0,
        }
    }
    
    /// 向容器中添加一个新的检测结果
    /// 
    /// 如果容器已满，则不会添加新元素
    pub fn push(&mut self, detection: Detection) {
        if self.len < DETECTIONS_CAPACITY {
            self.bounds[self.len] = detection;
            self.len += 1;
        }
    }
    
    /// 清空容器中的所有检测结果
    pub fn clear(&mut self) {
        self.len = 0;
    }
    
    /// 返回容器中检测结果的数量
    pub fn len(&self) -> usize {
        self.len
    }
    
    /// 检查容器是否为空
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    
    /// 获取容器中所有检测结果的切片引用
    pub fn as_slice(&self) -> &[Detection] {
        &self.bounds[..self.len]
    }
    
    /// 获取容器中所有检测结果的可变切片引用
    pub fn as_mut_slice(&mut self) -> &mut [Detection] {
        &mut self.bounds[..self.len]
    }
    
    /// 根据索引获取检测结果的引用
    pub fn get(&self, index: usize) -> Option<&Detection> {
        if index < self.len {
            Some(&self.bounds[index])
        } else {
            None
        }
    }
    
    /// 根据索引获取检测结果的可变引用
    pub fn get_mut(&mut self, index: usize) -> Option<&mut Detection> {
        if index < self.len {
            Some(&mut self.bounds[index])
        } else {
            None
        }
    }
    
    /// 获取第一个检测结果的引用
    pub fn first(&self) -> Option<&Detection> {
        if self.len > 0 {
            Some(&self.bounds[0])
        } else {
            None
        }
    }
    
    /// 获取最后一个检测结果的引用
    pub fn last(&self) -> Option<&Detection> {
        if self.len > 0 {
            Some(&self.bounds[self.len - 1])
        } else {
            None
        }
    }
    
    /// 对检测结果按置信度进行排序（降序）
    pub fn sort_by_confidence(&mut self) {
        let slice = self.as_mut_slice();
        slice.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
    }
    
    /// 对检测结果按指定比较函数进行排序
    pub fn sort_by<F>(&mut self, compare: F) 
    where 
        F: FnMut(&Detection, &Detection) -> std::cmp::Ordering,
    {
        let slice = self.as_mut_slice();
        slice.sort_by(compare);
    }
    
    /// 提供只读迭代器
    pub fn iter(&self) -> std::slice::Iter<'_, Detection> {
        self.as_slice().iter()
    }
    
    /// 提供可变迭代器
    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, Detection> {
        self.as_mut_slice().iter_mut()
    }
    
    /// 保留满足条件的检测结果
    pub fn retain<F>(&mut self, mut f: F) 
    where 
        F: FnMut(&Detection) -> bool,
    {
        let mut i = 0;
        while i < self.len {
            if !f(&self.bounds[i]) {
                // 移动后续元素
                for j in i..(self.len - 1) {
                    self.bounds[j] = self.bounds[j + 1].clone();
                }
                self.len -= 1;
            } else {
                i += 1;
            }
        }
    }
}

// 实现只读迭代器支持
impl<'a> IntoIterator for &'a Bounds {
    type Item = &'a Detection;
    type IntoIter = std::slice::Iter<'a, Detection>;
    
    fn into_iter(self) -> Self::IntoIter {
        self.as_slice().iter()
    }
}

// 实现可变迭代器支持
impl<'a> IntoIterator for &'a mut Bounds {
    type Item = &'a mut Detection;
    type IntoIter = std::slice::IterMut<'a, Detection>;
    
    fn into_iter(self) -> Self::IntoIter {
        self.as_mut_slice().iter_mut()
    }
}

// 实现默认trait
impl Default for Bounds {
    fn default() -> Self {
        Self::new()
    }
}

// 实现Debug trait
impl std::fmt::Debug for Bounds {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Bounds")
            .field("len", &self.len)
            .field("bounds", &self.as_slice())
            .finish()
    }
}