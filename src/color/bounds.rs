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
pub struct ClrBud {
    pub the_box: Box2D,
    pub class_id: u32,
    pub class_name: String,
    pub confidence: f32,
}

impl ClrBud {
    /// 创建一个新的检测结果
    /// 
    /// # 参数
    /// * `bbox` - 边界框
    /// * `class_id` - 类别ID
    /// * `class_name` - 类别名称
    /// * `confidence` - 置信度
    pub fn new(the_box: Box2D, class_id: u32, class_name: String, confidence: f32) -> Self {
        Self { the_box, class_id, class_name, confidence }
    }
}

