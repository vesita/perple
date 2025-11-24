
use crate::utils::boxes::Box2D;

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