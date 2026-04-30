use crate::utils::boxes::Box3D;

/// 3D目标检测结果
///
/// 表示一个检测到的3D对象，包括边界框、类别ID、类别名称和置信度。
#[derive(Debug, Clone)]
pub struct CldBud {
    pub the_box: Box3D,
    pub class_id: u32,
    pub class_name: String,
    pub confidence: f32,
    /// 点云质心（AABB几何中心的替代，用于KF测量更稳定）
    pub centroid: [f32; 3],
}

impl CldBud {
    /// 创建一个新的检测结果
    pub fn new(the_box: Box3D, class_id: u32, class_name: String, confidence: f32) -> Self {
        Self {
            centroid: the_box.center_single(),
            the_box,
            class_id,
            class_name,
            confidence,
        }
    }

    /// 创建带质心的检测结果
    pub fn with_centroid(the_box: Box3D, class_id: u32, class_name: String, confidence: f32, centroid: [f32; 3]) -> Self {
        Self { the_box, class_id, class_name, confidence, centroid }
    }
}

impl Default for CldBud {
    fn default() -> Self {
        Self {
            the_box: Box3D::empty_box(),
            class_id: 0,
            class_name: String::new(),
            confidence: 0.0,
            centroid: [0.0; 3],
        }
    }
}
