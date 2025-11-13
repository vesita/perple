pub const STREAM_CAPACITY: usize = 16;  // 减小容量以避免栈溢出
pub const DETECTIONS_CAPACITY: usize = 32;
pub const PERSON_CLASS_LABEL: &str = "person";

// 目标检测超参数配置
pub const DEFAULT_INPUT_WIDTH: usize = 640;
pub const DEFAULT_INPUT_HEIGHT: usize = 640;
pub const DEFAULT_CONFIDENCE_THRESHOLD: f32 = 0.6;
pub const DEFAULT_NMS_THRESHOLD: f32 = 0.7;