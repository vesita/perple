pub const STREAM_CAPACITY: usize = 16;  // 减小容量以避免栈溢出
pub const DETECTIONS_CAPACITY: usize = 16;
pub const PERSON_CLASS_LABEL: &str = "person";
pub const POINTS_CAPACITY: usize = 16384;
pub const RESOLUTION: f32 = 0.07; // 米

// 目标检测超参数配置
pub const DEFAULT_INPUT_WIDTH: usize = 640;
pub const DEFAULT_INPUT_HEIGHT: usize = 640;
pub const DEFAULT_CONFIDENCE_THRESHOLD: f32 = 0.6;
pub const DEFAULT_NMS_THRESHOLD: f32 = 0.7;

// DBSCAN聚类算法参数
pub const DBSCAN_MIN_POINTS: usize = 3;