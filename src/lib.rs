pub mod utils;
pub mod color;

// 重新导出color模块中的常用类型和函数
pub use color::{YoloDetector, Detection, BoundingBox, draw_detections};
pub use color::{load_image, resize_image, image_to_tensor};
pub use color::{load_model};