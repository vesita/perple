pub mod utils;
pub mod color;
pub mod perple;
pub mod config;

// 重新导出color模块中的常用类型和函数
pub use color::{YoloDetector, Detection, BoundingBox, process_detections, to_bounds, draw_detections};
pub use color::{load_image, resize_image, image_to_tensor, input_image};
pub use color::{load_model, nms_tensor};
