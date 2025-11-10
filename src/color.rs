//! Color模块 - 实现基于YOLO的目标检测功能
//! 
//! 该模块提供了一整套目标检测功能，包括：
//! - 模型加载
//! - 图像预处理
//! - 模型推理
//! - 结果后处理
//! - 可视化绘制
//! 
//! # 主要组件
//! 
//! - YoloDetector：核心检测器结构，封装了检测流程
//! - load_model：加载ONNX格式的YOLO模型
//! - load_image：加载图像文件
//! - draw_detections：在图像上绘制检测结果
//! 
//! # 工作流程
//! 
//! 1. 使用load_model加载ONNX模型
//! 2. 使用load_image加载待检测图像
//! 3. 创建YoloDetector实例并配置参数
//! 4. 调用detect方法执行检测
//! 5. 使用draw_detections绘制检测结果
//! 
//! # 示例
//! 
//! ```
//! use perple::{load_model, load_image, YoloDetector, draw_detections};
//! 
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let model = load_model("path/to/model.onnx")?;
//! let image = load_image("path/to/image.jpg")?;
//! 
//! let mut detector = YoloDetector::new(model, 640, 640)
//!     .with_confidence_threshold(0.5)
//!     .with_nms_threshold(0.7);
//! 
//! let detections = detector.detect(&image)?;
//! let result_image = draw_detections(&image, &detections);
//! # Ok(())
//! # }
//! ```

pub mod model;
pub mod image;
pub mod detect;
pub mod infer;
pub mod bounds;
pub mod array;

// 重新导出常用类型和函数
pub use detect::{YoloDetector, draw_detections};
pub use model::load_model;
pub use image::{load_image, resize_image, image_to_tensor};
pub use bounds::{Detection, BoundingBox};