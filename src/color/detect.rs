use ort::session::Session;
use image::{DynamicImage, GenericImageView};
use raqote::{DrawOptions, DrawTarget, LineJoin, PathBuilder, SolidSource, Source, StrokeStyle};
use std::time::Instant;
use crate::color::prevs::{resize_image, image_to_tensor};
use crate::color::infer::run_inference;
use crate::color::posts::{Detection, process_detections};

/// YOLO目标检测器
/// 
/// 封装了完整的检测流程，包括图像预处理、模型推理和结果后处理。
/// 
/// # 示例
/// 
/// ```
/// use perple::{YoloDetector, load_model};
/// 
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let model = load_model("path/to/model.onnx")?;
/// let mut detector = YoloDetector::new(model, 640, 640)
///     .with_confidence_threshold(0.5)
///     .with_nms_threshold(0.7);
/// # Ok(())
/// # }
/// ```
pub struct YoloDetector {
    /// ONNX模型会话
    model: Session,
    /// 模型输入宽度
    input_width: usize,
    /// 模型输入高度
    input_height: usize,
    /// 置信度阈值，低于此值的检测结果将被过滤
    confidence_threshold: f32,
    /// NMS（非极大值抑制）阈值，用于去除重复检测
    nms_threshold: f32,
}

impl YoloDetector {
    /// 创建新的YoloDetector实例
    /// 
    /// # 参数
    /// * `model` - 已加载的ONNX模型
    /// * `input_width` - 模型输入图像宽度
    /// * `input_height` - 模型输入图像高度
    /// 
    /// # 返回值
    /// 返回新的YoloDetector实例
    /// 
    /// # 示例
    /// 
    /// ```
    /// use perple::{YoloDetector, load_model};
    /// 
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let model = load_model("path/to/model.onnx")?;
    /// let detector = YoloDetector::new(model, 640, 640);
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(model: Session, input_width: usize, input_height: usize) -> Self {
        Self {
            model,
            input_width,
            input_height,
            confidence_threshold: 0.5,
            nms_threshold: 0.7,
        }
    }

    /// 设置置信度阈值
    /// 
    /// # 参数
    /// * `threshold` - 置信度阈值 (0.0 - 1.0)
    /// 
    /// # 返回值
    /// 返回配置了新置信度阈值的YoloDetector实例
    pub fn with_confidence_threshold(mut self, threshold: f32) -> Self {
        self.confidence_threshold = threshold;
        self
    }

    /// 设置NMS阈值
    /// 
    /// # 参数
    /// * `threshold` - NMS阈值 (0.0 - 1.0)
    /// 
    /// # 返回值
    /// 返回配置了新NMS阈值的YoloDetector实例
    pub fn with_nms_threshold(mut self, threshold: f32) -> Self {
        self.nms_threshold = threshold;
        self
    }

    /// 完整的检测流程：从图像到检测结果
    /// 
    /// 执行完整的检测流程，包括图像预处理、模型推理和结果后处理。
    /// 
    /// # 参数
    /// * `img` - 待检测的图像
    /// 
    /// # 返回值
    /// 返回检测结果列表
    /// 
    /// # 错误处理
    /// 如果检测过程中发生错误会返回Err
    /// 
    /// # 示例
    /// 
    /// ```
    /// use perple::{YoloDetector, load_model, load_image};
    /// 
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let model = load_model("path/to/model.onnx")?;
    /// let image = load_image("path/to/image.jpg")?;
    /// let mut detector = YoloDetector::new(model, 640, 640);
    /// let detections = detector.detect(&image)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn detect(&mut self, img: &DynamicImage) -> Result<Vec<Detection>, Box<dyn std::error::Error>> {
        let (img_width, img_height) = (img.width() as f32, img.height() as f32);
        let resized_img = resize_image(img, self.input_width as u32, self.input_height as u32);
        let input_tensor = image_to_tensor(&resized_img, self.input_height, self.input_width);
        
        // 为模型推理添加计时
        let start_time = Instant::now();
        let output = run_inference(&mut self.model, &input_tensor)?;
        let duration = start_time.elapsed();
        println!("模型推理耗时: {:?}", duration);
        
        let detections = process_detections(
            output,
            img_width,
            img_height,
            self.input_width,
            self.input_height,
            self.confidence_threshold,
            self.nms_threshold
        );
        Ok(detections)
    }
}

/// 在图像上绘制检测结果
/// 
/// 使用不同颜色绘制检测框，person类别使用青色，其他类别使用红色。
/// 
/// # 参数
/// * `image` - 原始图像
/// * `detections` - 检测结果列表
/// 
/// # 返回值
/// 返回绘制了检测框的图像
/// 
/// # 示例
/// 
/// ```
/// use perple::{load_image, draw_detections};
/// 
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let image = load_image("path/to/image.jpg")?;
/// // 假设已有检测结果
/// // let result_image = draw_detections(&image, &detections);
/// # Ok(())
/// # }
/// ```
pub fn draw_detections(image: &DynamicImage, detections: &[Detection]) -> DynamicImage {
    let (img_width, img_height) = image.dimensions();
    let mut dt = DrawTarget::new(img_width as i32, img_height as i32);
    
    // 将原始图像绘制到DrawTarget上
    let rgba_image = image.to_rgba8();
    let image_data: Vec<u32> = rgba_image.chunks(4).map(|pixel| {
        let b = pixel[2];
        let g = pixel[1];
        let r = pixel[0];
        let a = pixel[3];
        u32::from_le_bytes([b, g, r, a])
    }).collect();
    
    let img = raqote::Image {
        width: img_width as i32,
        height: img_height as i32,
        data: &image_data,
    };
    
    dt.draw_image_at(0.0, 0.0, &img, &DrawOptions::new());

    for detection in detections {
        let bbox = &detection.bbox;

        let mut pb = PathBuilder::new();
        let width = bbox.x2 - bbox.x1;
        let height = bbox.y2 - bbox.y1;

        pb.rect(bbox.x1, bbox.y1, width, height);
        let path = pb.finish();
        
        // 根据类别设置不同颜色
        let color = match detection.class_id {
            0 => SolidSource { r: 0x00, g: 0xFF, b: 0xFF, a: 0xFF }, // 青色 - person类别
            _ => SolidSource { r: 0xFF, g: 0x00, b: 0x00, a: 0xFF }, // 红色 - 其他类别
        };
        
        dt.stroke(
            &path,
            &Source::Solid(color),
            &StrokeStyle {
                join: LineJoin::Round,
                width: 2.0,
                ..StrokeStyle::default()
            },
            &DrawOptions::default()
        );
        
        // 可以添加文本标签显示类别和置信度
        // 这里暂时省略，如需要可后续添加
    }

    // 将DrawTarget转换回图像
    let pixels: Vec<u8> = dt.get_data().iter().flat_map(|&pixel| {
        let bytes = pixel.to_le_bytes();
        vec![bytes[2], bytes[1], bytes[0], bytes[3]] // BGRA to RGBA
    }).collect();
    
    DynamicImage::ImageRgba8(
        image::ImageBuffer::from_raw(img_width, img_height, pixels)
            .expect("Failed to create image from rendered data")
    )
}