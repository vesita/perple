use ort::{session::{Session, input}, value::{TensorValueType, Value}};
use image::{DynamicImage, GenericImageView};
use raqote::{DrawOptions, DrawTarget, LineJoin, PathBuilder, SolidSource, Source, StrokeStyle};
use std::time::Instant;
use crate::{color::{array::to_input, bounds::{Bounds, Detection}, image::{ScaleMessage, input_image, resize_image, image_to_tensor}, utils::{nms_tensor}}, config::{DETECTIONS_CAPACITY, DEFAULT_INPUT_WIDTH, DEFAULT_INPUT_HEIGHT, DEFAULT_CONFIDENCE_THRESHOLD, DEFAULT_NMS_THRESHOLD}, load_model};
use ndarray::{Array2, Array4, s};
use ort::{value::Tensor, inputs};

/// YOLO目标检测器
/// 
/// 封装了完整的检测流程，包括图像预处理、模型推理和结果后处理。
/// 
/// # 示例
/// 
/// ```
/// use perple::color::{YoloDetector, load_model};
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
    /// NMS处理中使用的缓存数组，避免重复分配内存
    picked_indices: [bool; DETECTIONS_CAPACITY],
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
    /// use perple::color::{YoloDetector, load_model};
    /// 
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let model = load_model("path/to/model.onnx")?;
    /// let detector = YoloDetector::new(model, 640, 640);
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(model_path: &str, input_width: usize, input_height: usize) -> Self {
        let model = load_model(model_path).expect("模型加载失败");
        Self {
            model,
            input_width,
            input_height,
            confidence_threshold: DEFAULT_CONFIDENCE_THRESHOLD,
            nms_threshold: DEFAULT_NMS_THRESHOLD,
            picked_indices: [false; DETECTIONS_CAPACITY],
        }
    }

    /// 创建新的YoloDetector实例，使用默认输入尺寸
    /// 
    /// # 参数
    /// * `model_path` - 模型文件路径
    /// 
    /// # 返回值
    /// 返回新的YoloDetector实例
    pub fn with_default_size(model_path: &str) -> Self {
        Self::new(model_path, DEFAULT_INPUT_WIDTH, DEFAULT_INPUT_HEIGHT)
    }

    /// 执行模型推理
    /// 
    /// # 参数
    /// * `input` - 输入张量
    /// * `outputs` - 输出结果容器
    /// * `message` - 图像缩放信息
    /// 
    /// # 返回值
    /// 返回推理结果
    pub fn infer(&mut self,
        input: &Value<TensorValueType<f32>>,
        outputs: &mut Bounds,
        message: &ScaleMessage,
    ) -> Result<(), Box<dyn std::error::Error>> {
        outputs.clear();
        let mut result = self.model.run(inputs!["images" => input])?;
        nms_tensor(&mut result, outputs, message, &mut self.picked_indices, self.confidence_threshold, self.nms_threshold);
        Ok(())
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
    
    /// 设置置信度阈值（可变引用版本）
    pub fn set_confidence_threshold(&mut self, threshold: f32) {
        self.confidence_threshold = threshold;
    }
    
    /// 设置NMS阈值（可变引用版本）
    pub fn set_nms_threshold(&mut self, threshold: f32) {
        self.nms_threshold = threshold;
    }
    
    /// 获取当前置信度阈值
    pub fn confidence_threshold(&self) -> f32 {
        self.confidence_threshold
    }
    
    /// 获取当前NMS阈值
    pub fn nms_threshold(&self) -> f32 {
        self.nms_threshold
    }
    
    /// 获取模型输入宽度
    pub fn input_width(&self) -> usize {
        self.input_width
    }
    
    /// 获取模型输入高度
    pub fn input_height(&self) -> usize {
        self.input_height
    }

    /// 运行模型推理
    /// 
    /// 使用ONNX模型对输入张量进行推理，返回处理后的结果。
    /// 
    /// # 参数
    /// * `input` - 输入张量，形状应为(1, 3, height, width)
    /// 
    /// # 返回值
    /// 返回处理后的模型输出，形状为(num_boxes, 5)，包含[x, y, w, h, conf]
    /// 
    /// # 错误处理
    /// 如果推理过程中发生错误会返回Err
    pub fn infer_old(&mut self, input: &Array4<f32>) -> Result<Array2<f32>, Box<dyn std::error::Error>> {
        // 运行模型推理
        let input_tensor = to_input(input);
        let outputs = self.model.run(inputs!["images" => input_tensor])?;
        
        // 提取输出并处理
        let output = outputs[0].try_extract_tensor::<f32>()?;
        let shape = output.0.clone();
        
        // 验证输出形状
        if shape.len() != 3 || shape[0] != 1 {
            return Err("模型输出形状不符合预期".into());
        }
        
        // YOLO模型输出形状为 [1, num_boxes, num_params]
        // 其中num_params通常为6: [x, y, w, h, conf, class_conf] 
        let data = output.1.to_vec();
        
        // 将数据重塑为二维数组 [num_boxes, num_params]
        let array_2d = Array2::from_shape_vec((shape[1] as usize, shape[2] as usize), data)?;
        
        // 只取前5列 [x, y, w, h, conf]
        let array = array_2d.slice(s![.., 0..5]).to_owned();
        
        Ok(array)
    }

    /// 完整的检测流程：从图像到检测结果（旧版）
    /// 
    /// 对输入图像执行完整的检测流程，包括预处理、推理和后处理。
    /// 
    /// # 参数
    /// * `image` - 输入图像
    /// 
    /// # 返回值
    /// 返回检测结果列表
    /// 
    /// # 错误处理
    /// 如果检测过程中发生错误会返回Err
    pub fn detect(&mut self, image: &DynamicImage) -> Result<Bounds, Box<dyn std::error::Error>> {
        // 调整图像大小
        let resized = resize_image(image, self.input_width as u32, self.input_height as u32);
        
        // 转换为张量
        let tensor = image_to_tensor(&resized, self.input_height, self.input_width);
        
        // 运行推理
        let input_tensor = to_input(&tensor);
        let mut outputs = Bounds::new();
        let scale_message = ScaleMessage {
            o_width: image.width(),
            o_height: image.height(),
            s_width: self.input_width as u32,
            s_height: self.input_height as u32,
        };
        
        self.infer(&input_tensor, &mut outputs, &scale_message)?;
        
        Ok(outputs)
    }
    
    /// 对一批图像执行检测
    /// 
    /// # 参数
    /// * `images` - 图像数组
    /// 
    /// # 返回值
    /// 返回每张图像的检测结果
    pub fn detect_batch(&mut self, images: &[DynamicImage]) -> Result<Vec<Bounds>, Box<dyn std::error::Error>> {
        let mut results = Vec::with_capacity(images.len());
        
        for image in images {
            let result = self.detect(image)?;
            results.push(result);
        }
        
        Ok(results)
    }
}

// 为YoloDetector实现Debug trait
impl std::fmt::Debug for YoloDetector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("YoloDetector")
            .field("input_width", &self.input_width)
            .field("input_height", &self.input_height)
            .field("confidence_threshold", &self.confidence_threshold)
            .field("nms_threshold", &self.nms_threshold)
            .finish()
    }
}