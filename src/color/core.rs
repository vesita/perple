use image::DynamicImage;
use log::info;
use std::fmt;
use std::sync::Arc;
use std::time::Instant;

use crate::color::{ClrBud, image::ScaleMessage, look::Look, UndistortMap};
use crate::color::{YoloDetector, fill_input_image};
use crate::config::fixif;
use crate::swapl::global_swapl;
use crate::utils::stream::{DualBuf, Eap, Stream, StreamError};

use ort::value::{Tensor, TensorValueType, Value};

/// Color模块的错误类型
#[derive(Debug)]
pub enum ColorError {
    /// 流缓冲区相关错误
    StreamError(StreamError),
    /// 模型推理错误
    InferenceError(String),
    /// 线程锁中毒错误
    PoisonError(String),
    /// 提交写入操作错误
    CommitError(String),
    /// 其他错误
    Other(String),
}

impl fmt::Display for ColorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ColorError::StreamError(e) => write!(f, "流错误: {}", e),
            ColorError::InferenceError(e) => write!(f, "推理错误: {}", e),
            ColorError::PoisonError(e) => write!(f, "线程锁中毒错误: {}", e),
            ColorError::CommitError(e) => write!(f, "提交写入操作错误: {}", e),
            ColorError::Other(e) => write!(f, "其他错误: {}", e),
        }
    }
}

impl std::error::Error for ColorError {}

impl From<StreamError> for ColorError {
    fn from(error: StreamError) -> Self {
        ColorError::StreamError(error)
    }
}

/// Color模块的核心结构，用于执行目标检测
///
/// 这个结构体封装了整个目标检测流程，包括：
/// - 图像输入管理
/// - 模型推理
/// - 检测结果输出
pub struct Color {
    /// 输入流（图像）
    in_stream: Eap<Stream<DynamicImage>>,
    /// YOLO检测器
    model: YoloDetector,
    /// 图像缩放信息
    message: ScaleMessage,
    /// Tensor Value缓存，用于避免拷贝
    tensor_value: Value<TensorValueType<f32>>,
    /// 本地推理缓冲区（避免推理时持有输出锁）
    local_bounds: Vec<ClrBud>,
    /// 去畸变映射（预计算，复用）
    undistort_map: Option<UndistortMap>,
    /// 双缓冲 producer：检测阶段写入 YOLO 结果（后融合阶段读 consumer）
    clr_objs: DualBuf<Vec<ClrBud>>,
}

pub struct Camera {
    data: Color,
    look: Look,
}

impl Color {
    // 构造函数和初始化方法
    // ------------------------------------------------------------------------

    /// 创建一个新的Color实例
    ///
    /// # 参数
    /// * `input_stream` - 输入图像流的线程安全引用
    /// * `clr_objs` - DualBuf producer，检测阶段写入 YOLO 结果
    pub fn new(
        input_stream: Eap<Stream<DynamicImage>>,
        clr_objs: DualBuf<Vec<ClrBud>>,
    ) -> Self {
        // 初始化YOLO检测器
        let model = YoloDetector::new();

        let ixi = fixif();
        let input_width = ixi.default_input_width;
        let input_height = ixi.default_input_height;

        // 初始化一个空的tensor value
        let initial_data = vec![0.0f32; 3 * input_height * input_width];
        let tensor_value =
            Tensor::from_array(([1, 3, input_height, input_width], initial_data)).unwrap();

        Self {
            in_stream: input_stream,
            model,
            message: ScaleMessage {
                o_width: 0,
                o_height: 0,
                s_width: input_width as u32,
                s_height: input_height as u32,
                pad_x: 0.0,
                pad_y: 0.0,
                scale: 1.0,
            },
            tensor_value,
            local_bounds: Vec::new(),
            undistort_map: None,
            clr_objs,
        }
    }

    /// 执行一次检测操作
    ///
    /// 该方法会：
    /// 1. 从输入流获取图像
    /// 2. 准备模型输入张量
    /// 3. 执行模型推理
    /// 4. 将结果写入输出流
    pub async fn act(&mut self) -> Result<(), ColorError> {
        // 从输入流中读取图像
        let mut input = {
            let mut stream = self.in_stream.lock().unwrap();
            match stream.read() {
                Some(img) => img,
                None => return Ok(()), // 没有数据可处理，这不是错误
            }
        };

        // 去畸变（如果配置了 dist_coeffs）
        if let Some(ref dist) = fixif().camera.dist_coeffs {
            if self.undistort_map.is_none() {
                self.undistort_map = Some(UndistortMap::new(
                    &fixif().camera.intrinsic,
                    dist,
                    input.width(),
                    input.height(),
                ));
            }
            if let Some(ref map) = self.undistort_map {
                let t = Instant::now();
                input = map.apply(&input);
                log::debug!("去畸变耗时: {:?}", t.elapsed());
            }
        }

        // 处理图像
        self.message.o_width = input.width();
        self.message.o_height = input.height();

        // Letterbox: 计算保持宽高比的缩放和填充偏移
        let target_w = self.message.s_width as f32;
        let target_h = self.message.s_height as f32;
        let src_w = input.width() as f32;
        let src_h = input.height() as f32;
        let scale = (target_w / src_w).min(target_h / src_h);
        let new_w = (src_w * scale).round();
        let new_h = (src_h * scale).round();
        self.message.pad_x = (target_w - new_w) / 2.0;
        self.message.pad_y = (target_h - new_h) / 2.0;
        self.message.scale = scale;

        // 填充 tensor value，避免拷贝
        fill_input_image(
            &input,
            self.model.input_height(),
            self.model.input_width(),
            &mut self.tensor_value,
        );

        // 执行推理并计时（不持有输出锁，避免阻塞 lidar 的 YOLO 细化）
        let start_time = Instant::now();

        self.local_bounds.clear();
        let infer_ok = self.model.infer(&self.tensor_value, &mut self.local_bounds, &self.message).is_ok();

        // 推理完成后，写入 DualBuf producer（检测阶段，无跨阶段竞争）
        if infer_ok {
            *self.clr_objs.producer().lock().unwrap() = std::mem::take(&mut self.local_bounds);
        } else {
            eprintln!("推理过程中发生错误");
            return Err(ColorError::InferenceError("模型推理失败".to_string()));
        }

        let duration = start_time.elapsed();
        println!("模型推理耗时：{:?}", duration);
        Ok(())
    }

    // Getter方法
    // ------------------------------------------------------------------------

    /// 获取模型引用
    pub fn model(&self) -> &YoloDetector {
        &self.model
    }

    /// 获取可变模型引用
    pub fn model_mut(&mut self) -> &mut YoloDetector {
        &mut self.model
    }

    /// 获取输入输出流的引用
    // 模型参数设置方法
    // ------------------------------------------------------------------------

    /// 更新模型置信度阈值
    pub fn set_confidence_threshold(&mut self, threshold: f32) {
        self.model.set_confidence_threshold(threshold);
    }

    /// 更新模型NMS阈值
    pub fn set_nms_threshold(&mut self, threshold: f32) {
        self.model.set_nms_threshold(threshold);
    }
}

impl Camera {
    /// 创建 Camera 实例，通过全局 Swapl 数据中枢进行数据交互
    ///
    /// 所有数据交互都通过全局 Swapl 完成，实现了模块间的松耦合设计。
    /// Camera 模块内部保留指向各模块的指针，但不再需要外部传入数据流引用
  pub fn new() -> Self {
        // 获取全局数据交换中枢
      let pool = global_swapl();

      info!("Camera 模块初始化");

        // clr_objs 改用 DualBuf：Camera 写 producer，Fuse 读 consumer
        let clr_objs: DualBuf<Vec<ClrBud>> = pool.clr_objs.clone();

        Self {
           data: Color::new(Arc::clone(&pool.colors), clr_objs.clone()),
            look: Look::new(clr_objs, Arc::clone(&pool.sights)),
        }
    }

    pub async fn act(&mut self) -> Result<(), ColorError> {
        self.data.act().await?;
        self.look.act().await;
        Ok(())
    }
}
