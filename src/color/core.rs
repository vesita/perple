use image::DynamicImage;
use log::info;
use std::fmt;
use std::sync::Arc;
use std::time::Instant;

use crate::color::{ClrBud, image::ScaleMessage, look::Look};
use crate::color::{YoloDetector, fill_input_image};
use crate::config::fixif;
use crate::swapl::global_swapl;
use crate::utils::stream::{Cream, Eap, Stream, StreamError};

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
    /// 输入输出流（线程安全）
    cream: Cream<DynamicImage, Vec<ClrBud>>,
    /// YOLO检测器
    model: YoloDetector,
    /// 图像缩放信息
    message: ScaleMessage,
    /// Tensor Value缓存，用于避免拷贝
    tensor_value: Value<TensorValueType<f32>>,
    /// 本地推理缓冲区（避免推理时持有输出锁）
    local_bounds: Vec<ClrBud>,
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
    /// * `output_stream` - 输出结果流的线程安全引用
    /// * `model_path` - 模型文件路径
    ///
    /// # 返回值
    /// 返回新的Color实例
    pub fn new(
        input_stream: Eap<Stream<DynamicImage>>,
        output_stream: Eap<Stream<Vec<ClrBud>>>,
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
            cream: Cream {
                in_stream: input_stream,
                out_stream: output_stream,
            },
            model,
            message: ScaleMessage {
                o_width: 0,
                o_height: 0,
                s_width: input_width as u32,
                s_height: input_height as u32,
            },
            tensor_value,
            local_bounds: Vec::new(),
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
        let input = {
            let mut stream = self.cream.in_stream.lock().await;
            match stream.read() {
                Some(img) => img,
                None => return Ok(()), // 没有数据可处理，这不是错误
            }
        };

        // 处理图像
        self.message.o_width = input.width();
        self.message.o_height = input.height();

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

        // 推理完成后，获取输出锁并写入结果
        {
            let mut output_stream = self.cream.out_stream.lock().await;
            if infer_ok {
                let write_result = output_stream.get_write_mut();
                match write_result {
                    Ok(slot) => {
                        let bounds = slot.get_or_insert_with(|| Vec::new());
                        std::mem::swap(bounds, &mut self.local_bounds);
                        output_stream
                            .commit_write()
                            .map_err(|e| ColorError::CommitError(format!("{:?}", e)))?;
                    }
                    Err(e) => {
                        return Err(ColorError::from(e));
                    }
                }
            } else {
                eprintln!("推理过程中发生错误");
                let _ = output_stream.commit_write();
                return Err(ColorError::InferenceError("模型推理失败".to_string()));
            }
        } // 在这里释放 output_stream 锁

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
    pub fn cream(&self) -> &Cream<DynamicImage, Vec<ClrBud>> {
        &self.cream
    }

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

        Self {
           data: Color::new(Arc::clone(&pool.colors), Arc::clone(&pool.clr_objs)),
            look: Look::new(Arc::clone(&pool.clr_objs), Arc::clone(&pool.sights)),
        }
    }

    pub async fn act(&mut self) -> Result<(), ColorError> {
        self.data.act().await?;
        self.look.act().await;
        Ok(())
    }
}
