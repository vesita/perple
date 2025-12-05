use image::DynamicImage;
use nalgebra::{Matrix4, Vector3, Vector4};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};
use std::fmt;

use crate::color::{YoloDetector, fill_input_image};
use crate::utils::sight::Sight;
use crate::utils::world::OnWorld;
use crate::color::{ClrBud, image::{ScaleMessage}, look::Look};
use crate::config::fixif;
use crate::utils::stream::{Stream, Cream, StreamError};
use ort::value::{Value, Tensor, TensorValueType};

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

impl<T> From<PoisonError<T>> for ColorError {
    fn from(error: PoisonError<T>) -> Self {
        ColorError::PoisonError(format!("线程锁中毒: {:?}", error))
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
        input_stream: Arc<Mutex<Stream<DynamicImage>>>,
        bud_stream: Arc<Mutex<Stream<Vec<ClrBud>>>>,
        model_path: &str,
    ) -> Self {
        let ixi = fixif();
        let input_width = ixi.default_input_width;
        let input_height = ixi.default_input_height;
        
        // 初始化一个空的tensor value
        let initial_data = vec![0.0f32; 3 * input_height * input_width];
        let tensor_value = Tensor::from_array(
            ([1, 3, input_height, input_width], initial_data)
        ).unwrap();
        
        Self {
            cream: Cream {
                in_stream: input_stream,
                out_stream: bud_stream,
            },
            model: YoloDetector::new(model_path, input_width, input_height),
            message: ScaleMessage {
                o_width: 0,
                o_height: 0,
                s_width: input_width as u32,
                s_height: input_height as u32,
            },
            tensor_value,
        }
    }

    /// 执行一次检测操作
    /// 
    /// 该方法会：
    /// 1. 从输入流获取图像
    /// 2. 准备模型输入张量
    /// 3. 执行模型推理
    /// 4. 将结果写入输出流
    pub fn act(&mut self) -> Result<(), ColorError> {
        // 从输入流中读取图像
        let input = match self.cream.read() {
            Some(img) => img,
            None => return Ok(()), // 没有数据可处理，这不是错误
        };
        
        // 处理图像
        self.message.o_width = input.width();
        self.message.o_height = input.height();
        
        // 填充tensor value，避免拷贝
        fill_input_image(&input, self.model.input_height(),
                self.model.input_width(), &mut self.tensor_value);
        
        // 执行推理并计时
        let start_time = Instant::now();
        
        // 获取输出流的可变引用并填充数据
        let mut output_stream = self.cream.out_stream.lock()?;
        
        // 获取写入位置的可变引用
        let write_mut_result = output_stream.get_write_mut();
        match write_mut_result {
            Ok(slot) => {
                // 初始化或获取Vec<ClrBud>对象
                let bounds = slot.get_or_insert_with(|| Vec::new());
                bounds.clear(); // 清空之前的数据
                
                // 执行推理
                let infer_result = self.model.infer(&self.tensor_value, bounds, &self.message);
                match infer_result {
                    Ok(_) => {
                        // 提交写入操作
                        output_stream.commit_write()
                            .map_err(|e| ColorError::CommitError(format!("{:?}", e)))?;
                    },
                    Err(e) => {
                        eprintln!("推理过程中发生错误: {:?}", e);
                        // 即使推理出错，也尝试提交写入以保持流的一致性
                        let _ = output_stream.commit_write();
                        return Err(ColorError::InferenceError(format!("{:?}", e)));
                    }
                }
            },
            Err(e) => {
                return Err(ColorError::from(e));
            }
        }
        
        let duration = start_time.elapsed();
        println!("模型推理耗时: {:?}", duration);
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

    pub fn new(
        input_stream: Arc<Mutex<Stream<DynamicImage>>>,
        bud_stream: Arc<Mutex<Stream<Vec<ClrBud>>>>,
        sight_stream: Arc<Mutex<Stream<Vec<Sight>>>>,
        model_path: &str,
        config_path: &str,
    ) -> Self {
        Self {
            data: Color::new(input_stream, bud_stream.clone(), model_path),
            look: Look::new(bud_stream, sight_stream, config_path),
        }
    }

    pub fn act(&mut self) -> Result<(), ColorError> {
        let _ = self.data.act();
        let _ = self.look.act();
        Ok(())
    }
}

impl OnWorld for Camera {
    fn on_world(&self) -> Matrix4<f32> {
        self.look.extrinsic
    }
    
    fn set_by_angle(&mut self, tra: Vector3<f32>, rot: Vector3<f32>) {
       self.look.set_by_angle(tra, rot); 
    }
    
    fn set_by_radian(&mut self, tra: Vector3<f32>, rot: Vector3<f32>) {
        self.look.set_by_radians(tra, rot);
    }
    
    fn set_by_matrix(&mut self, matrix: &Matrix4<f32>) {
        self.look.extrinsic = *matrix;
    }
}