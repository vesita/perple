use image::DynamicImage;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use std::thread;

use crate::{YoloDetector, color::{bounds::Bounds, image::{ScaleMessage}}, config::{DEFAULT_INPUT_WIDTH, DEFAULT_INPUT_HEIGHT}, utils::stream::Stream};
use ort::value::{TensorValueType, Value, Tensor};

/// Color模块的核心结构，用于执行目标检测
/// 
/// 这个结构体封装了整个目标检测流程，包括：
/// - 图像输入管理
/// - 模型推理
/// - 检测结果输出
pub struct Color {
    /// 输入图像流（线程安全）
    input_stream: Arc<Mutex<Stream<DynamicImage>>>,
    /// 输出检测结果流（线程安全）
    output_stream: Arc<Mutex<Stream<Bounds>>>,
    /// YOLO检测器
    model: YoloDetector,
    /// 图像缩放信息
    message: ScaleMessage,
    /// 控制循环运行的标志
    running: bool,
    /// Tensor Value缓存，用于避免拷贝
    tensor_value: Value<TensorValueType<f32>>,
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
        output_stream: Arc<Mutex<Stream<Bounds>>>,
        model_path: &str,
    ) -> Self {
        let input_width = DEFAULT_INPUT_WIDTH;
        let input_height = DEFAULT_INPUT_HEIGHT;
        
        // 初始化一个空的tensor value
        let initial_data = vec![0.0f32; 3 * input_height * input_width];
        let tensor_value = Tensor::from_array(
            ([1, 3, input_height, input_width], initial_data)
        ).unwrap();
        
        Self {
            input_stream,
            output_stream,
            model: YoloDetector::new(model_path, input_width, input_height),
            message: ScaleMessage {
                o_width: 0,
                o_height: 0,
                s_width: input_width as u32,
                s_height: input_height as u32,
            },
            running: false,
            tensor_value,
        }
    }

    // 核心业务逻辑方法
    // ------------------------------------------------------------------------

    /// 执行一次检测操作
    /// 
    /// 该方法会：
    /// 1. 从输入流获取图像
    /// 2. 准备模型输入张量
    /// 3. 执行模型推理
    /// 4. 将结果写入输出流
    pub fn act(&mut self) {
        // 从输入流中读取图像
        let mut input_stream = self.input_stream.lock().unwrap();
        if let Some(input) = input_stream.read() {
            drop(input_stream); // 释放锁
            
            // 处理图像
            self.message.o_width = input.width();
            self.message.o_height = input.height();
            
            // 填充tensor value，避免拷贝
            crate::color::image::fill_input_image(&input, self.model.input_height(), self.model.input_width(), &mut self.tensor_value);
            
            // 执行推理并计时
            let start_time = Instant::now();
            
            // 使用新添加的直接引用方法优化性能
            let mut output_stream = self.output_stream.lock().unwrap();
            if let Ok(slot) = output_stream.get_write_mut() {
                // 初始化或获取Bounds对象
                let bounds = slot.get_or_insert_with(Bounds::new);
                bounds.clear(); // 清空之前的数据
                
                // 执行推理
                if let Err(e) = self.model.infer(&self.tensor_value, bounds, &self.message) {
                    eprintln!("推理过程中发生错误: {:?}", e);
                }
                
                // 提交写入操作
                if let Err(e) = output_stream.commit_write() {
                    eprintln!("提交写入操作时发生错误: {:?}", e);
                }
            } else {
                eprintln!("获取输出流写入位置失败: 缓冲区已满");
            }
            
            let duration = start_time.elapsed();
            println!("模型推理耗时: {:?}", duration);
        }
    }
    
    /// 循环执行检测操作，直到停止信号
    /// 
    /// 此方法会在每次检测后休眠一小段时间，避免过度占用CPU
    pub fn run_loop(&mut self) {
        self.running = true;
        while self.running {
            // 执行检测
            self.act();
            
            // 等待一段时间再进行下一次检测
            // 这里设置为100ms，可以根据需要调整
            thread::sleep(Duration::from_millis(100));
        }
    }

    // 控制方法
    // ------------------------------------------------------------------------

    /// 启动循环检测
    pub fn start(&mut self) {
        self.running = true;
    }
    
    /// 停止循环检测
    pub fn stop(&mut self) {
        self.running = false;
    }
    
    /// 检查是否正在运行
    pub fn is_running(&self) -> bool {
        self.running
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