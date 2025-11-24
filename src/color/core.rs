use image::DynamicImage;
use nalgebra::{Matrix3, Matrix4, Vector3, Vector4};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use std::thread;

use crate::{YoloDetector, color::{bounds::ClrBud, image::{ScaleMessage}}, config::{DETECTIONS_CAPACITY, DEFAULT_INPUT_WIDTH, DEFAULT_INPUT_HEIGHT}, utils::stream::{Stream, Cream}};
use ort::value::{TensorValueType, Value, Tensor};
use crate::utils::world::OnWorld;

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
    intrinsic: Matrix3<f32>,
    extrinsic: Matrix4<f32>,
}

impl Camera {
    pub fn new(
        input_stream: Arc<Mutex<Stream<DynamicImage>>>,
        output_stream: Arc<Mutex<Stream<Vec<ClrBud>>>>,
        model_path: &str,
    ) -> Self {
        Self {
            data: Color::new(input_stream, output_stream, model_path),
            intrinsic: Matrix3::identity(),
            extrinsic: Matrix4::identity(),
        }
    }
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
        output_stream: Arc<Mutex<Stream<Vec<ClrBud>>>>,
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
            cream: Cream {
                in_stream: input_stream,
                out_stream: output_stream,
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
        if let Some(input) = self.cream.read() {
            // 处理图像
            self.message.o_width = input.width();
            self.message.o_height = input.height();
            
            // 填充tensor value，避免拷贝
            crate::color::image::fill_input_image(&input, self.model.input_height(), self.model.input_width(), &mut self.tensor_value);
            
            // 执行推理并计时
            let start_time = Instant::now();
            
            // 使用新添加的直接引用方法优化性能
            let mut output_stream = self.cream.out_stream.lock().unwrap();
            if let Ok(slot) = output_stream.get_write_mut() {
                // 初始化或获取Bounds对象
                let bounds = slot.get_or_insert_with(|| Vec::new());
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

impl OnWorld for Camera {
    fn on_world(&self) -> Matrix4<f32> {
        self.extrinsic
    }

    fn set_by_angle(&mut self, tra: Vector3<f32>, rot: Vector3<f32>) {
        let rot_rad = Vector3::new(
            rot.x.to_radians(),
            rot.y.to_radians(),
            rot.z.to_radians(),
        );
        self.extrinsic = Matrix4::new_rotation(rot_rad) * Matrix4::new_translation(&tra);
    }

    fn set_by_radian(&mut self, tra: Vector3<f32>, rot: Vector3<f32>) {
        self.extrinsic = Matrix4::new_rotation(rot) * Matrix4::new_translation(&tra);
    }

    fn set_by_matrix(&mut self, matrix: &Matrix4<f32>) {
        self.extrinsic = *matrix;
    }
    
    // fn point_iter(&self) -> Box<dyn Iterator<Item = Vector3<f32>>> {
    //     // 从输入流中获取当前检测框数据
    //     let points: Vec<Vector3<f32>> = if let Some(clr_buds) = self.data.cream.out_stream
    //         .lock()
    //         .unwrap()
    //         .get_read_ref()
    //         .and_then(|opt| opt.as_ref()) {
    //         // 获取内外参矩阵的逆矩阵用于坐标变换
    //         let inv_extrinsic = self.extrinsic.try_inverse().unwrap_or_else(|| Matrix4::identity());
    //         let inv_intrinsic = self.intrinsic.try_inverse().unwrap_or_else(|| Matrix3::identity());
            
    //         // 将每个检测框的中心点从图像坐标转换为世界坐标
    //         clr_buds.iter()
    //             .map(|bud| {
    //                 // 计算检测框的中心点
    //                 let center_x = (bud.the_box.x1 + bud.the_box.x2) / 2.0;
    //                 let center_y = (bud.the_box.y1 + bud.the_box.y2) / 2.0;
                    
    //                 // 将2D像素坐标转换为归一化相机坐标
    //                 let normalized_camera_coords = inv_intrinsic * Vector3::new(center_x, center_y, 0.0);
                    
    //                 // 构造齐次坐标向量
    //                 let homogeneous_coords = Vector4::new(
    //                     normalized_camera_coords.x,
    //                     normalized_camera_coords.y,
    //                     normalized_camera_coords.z,
    //                     1.0
    //                 );
                    
    //                 // 应用外参矩阵的逆变换，得到世界坐标
    //                 let world_coords = inv_extrinsic * homogeneous_coords;
                    
    //                 // 返回3D世界坐标（忽略齐次坐标）
    //                 Vector3::new(world_coords.x, world_coords.y, world_coords.z)
    //             })
    //             .collect()
    //     } else {
    //         Vec::new()
    //     };

    //     Box::new(points.into_iter())
    // }
}
