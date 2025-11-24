use std::sync::{Arc, Mutex};
use std::time::Instant;

use nalgebra::{Matrix3, Matrix4, Vector3, Vector4};

use crate::utils::world::OnWorld;
use crate::{cloud::{bounds::CldBud, claster::Claster, lifra::Lifra}, utils::stream::{Stream, Cream}};

/// Lidar模块的核心结构，用于执行点云处理
pub struct Cloud {
    cream: Cream<Lifra, Vec<CldBud>>,
    // 添加claster作为成员变量以避免重复创建
    claster: Claster,
}

pub struct Lidar {
    data: Cloud,
    extrinsic: Matrix4<f32>,
}

impl Cloud {
    pub fn new(
        input_stream: Arc<Mutex<Stream<Lifra>>>,
        output_stream: Arc<Mutex<Stream<Vec<CldBud>>>>,
    ) -> Self {
        Self {
            cream: Cream {
                in_stream: input_stream,
                out_stream: output_stream,
            },
            claster: Claster::new(),
        }
    }

    /// 执行一次点云处理操作
    /// 
    /// 该方法会：
    /// 1. 从输入流获取点云数据
    /// 2. 使用Claster直接处理整个帧数据
    /// 3. 将结果写入输出流
    pub fn act(&mut self) {
        // 从输入流中读取点云数据
        let lifra = match self.read_input() {
            Some(data) => data,
            None => return,
        };
        
        // 处理点云数据
        let start_time = Instant::now();
        self.process_frame(&lifra);
        let process_duration = start_time.elapsed();
        
        // 将结果写入输出流
        self.write_output();
        
        let duration = start_time.elapsed();
        println!("点云处理耗时: {:?}", process_duration);
        println!("点云IO耗时: {:?}", duration - process_duration);
    }
    
    /// 从输入流中读取点云数据
    fn read_input(&mut self) -> Option<Lifra> {
        self.cream.read()
    }
    
    /// 处理点云帧数据
    fn process_frame(&mut self, lifra: &Lifra) {
        // 直接使用Claster处理整个帧数据
        self.claster.claster(lifra);
    }
    
    /// 将处理结果写入输出流
    fn write_output(&mut self) {
        let mut output_stream = self.cream.out_stream.lock().unwrap();
        if let Ok(slot) = output_stream.get_write_mut() {
            // 初始化或获取LidBud对象
            let bounds = slot.get_or_insert_with(|| Vec::new());
            bounds.clear(); // 清空之前的数据
            
            // 将聚类结果转换为LidBud格式
            // 将所有聚类对象添加到LidBud中
            for box3d in self.claster.objects().iter() {
                bounds.push(CldBud {
                    the_box: box3d.clone(),
                    class_id: 0,
                    class_name: String::new(),
                });
            }
            
            // 提交写入操作
            if let Err(e) = output_stream.commit_write() {
                eprintln!("提交写入操作时发生错误: {:?}", e);
            }
        } else {
            eprintln!("获取输出流写入位置失败: 缓冲区已满");
        }
    }

    /// 获取输入输出流的引用
    pub fn cream(&self) -> &Cream<Lifra, Vec<CldBud>> {
        &self.cream
    }
}

impl OnWorld for Lidar { 
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
}