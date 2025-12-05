use std::sync::{Arc, Mutex, PoisonError};
use std::time::Instant;
use std::fmt;

use nalgebra::{Matrix4, Vector3, Vector4};

use crate::utils::world::OnWorld;
use crate::{cloud::{CldBud, claster::Claster, lifra::Lifra}, utils::stream::{Stream, Cream, StreamError}};
use crate::config::fixif;

/// Lidar模块的错误类型
#[derive(Debug)]
pub enum LidarError {
    /// 流缓冲区相关错误
    StreamError(StreamError),
    /// 线程锁中毒错误
    PoisonError(String),
    /// 提交写入操作错误
    CommitError(String),
    /// 其他错误
    Other(String),
}

impl fmt::Display for LidarError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LidarError::StreamError(e) => write!(f, "流错误: {}", e),
            LidarError::PoisonError(e) => write!(f, "线程锁中毒错误: {}", e),
            LidarError::CommitError(e) => write!(f, "提交写入操作错误: {}", e),
            LidarError::Other(e) => write!(f, "其他错误: {}", e),
        }
    }
}

impl std::error::Error for LidarError {}

impl From<StreamError> for LidarError {
    fn from(error: StreamError) -> Self {
        LidarError::StreamError(error)
    }
}

impl<T> From<PoisonError<T>> for LidarError {
    fn from(error: PoisonError<T>) -> Self {
        LidarError::PoisonError(format!("线程锁中毒: {:?}", error))
    }
}

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
    pub fn fast_act(&mut self) -> Result<(), LidarError> {
        // 从输入流中读取点云数据
        let lifra = match self.read_input() {
            Some(data) => data,
            None => return Ok(()), // 没有数据可处理，这不是错误
        };
        
        // 处理点云数据
        let start_time = Instant::now();
        self.process_frame(&lifra);
        let process_duration = start_time.elapsed();
        
        // 将结果写入输出流
        self.write_output()?;
        
        let duration = start_time.elapsed();
        println!("点云处理耗时: {:?}", process_duration);
        println!("点云IO耗时: {:?}", duration - process_duration);
        Ok(())
    }
    
    /// 带前置和后置处理函数的执行方法
    /// 
    /// 该方法会：
    /// 1. 执行前置处理函数
    /// 2. 从输入流获取点云数据
    /// 3. 使用Claster直接处理整个帧数据
    /// 4. 将结果写入输出流
    /// 5. 执行后置处理函数
    /// 
    /// # 参数
    /// * `pre_process` - 前置处理函数，接收并返回Lifra点云数据
    /// * `post_process` - 后置处理函数，在处理完成后调用
    pub fn act<F>(&mut self, prep: F) -> Result<(), LidarError>
    where 
        F: FnOnce(Lifra) -> Lifra,
    {
        // 从输入流中读取点云数据
        let lifra = match self.read_input() {
            Some(data) => data,
            None => return Ok(()), // 没有数据可处理，这不是错误
        };
        
        // 执行前置处理
        let clouds_in_world = prep(lifra);
        
        // 处理点云数据
        let start_time = Instant::now();
        self.process_frame(&clouds_in_world);
        let process_duration = start_time.elapsed();
        
        // 将结果写入输出流
        self.write_output()?;
        
        let duration = start_time.elapsed();
        println!("点云处理耗时: {:?}", process_duration);
        println!("点云IO耗时: {:?}", duration - process_duration);
        Ok(())
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
    fn write_output(&mut self) -> Result<(), LidarError> {
        let mut output_stream = self.cream.out_stream.lock()?;
        let write_mut_result = output_stream.get_write_mut();
        match write_mut_result {
            Ok(slot) => {
                // 初始化或获取CldBud对象
                let bounds = slot.get_or_insert_with(|| Vec::new());
                bounds.clear(); // 清空之前的数据
                
                // 将聚类结果转换为CldBud格式
                // 将所有聚类对象添加到CldBud中
                for box3d in self.claster.objects().iter() {
                    bounds.push(CldBud::new(
                        *box3d,
                        0,              // class_id: 默认为0，表示未分类
                        String::new(),  // class_name: 默认为空字符串
                        0.0             // confidence: 默认置信度为0.0
                    ));
                }
                
                // 提交写入操作
                output_stream.commit_write()
                    .map_err(|e| LidarError::CommitError(format!("{:?}", e)))?;
            },
            Err(e) => {
                return Err(LidarError::from(e));
            }
        }
        Ok(())
    }

    /// 获取输入输出流的引用
    pub fn cream(&self) -> &Cream<Lifra, Vec<CldBud>> {
        &self.cream
    }
}

impl Lidar {
    pub fn new(
        input_stream: Arc<Mutex<Stream<Lifra>>>,
        output_stream: Arc<Mutex<Stream<Vec<CldBud>>>>,
    ) -> Self {
        // 从全局配置中获取lidar外参
        let lidar_config = &fixif().lidar;
        
        // 将数组转换为矩阵
        let extrinsic = Matrix4::from_iterator(lidar_config.extrinsic.iter().flatten().cloned());
        
        Self {
            data: Cloud::new(input_stream, output_stream),
            extrinsic,
        }
    }

    pub fn act(&mut self) -> Result<(), LidarError> {
        // 创建一个前置处理函数，将点云从雷达坐标系转换到世界坐标系
        let extrinsic = self.extrinsic;
        let pre_process = move |lifra: Lifra| -> Lifra {
            // 获取点云数据并进行坐标变换
            let transformed_points: Vec<[f32; 3]> = lifra.points()
                .iter()
                .map(|point| {
                    // 将点转换为齐次坐标
                    let point_h = Vector4::new(point[0], point[1], point[2], 1.0);
                    // 应用外参矩阵进行坐标变换
                    let transformed = extrinsic * point_h;
                    // 转换回非齐次坐标
                    [transformed.x, transformed.y, transformed.z]
                })
                .collect();
            
            // 使用变换后的点创建新的Lifra实例
            Lifra::from_points(transformed_points)
        };
        
        // 调用带处理器的act方法
        self.data.act(pre_process)
    }
    
    /// 获取指定索引位置的帧数据并应用外参矩阵变换
    /// 
    /// # 参数
    /// * `index` - 要获取的帧数据的索引
    /// 
    /// # 返回值
    /// 如果指定索引处存在数据，则返回经过外参矩阵变换的点云数据，否则返回None
    pub fn get_at_with_transform(&self, index: usize) -> Option<Lifra> {
        // 锁定输入流
        let input_stream = self.data.cream().in_stream.lock().ok()?;
        
        // 使用get_at方法获取指定索引的数据
        let lifra = input_stream.get_at(index)?;
        
        // 应用外参矩阵变换
        let extrinsic = self.extrinsic;
        let transformed_points: Vec<[f32; 3]> = lifra.points()
            .iter()
            .map(|point| {
                // 将点转换为齐次坐标
                let point_h = Vector4::new(point[0], point[1], point[2], 1.0);
                // 应用外参矩阵进行坐标变换
                let transformed = extrinsic * point_h;
                // 转换回非齐次坐标
                [transformed.x, transformed.y, transformed.z]
            })
            .collect();
        
        // 使用变换后的点创建新的Lifra实例并返回
        Some(Lifra::from_points(transformed_points))
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