extern crate nalgebra as na;

use na::{Matrix4, Vector3};
use std::collections::HashMap;

use crate::cloud::CldBud;
use crate::color::ClrBud;
use crate::utils::stream::{Eap, Stream};
use image::DynamicImage;

/// World模块负责管理3D世界中的各种设备及其坐标变换
/// 符合项目中其他模块的数据交互设计模式，通过数据流进行通信
pub struct World {
    /// 设备列表
    equips: Vec<Box<dyn OnWorld>>,
    /// 设备ID映射
    equip_id: HashMap<String, usize>,
    /// 图像数据输入流
    image_input_stream: Option<Eap<Stream<DynamicImage>>>,
    /// 图像检测结果输入流
    image_result_stream: Option<Eap<Stream<ClrBud>>>,
    /// 点云数据输入流
    lidar_input_stream: Option<Eap<Stream<Vec<[f32; 3]>>>>,
    /// 点云检测结果输入流
    lidar_result_stream: Option<Eap<Stream<CldBud>>>,
}

/// World中设备的trait，定义了设备在世界坐标系中的行为
pub trait OnWorld {
    /// 获取设备的世界坐标变换矩阵
    fn on_world(&self) -> Matrix4<f32>;

    /// 通过角度设置设备姿态
    fn set_by_angle(&mut self, tra: Vector3<f32>, rot: Vector3<f32>);

    /// 通过弧度设置设备姿态
    fn set_by_radian(&mut self, tra: Vector3<f32>, rot: Vector3<f32>);

    /// 通过变换矩阵设置设备姿态
    fn set_by_matrix(&mut self, matrix: &Matrix4<f32>);

    // fn point_iter(&self) -> Box<dyn Iterator<Item = Vector3<f32>>>;
}

impl World {
    /// 创建一个新的World实例
    pub fn new() -> Self {
        Self {
            equips: vec![],
            equip_id: HashMap::new(),
            image_input_stream: None,
            image_result_stream: None,
            lidar_input_stream: None,
            lidar_result_stream: None,
        }
    }

    /// 设置图像数据输入流
    pub fn set_image_input_stream(&mut self, stream: Eap<Stream<DynamicImage>>) {
        self.image_input_stream = Some(stream);
    }

    /// 设置图像检测结果输入流
    pub fn set_image_result_stream(&mut self, stream: Eap<Stream<ClrBud>>) {
        self.image_result_stream = Some(stream);
    }

    /// 设置点云数据输入流
    pub fn set_lidar_input_stream(&mut self, stream: Eap<Stream<Vec<[f32; 3]>>>) {
        self.lidar_input_stream = Some(stream);
    }

    /// 设置点云检测结果输入流
    pub fn set_lidar_result_stream(&mut self, stream: Eap<Stream<CldBud>>) {
        self.lidar_result_stream = Some(stream);
    }

    /// 从数据流中读取图像数据
    pub async fn read_image_data(&self) -> Option<DynamicImage> {
        if let Some(stream) = &self.image_input_stream {
            return stream.lock().await.read();
        }
        None
    }

    /// 从数据流中读取图像检测结果
    pub async fn read_image_result(&self) -> Option<ClrBud> {
        if let Some(stream) = &self.image_result_stream {
            return stream.lock().await.read();
        }
        None
    }

    /// 从数据流中读取点云数据
    pub async fn read_lidar_data(&self) -> Option<Vec<[f32; 3]>> {
        if let Some(stream) = &self.lidar_input_stream {
            return stream.lock().await.read();
        }
        None
    }

    /// 从数据流中读取点云检测结果
    pub async fn read_lidar_result(&self) -> Option<CldBud> {
        if let Some(stream) = &self.lidar_result_stream {
            return stream.lock().await.read();
        }
        None
    }

    /// 添加设备到世界中
    pub fn add_equip(&mut self, name: String, equip: Box<dyn OnWorld>) -> usize {
        self.equips.push(equip);
        let index = self.equips.len() - 1;
        self.equip_id.insert(name, index);
        index
    }

    /// 根据名称获取设备
    pub fn get_equip(&self, name: &str) -> Option<&Box<dyn OnWorld>> {
        if let Some(&index) = self.equip_id.get(name) {
            self.equips.get(index)
        } else {
            None
        }
    }

    /// 根据名称获取可变设备引用
    pub fn get_equip_mut(&mut self, name: &str) -> Option<&mut Box<dyn OnWorld>> {
        if let Some(&index) = self.equip_id.get(name) {
            self.equips.get_mut(index)
        } else {
            None
        }
    }
}
