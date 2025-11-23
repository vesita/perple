use std::sync::{Arc, Mutex};
use image::DynamicImage;


use crate::color::ImgBud;
use crate::lidar::bounds::LidBud;
use crate::lidar::lifra::Lifra;
use crate::utils::stream::Stream;

/// 系统数据交换中枢
/// 
/// Swapl作为整个系统的数据中枢，负责管理所有的数据流。
/// 其他模块通过访问Swapl来进行数据交互，实现了松耦合的架构设计。
/// 所有的数据流都是线程安全的(Arc<Mutex<Stream<T>>>)，可以在多个线程间安全共享。
pub struct Swapl {
    /// 点云数据输入流
    pub lidars: Arc<Mutex<Stream<Lifra>>>,
    /// 点云检测结果输出流
    pub lid_objs: Arc<Mutex<Stream<LidBud>>>,
    /// 图像数据输入流
    pub images: Arc<Mutex<Stream<DynamicImage>>>,
    /// 图像检测结果输出流
    pub img_objs: Arc<Mutex<Stream<ImgBud>>>,
}

impl Swapl { 
    /// 创建一个新的数据交换中枢
    pub fn new() -> Self {
        Swapl {
            lidars: Arc::new(Mutex::new(Stream::new())),
            lid_objs: Arc::new(Mutex::new(Stream::new())),
            images: Arc::new(Mutex::new(Stream::new())),
            img_objs: Arc::new(Mutex::new(Stream::new())),
        }
    }
    
    /// 获取图像输入流的引用
    pub fn get_images_stream(&self) -> Arc<Mutex<Stream<DynamicImage>>> {
        Arc::clone(&self.images)
    }
    
    /// 获取图像检测结果流的引用
    pub fn get_img_objs_stream(&self) -> Arc<Mutex<Stream<ImgBud>>> {
        Arc::clone(&self.img_objs)
    }
    
    /// 获取点云输入流的引用
    pub fn get_lidars_stream(&self) -> Arc<Mutex<Stream<Lifra>>> {
        Arc::clone(&self.lidars)
    }
    
    /// 获取点云检测结果流的引用
    pub fn get_lid_objs_stream(&self) -> Arc<Mutex<Stream<LidBud>>> {
        Arc::clone(&self.lid_objs)
    }
}