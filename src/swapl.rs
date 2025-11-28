use std::sync::{Arc, Mutex};

use image::DynamicImage;

use nalgebra::Vector3;

use crate::cloud::lifra::Lifra;
use crate::cloud::CldBud;
use crate::color::ClrBud;
use crate::tracker::target::Target;
use crate::utils::sight::Sight;
use crate::utils::stream::Stream;

/// 系统数据交换中枢
/// 
/// Swapl作为整个系统的数据中枢，负责管理所有的数据流。
/// 其他模块通过访问Swapl来进行数据交互，实现了松耦合的架构设计。
/// 所有的数据流都是线程安全的(Arc<Mutex<Stream<T>>>)，可以在多个线程间安全共享。
pub struct Swapl {
    /// 点云数据输入流
    pub clouds: Arc<Mutex<Stream<Lifra>>>,
    /// 点云检测结果输出流
    pub cld_objs: Arc<Mutex<Stream<Vec<CldBud>>>>,
    /// 图像数据输入流
    pub colors: Arc<Mutex<Stream<DynamicImage>>>,
    /// 图像检测结果输出流
    pub clr_objs: Arc<Mutex<Stream<Vec<ClrBud>>>>,
    /// 3D投影结果输出流
    pub sights: Arc<Mutex<Stream<Vec<Sight>>>>,
    /// 目标检测结果输出流
    pub targets: Arc<Mutex<Stream<Vec<Target>>>>,
}

impl Swapl { 
    /// 创建一个新的数据交换中枢
    pub fn new() -> Self {
        Swapl {
            clouds: Arc::new(Mutex::new(Stream::new())),
            cld_objs: Arc::new(Mutex::new(Stream::new())),
            colors: Arc::new(Mutex::new(Stream::new())),
            clr_objs: Arc::new(Mutex::new(Stream::new())),
            sights: Arc::new(Mutex::new(Stream::new())),
            targets: Arc::new(Mutex::new(Stream::new())),
        }
    }
    
    
}