use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;
use image::DynamicImage;

use crate::color::core::Camera;
use crate::color::{ClrBud, core::Color};
use crate::cloud::core::{Cloud, Lidar};
use crate::cloud::lifra::Lifra;
use crate::cloud::CldBud;
use crate::utils::swapl::Swapl;
use crate::utils::stream::Stream;
use crate::utils::muloop::{MultiLoop, LoopMode};
use crate::utils::world::World;
use pcd_rs::DynRecord;

/// Perple主处理模块
/// 
/// 该模块通过Swapl数据中枢与其他模块进行数据交互，

pub struct Perple {
    /// 图像数据流（从Swapl数据中枢获取）
    pub clr_stream: Arc<Mutex<Stream<DynamicImage>>>,
    /// 图像检测结果流（从Swapl数据中枢获取）
    pub clr_bud_stream: Arc<Mutex<Stream<Vec<ClrBud>>>>,
    /// 点云数据流（从Swapl数据中枢获取）
    pub cld_stream: Arc<Mutex<Stream<Lifra>>>,
    /// 点云检测结果流（从Swapl数据中枢获取）
    pub cld_bud_stream: Arc<Mutex<Stream<Vec<CldBud>>>>,

    camera: Arc<Mutex<Camera>>,
    lidar: Arc<Mutex<Lidar>>,

    /// 控制类模块（可能跨线程访问，使用Arc<Mutex<T>>）
    color_loop: Arc<Mutex<MultiLoop>>,
    lidar_loop: Arc<Mutex<MultiLoop>>,
}

impl Perple {
    /// 创建Perple实例，通过Swapl数据中枢进行数据交互
    /// 
    /// 所有数据交互都通过Swapl完成，实现了模块间的松耦合设计。
    /// Perple模块只需要持有Swapl的引用，即可访问所有需要的数据流。
    pub fn new(
        pool: &Swapl,
        model_path: &str,
    ) -> Self {
        // 从Swapl数据中枢获取共享数据流引用
        let img_stream = pool.get_images_stream();
        let img_bud_stream = pool.get_img_objs_stream();
        let lid_stream = pool.get_lidars_stream();
        let lid_bud_stream = pool.get_lid_objs_stream();
        

        let camera = Arc::new(Mutex::new(Camera::new(
            Arc::clone(&img_stream),
            Arc::clone(&img_bud_stream),
            model_path,
        )));
        

        let lidar = Arc::new(Mutex::new(Lidar::new(
            Arc::clone(&lid_stream), 
            Arc::clone(&lid_bud_stream)
        )));
        
        Self {
            // 公用数据流，通过Swapl数据中枢进行访问
            clr_stream: img_stream,
            clr_bud_stream: img_bud_stream,
            cld_stream: lid_stream,
            cld_bud_stream: lid_bud_stream,
            camera,
            lidar,
            color_loop: Arc::new(Mutex::new(MultiLoop::new())),
            lidar_loop: Arc::new(Mutex::new(MultiLoop::new())),
        }
    }

    pub fn run(&mut self) -> Result<(), String> { 
        
        if let Ok(mut color) = self.color_loop.lock() {
            let _ = color.start_with_method(LoopMode::Signal, Arc::clone(&self.camera), |camera| {
                camera.act();
            }, 40);
        }
        if let Ok(mut lidar) = self.lidar_loop.lock() {
            let _ = lidar.start_with_method(LoopMode::Signal, Arc::clone(&self.lidar), |lidar| {
                lidar.act();
            }, 40);
        }
        Ok(())
    }

    /// 启动color模块的循环运行模式
    /// 支持按次数、按时间或信号控制循环
    pub fn start_color_loop_with_mode(&mut self, mode: LoopMode) -> Result<(), String> {
        // 获取color_loop的锁并启动循环
        let mut color_loop = self.color_loop.lock().unwrap();
        let camera_ref = Arc::clone(&self.camera);
        color_loop.start_with_method(mode, camera_ref, |camera| {
            camera.act();
        }, 100) // 100ms间隔
    }
    
    /// 启动color模块的循环运行模式（默认信号控制循环）
    pub fn start_color_loop(&mut self) -> Result<(), String> {
        self.start_color_loop_with_mode(LoopMode::Signal)
    }
    
    /// 启动指定次数的循环运行模式
    pub fn start_color_loop_count(&mut self, count: usize) -> Result<(), String> {
        self.start_color_loop_with_mode(LoopMode::Count(count))
    }
    
    /// 启动指定时间的循环运行模式（毫秒）
    pub fn start_color_loop_duration(&mut self, duration_ms: u64) -> Result<(), String> {
        self.start_color_loop_with_mode(LoopMode::Duration(duration_ms))
    }
    
    /// 停止color模块的循环运行模式
    pub fn stop_color_loop(&mut self) {
        let mut color_loop = self.color_loop.lock().unwrap();
        color_loop.stop();
    }
    
    /// 检查color模块是否正在运行
    pub fn is_color_running(&self) -> bool {
        let color_loop = self.color_loop.lock().unwrap();
        color_loop.is_running()
    }

    /// 启动lidar模块的循环运行模式
    /// 支持按次数、按时间或信号控制循环
    pub fn start_lidar_loop_with_mode(&mut self, mode: LoopMode) -> Result<(), String> {
        let mut lidar_loop = self.lidar_loop.lock().unwrap();
        let lidar_ref = Arc::clone(&self.lidar);
        lidar_loop.start_with_method(mode, lidar_ref, |lidar| {
            lidar.act();
        }, 100) // 100ms间隔
    }
    
    /// 启动lidar模块的循环运行模式（默认信号控制循环）
    pub fn start_lidar_loop(&mut self) -> Result<(), String> {
        self.start_lidar_loop_with_mode(LoopMode::Signal)
    }
    
    /// 启动指定次数的循环运行模式
    pub fn start_lidar_loop_count(&mut self, count: usize) -> Result<(), String> {
        self.start_lidar_loop_with_mode(LoopMode::Count(count))
    }
    
    /// 启动指定时间的循环运行模式（毫秒）
    pub fn start_lidar_loop_duration(&mut self, duration_ms: u64) -> Result<(), String> {
        self.start_lidar_loop_with_mode(LoopMode::Duration(duration_ms))
    }
    
    /// 停止lidar模块的循环运行模式
    pub fn stop_lidar_loop(&mut self) {
        let mut lidar_loop = self.lidar_loop.lock().unwrap();
        lidar_loop.stop();
    }
    
    /// 检查lidar模块是否正在运行
    pub fn is_lidar_running(&self) -> bool {
        let lidar_loop = self.lidar_loop.lock().unwrap();
        lidar_loop.is_running()
    }

    /// 更新图像流（推荐外部统一管理）
    pub fn update_image(&self, new_image: DynamicImage) {
        let mut img_stream = self.clr_stream.lock().unwrap();
        let _ = img_stream.write(new_image);
    }
    
    /// 等待颜色处理线程结束
    pub fn join_color_thread(&mut self) -> Result<(), String> {
        let mut color_loop = self.color_loop.lock().unwrap();
        color_loop.join()
    }
    
    /// 等待直到有检测结果可用
    pub fn wait_for_result(&self, timeout_ms: u64) -> bool {
        let start = std::time::Instant::now();
        while start.elapsed().as_millis() < timeout_ms as u128 {
            {
                let bounds_stream = self.clr_bud_stream.lock().unwrap();
                if bounds_stream.has_data() {
                    return true;
                }
            }
            thread::sleep(Duration::from_millis(10));
        }
        false
    }
}