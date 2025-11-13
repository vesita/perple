use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;
use image::DynamicImage;

use crate::color::{Bounds, core::Color};
use crate::utils::stream::Stream;
use crate::utils::muloop::{MultiLoop, LoopMode};

pub struct Perple {
    /// 公用数据流，由上级管理
    pub img_stream: Arc<Mutex<Stream<DynamicImage>>>,
    pub bounds_stream: Arc<Mutex<Stream<Bounds>>>,

    /// 内部模块私有数据
    color: Arc<Mutex<Color>>,
    color_loop: MultiLoop,
}

impl Perple {
    /// 推荐由外部传入公用数据流，减少拷贝和耦合
    pub fn new(
        img_stream: Arc<Mutex<Stream<DynamicImage>>>,
        bounds_stream: Arc<Mutex<Stream<Bounds>>>,
        model_path: &str,
    ) -> Self {
        let color = Color::new(
            Arc::clone(&img_stream),
            Arc::clone(&bounds_stream),
            model_path,
        );
        
        Self {
            img_stream,
            bounds_stream,
            color: Arc::new(Mutex::new(color)),
            color_loop: MultiLoop::new(),
        }
    }

    /// 启动color模块的循环运行模式
    /// 支持按次数、按时间或持续循环
    pub fn start_color_loop_with_mode(&mut self, mode: LoopMode) -> Result<(), String> {
        // 创建闭包，捕获color的引用
        let color = Arc::clone(&self.color);
        self.color_loop.start(mode, move || {
            let mut color_guard = color.lock().unwrap();
            color_guard.act();
        }, 100) // 100ms间隔
    }
    
    /// 启动color模块的循环运行模式（默认持续循环）
    pub fn start_color_loop(&mut self) -> Result<(), String> {
        self.start_color_loop_with_mode(LoopMode::Continuous)
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
        self.color_loop.stop();
    }
    
    /// 检查color模块是否正在运行
    pub fn is_color_running(&self) -> bool {
        self.color_loop.is_running()
    }

    /// 更新图像流（推荐外部统一管理）
    pub fn update_image(&self, new_image: DynamicImage) {
        let mut img_stream = self.img_stream.lock().unwrap();
        let _ = img_stream.write(new_image);
    }
    
    /// 等待颜色处理线程结束
    pub fn join_color_thread(&mut self) -> Result<(), String> {
        self.color_loop.join()
    }
    
    /// 等待直到有检测结果可用
    pub fn wait_for_result(&self, timeout_ms: u64) -> bool {
        let start = std::time::Instant::now();
        while start.elapsed().as_millis() < timeout_ms as u128 {
            {
                let bounds_stream = self.bounds_stream.lock().unwrap();
                if bounds_stream.has_data() {
                    return true;
                }
            }
            thread::sleep(Duration::from_millis(10));
        }
        false
    }
}