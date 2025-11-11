use std::sync::{Arc, Mutex};
use std::thread;
use image::DynamicImage;

use crate::color::{Bounds, core::Color};
use crate::utils::stream::Stream;

pub struct Perple {
    /// 公用数据流，由上级管理
    pub img_stream: Arc<Mutex<Stream<DynamicImage>>>,
    pub bounds_stream: Arc<Mutex<Stream<Bounds>>>,

    /// 内部模块私有数据
    color: Arc<Mutex<Color>>,
    color_thread: Option<thread::JoinHandle<()>>,
    color_running: Arc<Mutex<bool>>,
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
            color_thread: None,
            color_running: Arc::new(Mutex::new(false)),
        }
    }

    /// 启动color模块一次性动作
    pub fn act(&mut self) {
        let mut color = self.color.lock().unwrap();
        color.act();
    }
    
    /// 启动color模块的循环运行模式
    pub fn start_color_loop(&mut self) {
        let mut running = self.color_running.lock().unwrap();
        if !*running {
            *running = true;
            drop(running); // 释放锁
            
            let color_running = Arc::clone(&self.color_running);
            let color = Arc::clone(&self.color);
            
            self.color_thread = Some(thread::spawn(move || {
                while *color_running.lock().unwrap() {
                    {
                        let mut color = color.lock().unwrap();
                        color.act();
                    }
                    thread::sleep(std::time::Duration::from_millis(100)); // 控制处理频率
                }
            }));
        }
    }
    
    /// 停止color模块的循环运行模式
    pub fn stop_color_loop(&mut self) {
        let mut running = self.color_running.lock().unwrap();
        *running = false;
    }
    
    /// 检查color模块是否正在运行
    pub fn is_color_running(&self) -> bool {
        *self.color_running.lock().unwrap()
    }

    /// 更新图像流（推荐外部统一管理）
    pub fn update_image(&self, new_image: DynamicImage) {
        let mut img_stream = self.img_stream.lock().unwrap();
        let _ = img_stream.write(new_image);
    }
    
    /// 等待颜色处理线程结束
    pub fn join_color_thread(&mut self) {
        if let Some(handle) = self.color_thread.take() {
            let _ = handle.join();
        }
    }
}