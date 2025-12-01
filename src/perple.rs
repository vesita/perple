use std::sync::{Arc, Mutex};
use std::fmt;

use crate::color::core::Camera;
use crate::color::core::Color;
use crate::cloud::core::{Cloud, Lidar};
use crate::cloud::lifra::Lifra;
use crate::cloud::CldBud;
use crate::tracker::core::Tracker;
use crate::Swapl;
use crate::utils::stream::Stream;
use crate::utils::muloop::{MultiLoop, LoopMode};
use crate::utils::world::World;

/// Perple模块的错误类型
#[derive(Debug)]
pub enum PerpleError {
    /// 循环控制相关错误
    LoopError(String),
    /// 线程锁中毒错误
    PoisonError(String),
    /// 其他错误
    Other(String),
}

impl fmt::Display for PerpleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PerpleError::LoopError(e) => write!(f, "循环控制错误: {}", e),
            PerpleError::PoisonError(e) => write!(f, "线程锁中毒错误: {}", e),
            PerpleError::Other(e) => write!(f, "其他错误: {}", e),
        }
    }
}

impl std::error::Error for PerpleError {}

impl<T> From<std::sync::PoisonError<T>> for PerpleError {
    fn from(error: std::sync::PoisonError<T>) -> Self {
        PerpleError::PoisonError(format!("线程锁中毒: {:?}", error))
    }
}

/// Perple主处理模块
/// 
/// 该模块通过Swapl数据中枢与其他模块进行数据交互，

pub struct Perple {
    camera: Arc<Mutex<Camera>>,
    lidar: Arc<Mutex<Lidar>>,
    tracker: Arc<Mutex<Tracker>>,

    /// 控制类模块（可能跨线程访问，使用Arc<Mutex<T>>）
    color_loop: Arc<Mutex<MultiLoop>>,
    lidar_loop: Arc<Mutex<MultiLoop>>,
    tracker_loop: Arc<Mutex<MultiLoop>>,
}

impl Perple {
    /// 创建Perple实例，通过Swapl数据中枢进行数据交互
    /// 
    /// 所有数据交互都通过Swapl完成，实现了模块间的松耦合设计。
    /// Perple模块只需要持有Swapl的引用，即可访问所有需要的数据流。
    pub fn new(
        pool: Arc<Swapl>,
        model_path: &str,
        config_path: &str,
    ) -> Self {
        let camera = Arc::new(Mutex::new(Camera::new(
            Arc::clone(&pool.colors),
            Arc::clone(&pool.clr_objs),
            Arc::clone(&pool.sights),
            model_path,
            config_path,
        )));
        
        let lidar = Arc::new(Mutex::new(Lidar::new(
            Arc::clone(&pool.clouds), 
            Arc::clone(&pool.cld_objs)
        )));

        let tracker = Arc::new(Mutex::new(Tracker::new(
            Arc::clone(&pool.sights),
            Arc::clone(&pool.cld_objs),
            Arc::clone(&pool.targets),
        )));
        
        // 释放pool引用
        drop(pool);
        
        Self {
            camera,
            lidar,
            tracker,
            color_loop: Arc::new(Mutex::new(MultiLoop::new())),
            lidar_loop: Arc::new(Mutex::new(MultiLoop::new())),
            tracker_loop: Arc::new(Mutex::new(MultiLoop::new())),
        }
    }

    pub fn run(&mut self) -> Result<(), PerpleError> { 
        
        if let Ok(mut color) = self.color_loop.lock() {
            let _ = color.start_with_method(LoopMode::Signal, Arc::clone(&self.camera), |camera| {
                let _ = camera.act();
            }, 40).map_err(|e| PerpleError::LoopError(e))?;
        }
        if let Ok(mut lidar) = self.lidar_loop.lock() {
            let _ = lidar.start_with_method(LoopMode::Signal, Arc::clone(&self.lidar), |lidar| {
                let _ = lidar.act();
            }, 40).map_err(|e| PerpleError::LoopError(e))?;
        }
        if let Ok(mut tracker) = self.tracker_loop.lock() {
            let _ = tracker.start_with_method(LoopMode::Signal, Arc::clone(&self.tracker), |tracker| {
                let _ = tracker.run();
            }, 40).map_err(|e| PerpleError::LoopError(e))?;
        }
        Ok(())
    }

    /// 启动color模块的循环运行模式
    /// 支持按次数、按时间或信号控制循环
    pub fn start_color_loop_with_mode(&mut self, mode: LoopMode) -> Result<(), PerpleError> {
        // 获取color_loop的锁并启动循环
        let mut color_loop = self.color_loop.lock()?;
        let camera_ref = Arc::clone(&self.camera);
        color_loop.start_with_method(mode, camera_ref, |camera| {
            let _ = camera.act();
        }, 100) // 100ms间隔
        .map_err(|e| PerpleError::LoopError(e))
    }
    
    /// 启动color模块的循环运行模式（默认信号控制循环）
    pub fn start_color_loop(&mut self) -> Result<(), PerpleError> {
        self.start_color_loop_with_mode(LoopMode::Signal)
    }
    
    /// 启动指定次数的循环运行模式
    pub fn start_color_loop_count(&mut self, count: usize) -> Result<(), PerpleError> {
        self.start_color_loop_with_mode(LoopMode::Count(count))
    }
    
    /// 启动指定时间的循环运行模式（毫秒）
    pub fn start_color_loop_duration(&mut self, duration_ms: u64) -> Result<(), PerpleError> {
        self.start_color_loop_with_mode(LoopMode::Duration(duration_ms))
    }
    
    /// 停止color模块的循环运行模式
    pub fn stop_color_loop(&mut self) -> Result<(), PerpleError> {
        let mut color_loop = self.color_loop.lock()?;
        color_loop.stop();
        Ok(())
    }
    
    /// 检查color模块是否正在运行
    pub fn is_color_running(&self) -> Result<bool, PerpleError> {
        let color_loop = self.color_loop.lock()?;
        Ok(color_loop.is_running())
    }

    /// 启动lidar模块的循环运行模式
    /// 支持按次数、按时间或信号控制循环
    pub fn start_lidar_loop_with_mode(&mut self, mode: LoopMode) -> Result<(), PerpleError> {
        let mut lidar_loop = self.lidar_loop.lock()?;
        let lidar_ref = Arc::clone(&self.lidar);
        lidar_loop.start_with_method(mode, lidar_ref, |lidar| {
            let _ = lidar.act();
        }, 100) // 100ms间隔
        .map_err(|e| PerpleError::LoopError(e))
    }
    
    /// 启动lidar模块的循环运行模式（默认信号控制循环）
    pub fn start_lidar_loop(&mut self) -> Result<(), PerpleError> {
        self.start_lidar_loop_with_mode(LoopMode::Signal)
    }
    
    /// 启动指定次数的循环运行模式
    pub fn start_lidar_loop_count(&mut self, count: usize) -> Result<(), PerpleError> {
        self.start_lidar_loop_with_mode(LoopMode::Count(count))
    }
    
    /// 启动指定时间的循环运行模式（毫秒）
    pub fn start_lidar_loop_duration(&mut self, duration_ms: u64) -> Result<(), PerpleError> {
        self.start_lidar_loop_with_mode(LoopMode::Duration(duration_ms))
    }
    
    /// 停止lidar模块的循环运行模式
    pub fn stop_lidar_loop(&mut self) -> Result<(), PerpleError> {
        let mut lidar_loop = self.lidar_loop.lock()?;
        lidar_loop.stop();
        Ok(())
    }
    
    /// 检查lidar模块是否正在运行
    pub fn is_lidar_running(&self) -> Result<bool, PerpleError> {
        let lidar_loop = self.lidar_loop.lock()?;
        Ok(lidar_loop.is_running())
    }

    /// 启动tracker模块的循环运行模式
    /// 支持按次数、按时间或信号控制循环
    pub fn start_tracker_loop_with_mode(&mut self, mode: LoopMode) -> Result<(), PerpleError> {
        let mut tracker_loop = self.tracker_loop.lock()?;
        let tracker_ref = Arc::clone(&self.tracker);
        tracker_loop.start_with_method(mode, tracker_ref, |tracker| {
            let _ = tracker.run();
        }, 100) // 100ms间隔
        .map_err(|e| PerpleError::LoopError(e))
    }
    
    /// 启动tracker模块的循环运行模式（默认信号控制循环）
    pub fn start_tracker_loop(&mut self) -> Result<(), PerpleError> {
        self.start_tracker_loop_with_mode(LoopMode::Signal)
    }
    
    /// 启动指定次数的循环运行模式
    pub fn start_tracker_loop_count(&mut self, count: usize) -> Result<(), PerpleError> {
        self.start_tracker_loop_with_mode(LoopMode::Count(count))
    }
    
    /// 启动指定时间的循环运行模式（毫秒）
    pub fn start_tracker_loop_duration(&mut self, duration_ms: u64) -> Result<(), PerpleError> {
        self.start_tracker_loop_with_mode(LoopMode::Duration(duration_ms))
    }
    
    /// 停止tracker模块的循环运行模式
    pub fn stop_tracker_loop(&mut self) -> Result<(), PerpleError> {
        let mut tracker_loop = self.tracker_loop.lock()?;
        tracker_loop.stop();
        Ok(())
    }
    
    /// 检查tracker模块是否正在运行
    pub fn is_tracker_running(&self) -> Result<bool, PerpleError> {
        let tracker_loop = self.tracker_loop.lock()?;
        Ok(tracker_loop.is_running())
    }

    
    /// 等待颜色处理线程结束
    pub fn join_color_thread(&mut self) -> Result<(), PerpleError> {
        let mut color_loop = self.color_loop.lock()?;
        color_loop.join().map_err(|e| PerpleError::LoopError(e))
    }
}