use std::fmt;
use std::sync::Arc;

use crate::cloud::core::Lidar;
use crate::color::core::Camera;

use crate::tracker::core::Tracker;
use crate::utils::muloop::{LoopMode, MultiLoop};
use crate::utils::stream::{Eap, new_eap};

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
    pub camera: Eap<Camera>,
    pub lidar: Eap<Lidar>,
    pub tracker: Eap<Tracker>,

    /// 控制类模块（可能跨线程访问，使用Eap<T>>）
    color_loop: Eap<MultiLoop>,
    lidar_loop: Eap<MultiLoop>,
    tracker_loop: Eap<MultiLoop>,
}

impl Perple {
    /// 创建Perple实例，通过全局Swapl数据中枢进行数据交互
    ///
    /// 所有数据交互都通过全局Swapl完成，实现了模块间的松耦合设计。
    /// Perple模块内部保留指向各模块的指针，但不再需要外部传入Swapl引用
    /// 模型路径从全局配置中获取
    pub fn new() -> Self {
        let camera = new_eap(Camera::new());

        let lidar = new_eap(Lidar::new());

        let tracker = new_eap(Tracker::new());

        Self {
            camera,
            lidar,
            tracker,
            color_loop: new_eap(MultiLoop::new()),
            lidar_loop: new_eap(MultiLoop::new()),
            tracker_loop: new_eap(MultiLoop::new()),
        }
    }

    pub async fn run(&mut self) -> Result<(), PerpleError> {
        self.color_loop
            .lock()
            .await
            .start_with_method(
                LoopMode::Signal,
                Arc::clone(&self.camera),
                |camera| {
                    let _ = camera.act();
                },
                40,
            )
            .await
            .map_err(|e| PerpleError::LoopError(e))?;
        self.lidar_loop
            .lock()
            .await
            .start_with_method(
                LoopMode::Signal,
                Arc::clone(&self.lidar),
                |lidar| {
                    let _ = lidar.act();
                },
                40,
            )
            .await
            .map_err(|e| PerpleError::LoopError(e))?;
        self.tracker_loop
            .lock()
            .await
            .start_with_method(
                LoopMode::Signal,
                Arc::clone(&self.tracker),
                |tracker| {
                    let _ = tracker.run();
                },
                40,
            )
            .await
            .map_err(|e| PerpleError::LoopError(e))?;
        Ok(())
    }

    /// 启动color模块的循环运行模式
    /// 支持按次数、按时间或信号控制循环
    pub async fn start_color_loop_with_mode(&mut self, mode: LoopMode) -> Result<(), PerpleError> {
        // 获取color_loop的锁并启动循环
        let mut color_loop = self.color_loop.lock().await;
        let camera_ref = Arc::clone(&self.camera);
        color_loop
            .start_with_method(
                mode,
                camera_ref,
                |camera| {
                    let _ = camera.act();
                },
                100,
            ) // 100ms间隔
            .await
            .map_err(|e| PerpleError::LoopError(e))
    }

    /// 启动color模块的循环运行模式（默认信号控制循环）
    pub async fn start_color_loop(&mut self) -> Result<(), PerpleError> {
        self.start_color_loop_with_mode(LoopMode::Signal).await
    }

    /// 启动指定次数的循环运行模式
    pub async fn start_color_loop_count(&mut self, count: usize) -> Result<(), PerpleError> {
        self.start_color_loop_with_mode(LoopMode::Count(count))
            .await
    }

    /// 启动指定时间的循环运行模式（毫秒）
    pub async fn start_color_loop_duration(&mut self, duration_ms: u64) -> Result<(), PerpleError> {
        self.start_color_loop_with_mode(LoopMode::Duration(duration_ms))
            .await
    }

    /// 停止color模块的循环运行模式
    pub async fn stop_color_loop(&mut self) -> Result<(), PerpleError> {
        let mut color_loop = self.color_loop.lock().await;
        let _ = color_loop.stop();
        Ok(())
    }

    /// 检查color模块是否正在运行
    pub async fn is_color_running(&self) -> Result<bool, PerpleError> {
        let color_loop = self.color_loop.lock().await;
        Ok(color_loop.is_running().await)
    }

    /// 启动lidar模块的循环运行模式
    /// 支持按次数、按时间或信号控制循环
    pub async fn start_lidar_loop_with_mode(&mut self, mode: LoopMode) -> Result<(), PerpleError> {
        let mut lidar_loop = self.lidar_loop.lock().await;
        let lidar_ref = Arc::clone(&self.lidar);
        lidar_loop
            .start_with_method(
                mode,
                lidar_ref,
                |lidar| {
                    let _ = lidar.act();
                },
                100,
            ) // 100ms间隔
            .await
            .map_err(|e| PerpleError::LoopError(e))
    }

    /// 启动lidar模块的循环运行模式（默认信号控制循环）
    pub async fn start_lidar_loop(&mut self) -> Result<(), PerpleError> {
        self.start_lidar_loop_with_mode(LoopMode::Signal).await
    }

    /// 启动指定次数的循环运行模式
    pub async fn start_lidar_loop_count(&mut self, count: usize) -> Result<(), PerpleError> {
        self.start_lidar_loop_with_mode(LoopMode::Count(count))
            .await
    }

    /// 启动指定时间的循环运行模式（毫秒）
    pub async fn start_lidar_loop_duration(&mut self, duration_ms: u64) -> Result<(), PerpleError> {
        self.start_lidar_loop_with_mode(LoopMode::Duration(duration_ms))
            .await
    }

    /// 停止lidar模块的循环运行模式
    pub async fn stop_lidar_loop(&mut self) -> Result<(), PerpleError> {
        let mut lidar_loop = self.lidar_loop.lock().await;
        let _ = lidar_loop.stop();
        Ok(())
    }

    /// 检查lidar模块是否正在运行
    pub async fn is_lidar_running(&self) -> Result<bool, PerpleError> {
        let lidar_loop = self.lidar_loop.lock().await;
        Ok(lidar_loop.is_running().await)
    }

    /// 启动tracker模块的循环运行模式
    /// 支持按次数、按时间或信号控制循环
    pub async fn start_tracker_loop_with_mode(
        &mut self,
        mode: LoopMode,
    ) -> Result<(), PerpleError> {
        let mut tracker_loop = self.tracker_loop.lock().await;
        let tracker_ref = Arc::clone(&self.tracker);
        tracker_loop
            .start_with_method(
                mode,
                tracker_ref,
                |tracker| {
                    let _ = tracker.run();
                },
                100,
            ) // 100ms间隔
            .await
            .map_err(|e| PerpleError::LoopError(e))
    }

    /// 启动tracker模块的循环运行模式（默认信号控制循环）
    pub async fn start_tracker_loop(&mut self) -> Result<(), PerpleError> {
        self.start_tracker_loop_with_mode(LoopMode::Signal).await
    }

    /// 启动指定次数的循环运行模式
    pub async fn start_tracker_loop_count(&mut self, count: usize) -> Result<(), PerpleError> {
        self.start_tracker_loop_with_mode(LoopMode::Count(count))
            .await
    }

    /// 启动指定时间的循环运行模式（毫秒）
    pub async fn start_tracker_loop_duration(
        &mut self,
        duration_ms: u64,
    ) -> Result<(), PerpleError> {
        self.start_tracker_loop_with_mode(LoopMode::Duration(duration_ms))
            .await
    }

    /// 停止tracker模块的循环运行模式
    pub async fn stop_tracker_loop(&mut self) -> Result<(), PerpleError> {
        let mut tracker_loop = self.tracker_loop.lock().await;
        let _= tracker_loop.stop();
        Ok(())
    }

    /// 检查tracker模块是否正在运行
    pub async fn is_tracker_running(&self) -> Result<bool, PerpleError> {
        let tracker_loop = self.tracker_loop.lock().await;
        Ok(tracker_loop.is_running().await)
    }

    /// 等待颜色处理线程结束
    pub async fn join_color_thread(&mut self) -> Result<(), PerpleError> {
        let mut color_loop = self.color_loop.lock().await;
        color_loop
            .join()
            .await
            .map_err(|e| PerpleError::LoopError(e))
    }
}
