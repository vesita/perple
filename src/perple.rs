use std::fmt;
use std::sync::Arc;

use crate::cloud::core::Lidar;
use crate::color::core::Camera;
use crate::fuse::Fuse;

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
    pub fuse: Eap<Fuse>,

    /// 控制类模块（可能跨线程访问，使用Eap<T>>）
    color_loop: Eap<MultiLoop>,
    lidar_loop: Eap<MultiLoop>,
    tracker_loop: Eap<MultiLoop>,
    fuse_loop: Eap<MultiLoop>,
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

        let fuse = new_eap(Fuse::new());

        Self {
            camera,
            lidar,
            tracker,
            fuse,
            color_loop: new_eap(MultiLoop::new()),
            lidar_loop: new_eap(MultiLoop::new()),
            tracker_loop: new_eap(MultiLoop::new()),
            fuse_loop: new_eap(MultiLoop::new()),
        }
    }

    pub async fn run(&mut self) -> Result<(), PerpleError> {
        self.color_loop
            .lock()
            .unwrap()
            .start_with_async_method(
                LoopMode::Signal,
                Arc::clone(&self.camera),
                |camera| async move {
                    let mut cam = camera.lock().unwrap();
                    let _ = cam.act();
                },
                40,
            )
            .await
            .map_err(|e| PerpleError::LoopError(e))?;
        self.lidar_loop
            .lock()
            .unwrap()
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
        self.fuse_loop
            .lock()
            .unwrap()
            .start_with_method(
                LoopMode::Signal,
                Arc::clone(&self.fuse),
                |fuse| {
                    let _ = fuse.act();
                },
                40,
            )
            .await
            .map_err(|e| PerpleError::LoopError(e))?;
        self.tracker_loop
            .lock()
            .unwrap()
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
        let mut color_loop = self.color_loop.lock().unwrap();
        let camera_ref = Arc::clone(&self.camera);
        color_loop
            .start_with_async_method(
                mode,
                camera_ref,
                |camera| async move {
                    let mut cam = camera.lock().unwrap();
                    let _ = cam.act();
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
        let mut color_loop = self.color_loop.lock().unwrap();
        color_loop.stop().await;
        Ok(())
    }

    /// 检查color模块是否正在运行
    pub async fn is_color_running(&self) -> Result<bool, PerpleError> {
        let color_loop = self.color_loop.lock().unwrap();
        Ok(color_loop.is_running().await)
    }

    /// 启动lidar模块的循环运行模式
    /// 支持按次数、按时间或信号控制循环
    pub async fn start_lidar_loop_with_mode(&mut self, mode: LoopMode) -> Result<(), PerpleError> {
        let mut lidar_loop = self.lidar_loop.lock().unwrap();
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
        let mut lidar_loop = self.lidar_loop.lock().unwrap();
        lidar_loop.stop().await;
        Ok(())
    }

    /// 检查lidar模块是否正在运行
    pub async fn is_lidar_running(&self) -> Result<bool, PerpleError> {
        let lidar_loop = self.lidar_loop.lock().unwrap();
        Ok(lidar_loop.is_running().await)
    }

    /// 启动tracker模块的循环运行模式
    /// 支持按次数、按时间或信号控制循环
    pub async fn start_tracker_loop_with_mode(
        &mut self,
        mode: LoopMode,
    ) -> Result<(), PerpleError> {
        let mut tracker_loop = self.tracker_loop.lock().unwrap();
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
        let mut tracker_loop = self.tracker_loop.lock().unwrap();
        tracker_loop.stop().await;
        Ok(())
    }

    /// 检查tracker模块是否正在运行
    pub async fn is_tracker_running(&self) -> Result<bool, PerpleError> {
        let tracker_loop = self.tracker_loop.lock().unwrap();
        Ok(tracker_loop.is_running().await)
    }

    /// 启动fuse模块的循环运行模式
    pub async fn start_fuse_loop_with_mode(
        &mut self,
        mode: LoopMode,
    ) -> Result<(), PerpleError> {
        let mut fuse_loop = self.fuse_loop.lock().unwrap();
        let fuse_ref = Arc::clone(&self.fuse);
        fuse_loop
            .start_with_method(
                mode,
                fuse_ref,
                |fuse| {
                    let _ = fuse.act();
                },
                40,
            )
            .await
            .map_err(|e| PerpleError::LoopError(e))
    }

    /// 启动fuse模块的循环运行模式（默认信号控制循环）
    pub async fn start_fuse_loop(&mut self) -> Result<(), PerpleError> {
        self.start_fuse_loop_with_mode(LoopMode::Signal).await
    }

    /// 停止fuse模块的循环运行模式
    pub async fn stop_fuse_loop(&mut self) -> Result<(), PerpleError> {
        let mut fuse_loop = self.fuse_loop.lock().unwrap();
        fuse_loop.stop().await;
        Ok(())
    }

    /// 检查fuse模块是否正在运行
    pub async fn is_fuse_running(&self) -> Result<bool, PerpleError> {
        let fuse_loop = self.fuse_loop.lock().unwrap();
        Ok(fuse_loop.is_running().await)
    }

    /// 等待颜色处理线程结束
   pub async fn join_color_thread(&mut self) -> Result<(), PerpleError> {
       let mut color_loop = self.color_loop.lock().unwrap();
        color_loop
            .join()
            .await
            .map_err(|e| PerpleError::LoopError(e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::{sleep, Duration};

    /// 测试 Perple 实例的创建
    #[test]
    fn test_perple_new() {
       let perple = Perple::new();
        
        // 验证所有字段都已初始化
        assert!(perple.camera.try_lock().is_ok());
        assert!(perple.lidar.try_lock().is_ok());
        assert!(perple.tracker.try_lock().is_ok());
    }

    /// 测试 run 方法能否成功启动所有循环
    #[tokio::test]
    async fn test_run_starts_all_loops() {
       let mut perple = Perple::new();
        
        // 调用 run 方法
       let result = perple.run().await;
        
        // 验证 run 成功返回
        assert!(result.is_ok(), "run() 应该成功返回");
        
        // 验证所有循环都已启动
        assert!(perple.is_color_running().await.unwrap_or(false), "color 循环应该已启动");
        assert!(perple.is_lidar_running().await.unwrap_or(false), "lidar 循环应该已启动");
        assert!(perple.is_tracker_running().await.unwrap_or(false), "tracker 循环应该已启动");
        
        // 清理：停止所有循环
       let _ = perple.stop_color_loop().await;
       let _ = perple.stop_lidar_loop().await;
       let _ = perple.stop_tracker_loop().await;
    }

    /// 测试 run 方法启动后循环能正常执行
    #[tokio::test]
    async fn test_run_loops_execute() {
       let mut perple = Perple::new();
        
        // 启动所有循环
       let run_result = perple.run().await;
        assert!(run_result.is_ok());
        
        // 等待一小段时间，让循环有机会执行
        sleep(Duration::from_millis(100)).await;
        
        // 验证循环仍在运行
        assert!(perple.is_color_running().await.unwrap_or(false));
        assert!(perple.is_lidar_running().await.unwrap_or(false));
        assert!(perple.is_tracker_running().await.unwrap_or(false));
        
        // 清理：停止所有循环
       let _ = perple.stop_color_loop().await;
       let _ = perple.stop_lidar_loop().await;
       let _ = perple.stop_tracker_loop().await;
        
        // 等待一小段时间确保循环已停止
        sleep(Duration::from_millis(50)).await;
        
        // 验证循环已停止
        assert!(!perple.is_color_running().await.unwrap_or(false));
        assert!(!perple.is_lidar_running().await.unwrap_or(false));
        assert!(!perple.is_tracker_running().await.unwrap_or(false));
    }

    /// 测试多次调用 run 方法的错误处理
    #[tokio::test]
    async fn test_run_error_on_already_running() {
       let mut perple = Perple::new();
        
        // 第一次调用 run
       let result1 = perple.run().await;
        assert!(result1.is_ok());
        
        // 第二次调用 run 应该失败
       let result2 = perple.run().await;
        assert!(result2.is_err(), "第二次调用 run 应该返回错误");
        
        // 清理
       let _ = perple.stop_color_loop().await;
       let _ = perple.stop_lidar_loop().await;
       let _ = perple.stop_tracker_loop().await;
    }

    /// 测试单独启动各个循环的功能
    #[tokio::test]
    async fn test_individual_loop_start() {
       let mut perple = Perple::new();
        
        // 只启动 color 循环
       let color_result = perple.start_color_loop().await;
        assert!(color_result.is_ok());
        assert!(perple.is_color_running().await.unwrap_or(false));
        
        // 只启动 lidar 循环
       let lidar_result = perple.start_lidar_loop().await;
        assert!(lidar_result.is_ok());
        assert!(perple.is_lidar_running().await.unwrap_or(false));
        
        // 只启动 tracker 循环
       let tracker_result = perple.start_tracker_loop().await;
        assert!(tracker_result.is_ok());
        assert!(perple.is_tracker_running().await.unwrap_or(false));
        
        // 清理
       let _ = perple.stop_color_loop().await;
       let _ = perple.stop_lidar_loop().await;
       let _ = perple.stop_tracker_loop().await;
    }

    /// 测试按次数循环的模式
    #[tokio::test]
    async fn test_count_mode_loop() {
       let mut perple = Perple::new();
        
        // 启动一个只执行 3 次的循环
       let result = perple.start_color_loop_count(3).await;
        assert!(result.is_ok());
        
        // 等待循环完成（3 次 * 100ms = 300ms，加上一些余量）
        sleep(Duration::from_millis(500)).await;
        
        // 验证循环已自动停止
        assert!(!perple.is_color_running().await.unwrap_or(false));
    }

    /// 测试按时长循环的模式
    #[tokio::test]
    async fn test_duration_mode_loop() {
       let mut perple = Perple::new();
        
        // 启动一个运行 200ms 的循环
       let result = perple.start_color_loop_duration(200).await;
        assert!(result.is_ok());
        
        // 立即检查，应该还在运行
        assert!(perple.is_color_running().await.unwrap_or(false));
        
        // 等待超过指定时长
        sleep(Duration::from_millis(300)).await;
        
        // 验证循环已自动停止
        assert!(!perple.is_color_running().await.unwrap_or(false));
    }

    /// 测试信号控制的循环模式
    #[tokio::test]
    async fn test_signal_mode_loop() {
       let mut perple = Perple::new();
        
        // 启动信号控制的循环
       let result = perple.start_color_loop().await;
        assert!(result.is_ok());
        
        // 验证正在运行
        assert!(perple.is_color_running().await.unwrap_or(false));
        
        // 手动停止
       let stop_result = perple.stop_color_loop().await;
        assert!(stop_result.is_ok());
        
        // 等待一小段时间
        sleep(Duration::from_millis(50)).await;
        
        // 验证已停止
        assert!(!perple.is_color_running().await.unwrap_or(false));
    }

    /// 测试并发访问安全性
    #[tokio::test]
    async fn test_concurrent_access() {
       let mut perple = Perple::new();
        
        // 启动循环
       let run_result = perple.run().await;
        assert!(run_result.is_ok());
        
        // 并发检查多个循环的状态
       let camera_check = perple.camera.lock().unwrap();
       let lidar_check = perple.lidar.lock().unwrap();
       let tracker_check = perple.tracker.lock().unwrap();
        
        // 验证都能成功获取锁（没有死锁）
        drop(camera_check);
        drop(lidar_check);
        drop(tracker_check);
        
        // 清理
       let _ = perple.stop_color_loop().await;
       let _ = perple.stop_lidar_loop().await;
       let _ = perple.stop_tracker_loop().await;
    }
}
