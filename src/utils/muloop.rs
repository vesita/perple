use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

/// 循环模式枚举
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LoopMode {
    /// 按次数循环
    Count(usize),
    /// 按时间循环（毫秒）
    Duration(u64),
    /// 持续循环直到手动停止
    Continuous,
}

/// 循环控制结构体
pub struct MultiLoop {
    running: Arc<Mutex<bool>>,
    thread_handle: Option<thread::JoinHandle<()>>,
}

impl MultiLoop {
    /// 创建新的MultiLoop实例
    pub fn new() -> Self {
        Self {
            running: Arc::new(Mutex::new(false)),
            thread_handle: None,
        }
    }
    
    /// 启动循环
    /// 
    /// # 参数
    /// * `mode` - 循环模式
    /// * `callback` - 每次循环执行的回调函数
    /// * `interval_ms` - 每次循环之间的间隔（毫秒）
    pub fn start<F>(&mut self, mode: LoopMode, mut callback: F, interval_ms: u64) -> Result<(), String> 
    where
        F: FnMut() + Send + 'static,
    {
        let mut running = self.running.lock().unwrap();
        if *running {
            return Err("Loop is already running".to_string());
        }
        
        *running = true;
        drop(running); // 释放锁
        
        let loop_running = Arc::clone(&self.running);
        
        self.thread_handle = Some(thread::spawn(move || {
            match mode {
                LoopMode::Count(count) => {
                    let mut counter = 0;
                    while *loop_running.lock().unwrap() && counter < count {
                        callback();
                        counter += 1;
                        // 控制处理频率
                        thread::sleep(Duration::from_millis(interval_ms));
                    }
                    // 循环结束后自动停止
                    let mut running = loop_running.lock().unwrap();
                    *running = false;
                },
                LoopMode::Duration(duration_ms) => {
                    let start_time = std::time::Instant::now();
                    while *loop_running.lock().unwrap() && start_time.elapsed().as_millis() < duration_ms as u128 {
                        callback();
                        // 控制处理频率
                        thread::sleep(Duration::from_millis(interval_ms));
                    }
                    // 时间结束后自动停止
                    let mut running = loop_running.lock().unwrap();
                    *running = false;
                },
                LoopMode::Continuous => {
                    while *loop_running.lock().unwrap() {
                        callback();
                        // 控制处理频率
                        thread::sleep(Duration::from_millis(interval_ms));
                    }
                }
            }
        }));
        
        Ok(())
    }
    
    /// 停止循环
    pub fn stop(&mut self) {
        let mut running = self.running.lock().unwrap();
        *running = false;
    }
    
    /// 检查循环是否正在运行
    pub fn is_running(&self) -> bool {
        *self.running.lock().unwrap()
    }
    
    /// 等待线程结束
    pub fn join(&mut self) -> Result<(), String> {
        if let Some(handle) = self.thread_handle.take() {
            handle.join().map_err(|_| "Failed to join thread".to_string())?;
        }
        Ok(())
    }
}

impl Default for MultiLoop {
    fn default() -> Self {
        Self::new()
    }
}