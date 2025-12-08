use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use crate::utils::stream::Eap;

/// 循环模式枚举，用于指定循环的不同执行方式
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LoopMode {
    /// 按指定次数循环执行
    /// 
    /// # 泛型参数
    /// * `usize` - 循环执行的次数
    Count(usize),
    
    /// 按指定时长循环执行（毫秒）
    /// 
    /// # 泛型参数
    /// * `u64` - 循环执行的时长（毫秒）
    Duration(u64),
    
    /// 基于信号控制的循环模式，需要手动停止
    Signal,
}

/// 多线程循环控制器
/// 
/// 该结构体允许在单独的线程中执行循环任务，
/// 支持多种循环模式（按次数、按时长、信号控制），
/// 并提供开始、停止和等待完成等功能。
pub struct MultiLoop {
    /// 循环运行状态标志，使用互斥锁保证线程安全
    running: Eap<bool>,
    
    /// 循环执行线程句柄，用于等待线程完成
    thread_handle: Option<thread::JoinHandle<()>>,
}

impl MultiLoop {
    /// 创建一个新的 MultiLoop 实例
    /// 
    /// # 返回值
    /// 返回初始化后的 MultiLoop 实例，默认处于停止状态
    pub fn new() -> Self {
        Self {
            running: Arc::new(Mutex::new(false)),
            thread_handle: None,
        }
    }
    
    /// 启动循环执行任务
    /// 
    /// # 参数
    /// * `mode` - 循环模式，指定循环的执行方式
    /// * `callback` - 每次循环执行的回调函数
    /// * `interval_ms` - 每次循环之间的间隔时间（毫秒）
    /// 
    /// # 返回值
    /// 成功启动返回 Ok(())，如果循环已在运行则返回 Err 错误信息
    /// 
    /// # 泛型参数
    /// * `F` - 回调函数类型，必须实现 FnMut() + Send + 'static 特征
    pub fn start<F>(&mut self, mode: LoopMode, mut callback: F, interval_ms: u64) -> Result<(), String> 
    where
        F: FnMut() + Send + 'static,
    {
        let mut running = self.running.lock().unwrap();
        if *running {
            return Err("循环已在运行中".to_string());
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
                        // 控制处理频率，防止占用过多CPU资源
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
                        // 控制处理频率，防止占用过多CPU资源
                        thread::sleep(Duration::from_millis(interval_ms));
                    }
                    // 时间结束后自动停止
                    let mut running = loop_running.lock().unwrap();
                    *running = false;
                },
                LoopMode::Signal => {
                    while *loop_running.lock().unwrap() {
                        callback();
                        // 控制处理频率，防止占用过多CPU资源
                        thread::sleep(Duration::from_millis(interval_ms));
                    }
                }
            }
        }));
        
        Ok(())
    }
    
    /// 启动针对特定对象方法的循环执行
    /// 
    /// # 参数
    /// * `mode` - 循环模式，指定循环的执行方式
    /// * `object` - 需要对其方法进行循环调用的对象引用
    /// * `method` - 要执行的方法（通常是一个闭包，调用对象的方法）
    /// * `interval_ms` - 每次循环之间的间隔时间（毫秒）
    /// 
    /// # 返回值
    /// 成功启动返回 Ok(())，如果循环已在运行则返回 Err 错误信息
    /// 
    /// # 泛型参数
    /// * `T` - 对象类型，必须实现 Send + 'static 特征
    /// * `F` - 方法类型，必须实现 Fn(&mut T) + Send + 'static 特征
    pub fn start_with_method<T, F>(
        &mut self,
        mode: LoopMode,
        object: Eap<T>,
        method: F,
        interval_ms: u64,
    ) -> Result<(), String>
    where
        T: Send + 'static,
        F: Fn(&mut T) + Send + 'static,
    {
        let mut running = self.running.lock().unwrap();
        if *running {
            return Err("循环已在运行中".to_string());
        }

        *running = true;
        drop(running); // 释放锁

        let loop_running = Arc::clone(&self.running);

        self.thread_handle = Some(thread::spawn(move || {
            match mode {
                LoopMode::Count(count) => {
                    let mut counter = 0;
                    while *loop_running.lock().unwrap() && counter < count {
                        {
                            let mut obj = object.lock().unwrap();
                            method(&mut *obj);
                        }
                        counter += 1;
                        thread::sleep(Duration::from_millis(interval_ms));
                    }
                    let mut running = loop_running.lock().unwrap();
                    *running = false;
                }
                LoopMode::Duration(duration_ms) => {
                    let start_time = std::time::Instant::now();
                    while *loop_running.lock().unwrap()
                        && start_time.elapsed().as_millis() < duration_ms as u128
                    {
                        {
                            let mut obj = object.lock().unwrap();
                            method(&mut *obj);
                        }
                        thread::sleep(Duration::from_millis(interval_ms));
                    }
                    let mut running = loop_running.lock().unwrap();
                    *running = false;
                }
                LoopMode::Signal => {
                    while *loop_running.lock().unwrap() {
                        {
                            let mut obj = object.lock().unwrap();
                            method(&mut *obj);
                        }
                        thread::sleep(Duration::from_millis(interval_ms));
                    }
                }
            }
        }));

        Ok(())
    }
    
    /// 停止循环执行
    /// 
    /// 将运行状态设置为 false，使循环在下一次迭代时退出
    pub fn stop(&mut self) {
        let mut running = self.running.lock().unwrap();
        *running = false;
    }
    
    /// 检查循环是否正在运行
    /// 
    /// # 返回值
    /// 如果循环正在运行返回 true，否则返回 false
    pub fn is_running(&self) -> bool {
        *self.running.lock().unwrap()
    }
    
    /// 等待循环线程执行完成
    /// 
    /// # 返回值
    /// 线程成功完成返回 Ok(())，否则返回 Err 错误信息
    pub fn join(&mut self) -> Result<(), String> {
        if let Some(handle) = self.thread_handle.take() {
            handle.join().map_err(|_| "等待线程完成时发生错误".to_string())?;
        }
        Ok(())
    }
}

impl Default for MultiLoop {
    /// 默认实现，创建一个新的 MultiLoop 实例
    fn default() -> Self {
        Self::new()
    }
}