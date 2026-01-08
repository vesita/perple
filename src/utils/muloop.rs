use std::sync::Arc;
use std::time::Duration;

use tokio::time::sleep;
use tokio::task::JoinHandle;

use crate::utils::stream::{Eap, new_eap};
use log::error;

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

    /// 循环执行任务句柄，用于等待任务完成
    coroutine_handle: Option<JoinHandle<()>>,
}

impl MultiLoop {
    /// 创建一个新的 MultiLoop 实例
    ///
    /// # 返回值
    /// 返回初始化后的 MultiLoop 实例，默认处于停止状态
    pub fn new() -> Self {
        Self {
            running: new_eap(false),
            coroutine_handle: None,
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
    pub async fn start<F>(
        &mut self,
        mode: LoopMode,
        mut callback: F,
        interval_ms: u64,
    ) -> Result<(), String>
    where
        F: FnMut() + Send + 'static,
    {
        {
            let mut running = self.running.lock().await;
            if *running {
                error!("循环已在运行中");
                return Err("循环已在运行中".to_string());
            }
            *running = true;
        } // 释放锁

        let loop_running = Arc::clone(&self.running);

        // 启动tokio任务
        self.coroutine_handle = Some(tokio::spawn(async move {
            match mode {
                LoopMode::Count(count) => {
                    let mut counter = 0;
                    loop {
                        {
                            let running = loop_running.lock().await;
                            if !*running || counter >= count {
                                let mut running_mut = loop_running.lock().await;
                                *running_mut = false;
                                break;
                            }
                            // 释放锁后执行回调
                        }
                        
                        callback();
                        counter += 1;
                        
                        // 控制处理频率，防止占用过多CPU资源
                        sleep(Duration::from_millis(interval_ms)).await;
                    }
                }
                LoopMode::Duration(duration_ms) => {
                    let start_time = std::time::Instant::now();
                    loop {
                        {
                            let running = loop_running.lock().await;
                            if !*running || start_time.elapsed().as_millis() >= duration_ms as u128 {
                                let mut running_mut = loop_running.lock().await;
                                *running_mut = false;
                                break;
                            }
                            // 释放锁后执行回调
                        }
                        
                        callback();
                        
                        // 控制处理频率，防止占用过多CPU资源
                        sleep(Duration::from_millis(interval_ms)).await;
                    }
                }
                LoopMode::Signal => {
                    loop {
                        {
                            let running = loop_running.lock().await;
                            if !*running {
                                break;
                            }
                            // 释放锁后执行回调
                        }
                        
                        callback();
                        
                        // 控制处理频率，防止占用过多CPU资源
                        sleep(Duration::from_millis(interval_ms)).await;
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
    pub async fn start_with_method<T, F>(
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
        {
            let mut running = self.running.lock().await;
            if *running {
                error!("循环已在运行中");
                return Err("循环已在运行中".to_string());
            }
            *running = true;
        } // 释放锁

        let loop_running = Arc::clone(&self.running);
        let loop_object = Arc::clone(&object);

        self.coroutine_handle = Some(tokio::spawn(async move {
            match mode {
                LoopMode::Count(count) => {
                    let mut counter = 0;
                    loop {
                        {
                            let running = loop_running.lock().await;
                            if !*running || counter >= count {
                                let mut running_mut = loop_running.lock().await;
                                *running_mut = false;
                                break;
                            }
                            // 释放锁后执行方法
                        }
                        
                        {
                            let mut obj = loop_object.lock().await;
                            method(&mut *obj);
                        }
                        
                        counter += 1;
                        
                        // 控制处理频率，防止占用过多CPU资源
                        sleep(Duration::from_millis(interval_ms)).await;
                    }
                }
                LoopMode::Duration(duration_ms) => {
                    let start_time = std::time::Instant::now();
                    loop {
                        {
                            let running = loop_running.lock().await;
                            if !*running || start_time.elapsed().as_millis() >= duration_ms as u128 {
                                let mut running_mut = loop_running.lock().await;
                                *running_mut = false;
                                break;
                            }
                            // 释放锁后执行方法
                        }
                        
                        {
                            let mut obj = loop_object.lock().await;
                            method(&mut *obj);
                        }
                        
                        // 控制处理频率，防止占用过多CPU资源
                        sleep(Duration::from_millis(interval_ms)).await;
                    }
                }
                LoopMode::Signal => {
                    loop {
                        {
                            let running = loop_running.lock().await;
                            if !*running {
                                break;
                            }
                            // 释放锁后执行方法
                        }
                        
                        {
                            let mut obj = loop_object.lock().await;
                            method(&mut *obj);
                        }
                        
                        // 控制处理频率，防止占用过多CPU资源
                        sleep(Duration::from_millis(interval_ms)).await;
                    }
                }
            }
        }));

        Ok(())
    }

    /// 停止循环执行
    ///
    /// 将运行状态设置为 false，使循环在下一次迭代时退出
    pub async fn stop(&mut self) {
        let mut running = self.running.lock().await;
        *running = false;
    }

    /// 检查循环是否正在运行
    ///
    /// # 返回值
    /// 如果循环正在运行返回 true，否则返回 false
    pub async fn is_running(&self) -> bool {
        *self.running.lock().await
    }

    /// 等待循环任务执行完成
    ///
    /// # 返回值
    /// 任务成功完成返回 Ok(())，否则返回 Err 错误信息
    pub async fn join(&mut self) -> Result<(), String> {
        if let Some(handle) = self.coroutine_handle.take() {
            handle
                .await
                .map_err(|_| "等待任务完成时发生错误".to_string())?;
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
