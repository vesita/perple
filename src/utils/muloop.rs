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
///
/// # 使用注意
/// - 停止循环时，当前正在执行的回调可能会完成后再退出（取决于调用时机）
/// - 建议在对象析构前先调用 stop() 并等待 join() 完成，确保资源正确释放
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
    ///
    /// # 使用注意
    /// - 停止信号被检测到时，当前正在执行的回调可能会完成后再退出
    /// - 循环会在下一次检查运行状态时立即退出，不会等待当前回调完成
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

        // 启动 tokio 任务
        self.coroutine_handle = Some(tokio::spawn(async move {
            match mode {
                LoopMode::Count(count) => {
                    let mut counter= 0;
                    while counter < count {
                        // 检查是否应该停止
                        let should_stop = {
                            let running = loop_running.lock().await;
                            !*running
                        };
                        
                        if should_stop {
                            break;
                        }
                        
                        callback();
                        counter += 1;
                        
                        // 控制处理频率，防止占用过多 CPU 资源
                        sleep(Duration::from_millis(interval_ms)).await;
                    }
                    
                    // 确保退出时将状态设置为 false
                    let mut running = loop_running.lock().await;
                    *running = false;
                }
                LoopMode::Duration(duration_ms) => {
                    let start_time = std::time::Instant::now();
                    while start_time.elapsed().as_millis() < duration_ms as u128 {
                        // 检查是否应该停止
                        let should_stop = {
                            let running = loop_running.lock().await;
                            !*running
                        };
                        
                        if should_stop {
                            break;
                        }
                        
                        callback();
                        
                        // 控制处理频率，防止占用过多 CPU 资源
                        sleep(Duration::from_millis(interval_ms)).await;
                    }
                    
                    // 确保退出时将状态设置为 false
                    let mut running = loop_running.lock().await;
                    *running = false;
                }
                LoopMode::Signal => {
                    loop {
                        // 检查是否应该停止
                        let should_stop = {
                            let running = loop_running.lock().await;
                            !*running
                        };
                        
                        if should_stop {
                            break;
                        }
                        
                        callback();
                        
                        // 控制处理频率，防止占用过多 CPU 资源
                        sleep(Duration::from_millis(interval_ms)).await;
                    }
                    
                    // 确保退出时将状态设置为 false
                    let mut running = loop_running.lock().await;
                    *running = false;
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
    /// * `method` - 要执行的方法（通常是一个闭包，调用对象的同步方法）
    /// * `interval_ms` - 每次循环之间的间隔时间（毫秒）
    ///
    /// # 返回值
    /// 成功启动返回 Ok(())，如果循环已在运行则返回 Err 错误信息
    ///
    /// # 泛型参数
    /// * `T` - 对象类型，必须实现 Send + 'static 特征
    /// * `F` - 方法类型，必须实现 Fn(&mut T) + Send + 'static 特征
    ///
    /// # 使用注意
    /// - method 闭包参数需要显式类型注解以避免编译错误
    /// - 停止信号被检测到时，当前正在执行的方法可能会完成后再退出
  pub async fn start_with_method<T, F>(
        &mut self,
       mode: LoopMode,
       object: Eap<T>,
      method: F,
      interval_ms: u64,
    ) -> Result<(), String>
    where
       T: Send + 'static,
       F: Fn(&mut T) + Send + Sync + 'static,
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
    let method = Arc::new(method);

        self.coroutine_handle = Some(tokio::spawn(async move {
            match mode {
             LoopMode::Count(count) => {
                 let mut counter= 0;
                   while counter < count {
                        // 检查是否应该停止
                    let should_stop = {
                         let running = loop_running.lock().await;
                            !*running
                        };

                     if should_stop {
                           break;
                        }

                        // 在 blocking 线程池中执行同步方法
                        {
                       let obj = Arc::clone(&loop_object);
                      let method = Arc::clone(&method);
                        let _ = tokio::task::spawn_blocking(move || {
                             if let Ok(mut o) = obj.try_lock() {
                                method(&mut *o);
                                }
                           }).await;
                        }

                     counter += 1;

                        // 控制处理频率，防止占用过多 CPU 资源
                      sleep(Duration::from_millis(interval_ms)).await;
                    }

                    // 确保退出时将状态设置为 false
                 let mut running = loop_running.lock().await;
                   *running = false;
                }
             LoopMode::Duration(duration_ms) => {
                 let start_time = std::time::Instant::now();
                   while start_time.elapsed().as_millis() < duration_ms as u128 {
                        // 检查是否应该停止
                    let should_stop = {
                         let running = loop_running.lock().await;
                            !*running
                        };

                     if should_stop {
                           break;
                        }

                        // 在 blocking 线程池中执行同步方法
                        {
                       let obj = Arc::clone(&loop_object);
                      let method = Arc::clone(&method);
                        let _ = tokio::task::spawn_blocking(move || {
                             if let Ok(mut o) = obj.try_lock() {
                                method(&mut *o);
                                }
                           }).await;
                        }

                        // 控制处理频率，防止占用过多 CPU 资源
                      sleep(Duration::from_millis(interval_ms)).await;
                    }

                    // 确保退出时将状态设置为 false
                 let mut running = loop_running.lock().await;
                   *running = false;
                }
             LoopMode::Signal => {
                    loop {
                        // 检查是否应该停止
                    let should_stop = {
                         let running = loop_running.lock().await;
                            !*running
                        };

                     if should_stop {
                           break;
                        }

                        // 在 blocking 线程池中执行同步方法
                        {
                       let obj = Arc::clone(&loop_object);
                      let method = Arc::clone(&method);
                        let _ = tokio::task::spawn_blocking(move || {
                             if let Ok(mut o) = obj.try_lock() {
                                method(&mut *o);
                                }
                           }).await;
                        }

                        // 控制处理频率，防止占用过多 CPU 资源
                      sleep(Duration::from_millis(interval_ms)).await;
                    }

                    // 确保退出时将状态设置为 false
                 let mut running = loop_running.lock().await;
                   *running = false;
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
        drop(running); // 立即释放锁，让循环能够获取到停止标志
        
        // 等待任务实际完成，确保状态已更新
        if let Some(handle) = self.coroutine_handle.take() {
         let _ = handle.await;
        }
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
    ///
    /// # 使用注意
    /// 必须先调用 stop() 或者等待循环自然结束，否则 join() 会一直阻塞
    pub async fn join(&mut self) -> Result<(), String> {
        if let Some(handle) = self.coroutine_handle.take() {
            handle
                .await
                .map_err(|_| "等待任务完成时发生错误".to_string())?;
        }
        Ok(())
    }

    /// 启动针对特定对象异步方法的循环执行
    ///
    /// # 参数
    /// * `mode` - 循环模式，指定循环的执行方式
    /// * `object` - 需要对其方法进行循环调用的对象引用
    /// * `method` - 要执行的异步方法（通常是一个闭包，调用对象的异步方法）
    /// * `interval_ms` - 每次循环之间的间隔时间（毫秒）
    ///
    /// # 返回值
    /// 成功启动返回 Ok(())，如果循环已在运行则返回 Err 错误信息
    ///
    /// # 泛型参数
    /// * `T` - 对象类型，必须实现 Send + 'static 特征
    /// * `F` - 方法类型，必须实现 Fn(Eap<T>) -> Fut + Send + 'static 特征
    /// * `Fut` - Future 类型，必须实现 Future<Output = ()> + Send 特征
    ///
    /// # 使用注意
    /// - method 闭包参数需要显式类型注解以避免编译错误
    /// - 停止信号被检测到时，当前正在执行的方法可能会完成后再退出
   pub async fn start_with_async_method<T, F, Fut>(
        &mut self,
        mode: LoopMode,
        object: Eap<T>,
       method: F,
       interval_ms: u64,
    ) -> Result<(), String>
    where
        T: Send + 'static,
       F: Fn(Eap<T>) -> Fut + Send + 'static,
        Fut: std::future::Future<Output = ()> + Send,
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
                    while counter < count {
                        // 检查是否应该停止
                        let should_stop = {
                            let running = loop_running.lock().await;
                            !*running
                        };

                        if should_stop {
                            break;
                        }

                        {
                           let obj = Arc::clone(&loop_object);
                           method(obj).await;
                        }

                        counter += 1;

                        // 控制处理频率，防止占用过多 CPU 资源
                        sleep(Duration::from_millis(interval_ms)).await;
                    }

                    // 确保退出时将状态设置为 false
                    let mut running = loop_running.lock().await;
                    *running = false;
                }
                LoopMode::Duration(duration_ms) => {
                    let start_time = std::time::Instant::now();
                    while start_time.elapsed().as_millis() < duration_ms as u128 {
                        // 检查是否应该停止
                        let should_stop = {
                            let running = loop_running.lock().await;
                            !*running
                        };

                        if should_stop {
                            break;
                        }

                        {
                           let obj = Arc::clone(&loop_object);
                           method(obj).await;
                        }

                        // 控制处理频率，防止占用过多 CPU 资源
                        sleep(Duration::from_millis(interval_ms)).await;
                    }

                    // 确保退出时将状态设置为 false
                    let mut running = loop_running.lock().await;
                    *running = false;
                }
                LoopMode::Signal => {
                    loop {
                        // 检查是否应该停止
                        let should_stop = {
                            let running = loop_running.lock().await;
                            !*running
                        };

                        if should_stop {
                            break;
                        }

                        {
                           let obj = Arc::clone(&loop_object);
                           method(obj).await;
                        }

                        // 控制处理频率，防止占用过多 CPU 资源
                        sleep(Duration::from_millis(interval_ms)).await;
                    }

                    // 确保退出时将状态设置为 false
                    let mut running = loop_running.lock().await;
                    *running = false;
                }
            }
        }));

        Ok(())
    }
}

impl Drop for MultiLoop {
    /// 析构时确保任务被正确清理
    ///
    /// 注意：这里只是中止任务，不等待完成，因为 async drop 不支持 await
    /// 如果需要优雅地停止任务，应该在销毁前显式调用 stop() 和 join()
    fn drop(&mut self) {
        if let Some(handle) = &self.coroutine_handle {
            if !handle.is_finished() {
                handle.abort();
                log::warn!("MultiLoop 被销毁时任务仍在运行，已中止任务");
            }
        }
    }
}

impl Default for MultiLoop {
    /// 默认实现，创建一个新的 MultiLoop 实例
    fn default() -> Self {
        Self::new()
    }
}
