use tokio::time::{sleep, Duration};
use tokio::sync::Mutex;
use std::sync::Arc;

/// 一个简单的计数器，可在任务间共享
#[derive(Debug)]
struct Counter {
    value: Arc<Mutex<i32>>,
}

impl Counter {
    /// 创建一个新的计数器
    fn new() -> Self {
        Counter {
            value: Arc::new(Mutex::new(0)),
        }
    }

    /// 增加计数器
    async fn increment(&self) {
        let mut value = self.value.lock().await;
        *value += 1;
    }

    /// 获取当前值
    async fn get_value(&self) -> i32 {
        let value = self.value.lock().await;
        *value
    }

    /// 将计数器重置为零
    async fn reset(&self) {
        let mut value = self.value.lock().await;
        *value = 0;
    }
}

#[tokio::main]
async fn main() {
    let counter = Counter::new();
    
    // 为我们的任务克隆计数器
    let counter1 = counter.value.clone();
    let counter2 = counter.value.clone();
    let counter3 = counter.value.clone();
    
    println!("开始使用 tokio 的计数器示例");
    
    // 生成多个将并发增加计数器的任务
    let tasks = vec![
        tokio::spawn(async move {
            let counter = Counter { value: counter1 };
            for _ in 0..5 {
                counter.increment().await;
                let current = counter.get_value().await;
                println!("任务 1 将计数器增加到: {}", current);
                sleep(Duration::from_millis(100)).await;
            }
        }),
        tokio::spawn(async move {
            let counter = Counter { value: counter2 };
            for _ in 0..3 {
                counter.increment().await;
                let current = counter.get_value().await;
                println!("任务 2 将计数器增加到: {}", current);
                sleep(Duration::from_millis(150)).await;
            }
        }),
        tokio::spawn(async move {
            let counter = Counter { value: counter3 };
            sleep(Duration::from_millis(200)).await;
            counter.increment().await;
            let current = counter.get_value().await;
            println!("任务 3 将计数器增加到: {}", current);
        }),
    ];
    
    // 等待所有任务完成
    for task in tasks {
        task.await.unwrap();
    }
    
    let final_value = counter.get_value().await;
    println!("最终计数器值: {}", final_value);
    
    // 重置计数器
    counter.reset().await;
    let reset_value = counter.get_value().await;
    println!("重置后的计数器值: {}", reset_value);
}