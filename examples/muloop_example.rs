use perple::utils::muloop::{MultiLoop, LoopMode};
use std::sync::{Arc, Mutex};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("MultiLoop 通用循环控制示例");
    println!("========================");
    
    println!("1. 按次数循环模式");
    {
        let mut counter = 0;
        let mut muloop = MultiLoop::new();
        
        let start = Instant::now();
        muloop.start(LoopMode::Count(5), move || {
            counter += 1;
            println!("  执行第 {} 次", counter);
        }, 200)?; // 200ms间隔
        
        muloop.join()?;
        let duration = start.elapsed();
        println!("  执行5次耗时: {:?}", duration);
    }
    
    println!("\n2. 按时间循环模式");
    {
        let counter = Arc::new(Mutex::new(0));
        let counter_clone = Arc::clone(&counter);
        let mut muloop = MultiLoop::new();
        
        let start = Instant::now();
        muloop.start(LoopMode::Duration(1000), move || {
            let mut cnt = counter_clone.lock().unwrap();
            *cnt += 1;
            println!("  执行第 {} 次", *cnt);
        }, 150)?; // 150ms间隔
        
        muloop.join()?;
        let duration = start.elapsed();
        let final_count = *counter.lock().unwrap();
        println!("  在 {:?} 内执行了 {} 次", duration, final_count);
    }
    
    println!("\n3. 持续循环模式");
    {
        let counter = Arc::new(Mutex::new(0));
        let counter_clone = Arc::clone(&counter);
        let mut muloop = MultiLoop::new();
        
        muloop.start(LoopMode::Continuous, move || {
            let mut cnt = counter_clone.lock().unwrap();
            *cnt += 1;
        }, 100)?; // 100ms间隔
        
        // 等待一段时间后手动停止
        std::thread::sleep(std::time::Duration::from_millis(800));
        muloop.stop();
        muloop.join()?;
        
        let final_count = *counter.lock().unwrap();
        println!("  在 800ms 内执行了 {} 次", final_count);
    }
    
    println!("\n所有示例完成!");
    Ok(())
}