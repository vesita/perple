use perple::perple::Perple;
use perple::LoopMode;
use perple::{
    color::Bounds, load_image
};
use std::sync::{Arc, Mutex};
use std::time::{Instant, Duration};
use std::thread;
use perple::utils::stream::Stream;
use image::DynamicImage;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Perple 循环模式示例");
    println!("===================");
    
    // 加载图像
    let image = load_image("data/test/1562400315184.jpg")?;
    
    println!("1. 按次数循环模式（执行3次）");
    {
        // 创建数据流
        let img_stream = Arc::new(Mutex::new(Stream::new()));
        let bounds_stream = Arc::new(Mutex::new(Stream::new()));
        
        // 创建Perple实例
        let mut perple = Perple::new(
            Arc::clone(&img_stream),
            Arc::clone(&bounds_stream),
            "module/color/yolo11n.onnx",
        );
        
        // 更新图像到流中
        perple.update_image(image.clone());
        
        let start = Instant::now();
        
        // 启动3次循环处理
        perple.start_color_loop_count(3)?;
        
        // 等待处理完成
        perple.join_color_thread()?;
        
        let duration = start.elapsed();
        println!("  执行3次处理耗时: {:?}", duration);
        
        // 检查结果
        let bounds = {
            let mut bounds_stream = bounds_stream.lock().unwrap();
            bounds_stream.read().unwrap_or_else(|| Bounds::new())
        };
        println!("  检测到 {} 个目标", bounds.len());
    }
    
    println!("\n2. 按时间循环模式（执行2秒）");
    {
        // 创建数据流
        let img_stream = Arc::new(Mutex::new(Stream::new()));
        let bounds_stream = Arc::new(Mutex::new(Stream::new()));
        
        // 创建Perple实例
        let mut perple = Perple::new(
            Arc::clone(&img_stream),
            Arc::clone(&bounds_stream),
            "module/color/yolo11n.onnx",
        );
        
        // 更新图像到流中
        perple.update_image(image.clone());
        
        let start = Instant::now();
        
        // 启动2秒循环处理
        perple.start_color_loop_duration(2000)?;
        
        // 等待处理完成
        perple.join_color_thread()?;
        
        let duration = start.elapsed();
        println!("  执行2秒处理耗时: {:?}", duration);
        
        // 检查结果
        let bounds = {
            let mut bounds_stream = bounds_stream.lock().unwrap();
            bounds_stream.read().unwrap_or_else(|| Bounds::new())
        };
        println!("  检测到 {} 个目标", bounds.len());
    }
    
    println!("\n3. 持续循环模式（手动停止）");
    {
        // 创建数据流
        let img_stream = Arc::new(Mutex::new(Stream::new()));
        let bounds_stream = Arc::new(Mutex::new(Stream::new()));
        
        // 创建Perple实例
        let mut perple = Perple::new(
            Arc::clone(&img_stream),
            Arc::clone(&bounds_stream),
            "module/color/yolo11n.onnx",
        );
        
        // 更新图像到流中
        perple.update_image(image.clone());
        
        let start = Instant::now();
        
        // 启动持续循环处理
        perple.start_color_loop()?;
        
        // 等待一段时间后手动停止
        thread::sleep(Duration::from_millis(1500));
        perple.stop_color_loop();
        
        // 等待线程结束
        perple.join_color_thread()?;
        
        let duration = start.elapsed();
        println!("  持续循环处理耗时: {:?}", duration);
        
        // 检查结果
        let bounds = {
            let mut bounds_stream = bounds_stream.lock().unwrap();
            bounds_stream.read().unwrap_or_else(|| Bounds::new())
        };
        println!("  检测到 {} 个目标", bounds.len());
    }
    
    println!("\n4. 等待结果模式（获得结果后立即停止）");
    {
        // 创建数据流
        let img_stream = Arc::new(Mutex::new(Stream::new()));
        let bounds_stream = Arc::new(Mutex::new(Stream::new()));
        
        // 创建Perple实例
        let mut perple = Perple::new(
            Arc::clone(&img_stream),
            Arc::clone(&bounds_stream),
            "module/color/yolo11n.onnx",
        );
        
        // 更新图像到流中
        perple.update_image(image.clone());
        
        let start = Instant::now();
        
        // 启动持续循环处理
        perple.start_color_loop()?;
        
        // 等待结果或超时
        if perple.wait_for_result(5000) {
            println!("  在超时前获得结果");
        } else {
            println!("  处理超时");
        }
        
        // 停止处理并等待线程结束
        perple.stop_color_loop();
        perple.join_color_thread()?;
        
        let duration = start.elapsed();
        println!("  等待结果处理耗时: {:?}", duration);
        
        // 检查结果
        let bounds = {
            let mut bounds_stream = bounds_stream.lock().unwrap();
            bounds_stream.read().unwrap_or_else(|| Bounds::new())
        };
        println!("  检测到 {} 个目标", bounds.len());
    }
    
    println!("\n所有示例完成!");
    
    Ok(())
}