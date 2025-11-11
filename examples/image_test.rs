use perple::perple::Perple;
use perple::{
    color::Bounds, draw_detections, load_image
};
use std::sync::{Arc, Mutex};
use std::time::{Instant, Duration};
use std::thread;
use perple::utils::stream::Stream;
use image::DynamicImage;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Perple 图像测试示例");
    println!("===================");
    
    // 加载图像
    // 注意：确保图像文件路径正确
    let image = load_image("data/test/1562400315184.jpg")?;
    
    // 打印图像信息
    println!("原始图像尺寸: {}x{}", image.width(), image.height());
    
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
    
    println!("启动多线程循环处理模式...");
    let start_total = Instant::now();
    
    // 启动color模块的循环运行模式
    perple.start_color_loop();
    
    // 等待一段时间让处理完成
    thread::sleep(Duration::from_millis(1000)); // 增加等待时间到1秒
    
    // 停止color模块的循环运行模式
    perple.stop_color_loop();
    
    let total_duration = start_total.elapsed();
    println!("总处理耗时: {:?}", total_duration);
    
    // 从结果流中获取检测结果
    let bounds = {
        let mut bounds_stream = bounds_stream.lock().unwrap();
        bounds_stream.read().unwrap_or_else(|| Bounds::new())
    };
    
    println!("检测到 {} 个目标", bounds.len());
    
    // 显示检测结果
    for (i, detection) in bounds.iter().enumerate() {
        println!("  目标 {}: {} - 置信度: {:.2} - 位置: ({:.1}, {:.1}, {:.1}, {:.1})", 
                i + 1, 
                detection.class_name, 
                detection.confidence,
                detection.bbox.x1,
                detection.bbox.y1,
                detection.bbox.x2,
                detection.bbox.y2);
    }
    
    // 在图像上绘制检测框
    println!("\n正在绘制检测结果...");
    let result_image = draw_detections(&image, bounds.as_slice());
    
    // 保存结果图像
    let output_path = "results/image_test_result.jpg";
    if let Err(e) = result_image.save(output_path) {
        eprintln!("保存图像时出错: {}", e);
    } else {
        println!("结果已保存到: {}", output_path);
    }
    
    println!("\n示例完成!");
    println!("\n颜色说明:");
    println!("- 青色框: 检测到的person目标");
    println!("- 红色框: 其他检测到的目标");
    
    Ok(())
}