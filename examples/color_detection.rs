use perple::{load_image, load_model, YoloDetector, draw_detections};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Perple 颜色检测示例");
    println!("========================");
    
    // 加载模型和图像
    // 注意：确保模型文件和图像文件路径正确
    let model = load_model("module/color/yolo11n.onnx")?;
    let image = load_image("data/test/1562400315184.jpg")?;
    
    // 打印图像信息
    println!("原始图像尺寸: {}x{}", image.width(), image.height());
    
    // 创建检测器并进行检测
    // 使用640x640的输入尺寸，置信度阈值0.5，NMS阈值0.7
    let mut detector = YoloDetector::new(model, 640, 640)
        .with_confidence_threshold(0.5)
        .with_nms_threshold(0.7);
    
    println!("正在执行目标检测...");
    let detections = detector.detect(&image)?;
    println!("检测到 {} 个目标", detections.len());
    
    // 打印每个检测到的目标的详细信息
    for (i, detection) in detections.iter().enumerate() {
        println!("目标 {}: {} - 置信度: {:.2} - 位置: ({:.1}, {:.1}, {:.1}, {:.1})", 
                i + 1, 
                detection.class_name, 
                detection.confidence,
                detection.bbox.x1,
                detection.bbox.y1,
                detection.bbox.x2,
                detection.bbox.y2);
    }
    
    // 在图像上绘制检测框
    println!("正在绘制检测结果...");
    let result_image = draw_detections(&image, &detections);
    
    // 保存结果图像
    let output_path = "results/color_detection_result.jpg";
    result_image.save(output_path)?;
    println!("结果已保存到: {}", output_path);
    
    println!("\n颜色说明:");
    println!("- 青色框: 检测到的person目标");
    println!("- 红色框: 其他检测到的目标");
    
    Ok(())
}