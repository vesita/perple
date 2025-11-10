use perple::{load_image, load_model, YoloDetector};
use std::time::{Instant, Duration};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_performance() -> Result<(), Box<dyn std::error::Error>> {
        println!("Perple 模型速度基准测试");
        println!("========================");
        
        // 加载模型和图像
        let model = load_model("module/color/yolo11n.onnx")?;
        let image = load_image("data/test/1562400315184.jpg")?;
        
        // 打印图像信息
        println!("原始图像尺寸: {}x{}", image.width(), image.height());
        
        // 创建检测器
        let mut detector = YoloDetector::new(model, 640, 640)
            .with_confidence_threshold(0.5)
            .with_nms_threshold(0.7);
        
        // 预热运行一次
        println!("执行预热运行...");
        detector.detect(&image)?;
        
        // 多次运行以获取平均性能数据
        println!("执行多次推理以测量性能...");
        let iterations = 100;
        let mut total_duration = Duration::new(0, 0);
        let mut min_duration = Duration::new(u64::MAX, 0);
        let mut max_duration = Duration::new(0, 0);
        
        for i in 0..iterations {
            let start = Instant::now();
            detector.detect(&image)?;
            let duration = start.elapsed();
            
            total_duration += duration;
            if duration < min_duration {
                min_duration = duration;
            }
            if duration > max_duration {
                max_duration = duration;
            }
            
            // 每10次迭代打印一次进度
            if (i + 1) % 10 == 0 {
                println!("已完成 {} 次推理", i + 1);
            }
        }
        
        // 计算统计数据
        let average_duration = total_duration / iterations;
        
        println!("\n性能统计 ({} 次推理):", iterations);
        println!("平均耗时: {:?} ({} ms)", average_duration, average_duration.as_millis());
        println!("最小耗时: {:?} ({} ms)", min_duration, min_duration.as_millis());
        println!("最大耗时: {:?} ({} ms)", max_duration, max_duration.as_millis());
        println!("总耗时: {:?}", total_duration);
        
        // 计算FPS (Frames Per Second)
        let fps = 1.0 / (average_duration.as_secs_f64());
        println!("平均 FPS: {:.2}", fps);
        
        // 性能断言 - 确保平均推理时间在合理范围内（例如不超过1秒）
        assert!(average_duration.as_millis() < 1000, "平均推理时间过长: {:?} ms", average_duration.as_millis());
        
        Ok(())
    }
}
