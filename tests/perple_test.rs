use perple::perple::Perple;
use perple::utils::swapl::Swapl;
use perple::{
    color::ClrBud, 
    cloud::CldBud,

};
use std::time::Instant;
use pcd_rs::DynReader;

#[cfg(test)]
mod tests {
    use perple::color::load_image;

    use super::*;

    #[test]
    fn test_perple_with_test_data() {
        // TODO: 后续需要进行功能迭代 - 添加更多测试用例和边界条件测试
        
        // 创建Swapl数据中枢
        let pool = Swapl::new();
        
        // 创建Perple实例
        let mut perple = Perple::new(
            &pool,
            "module/color/yolo11n.onnx",
        );
        
        // 测试图像处理
        test_image_processing(&mut perple);
        
        // 测试点云处理
        test_lidar_processing(&mut perple);
        
        // TODO: 后续需要进行功能迭代 - 添加world模块相关测试
        // TODO: 后续需要进行功能迭代 - 添加更复杂的多模态数据融合测试
        // TODO: 后续需要进行功能迭代 - 添加性能基准测试
    }
    
    #[test]
    fn test_perple_run_method() {
        println!("开始测试Perple的run方法...");
        
        // 创建Swapl数据中枢
        let pool = Swapl::new();
        
        // 创建Perple实例
        let mut perple = Perple::new(
            &pool,
            "module/color/yolo11n.onnx",
        );
        
        // 测试run方法
        match perple.run() {
            Ok(_) => {
                println!("  Perple run方法执行成功");
                
                // 检查color和lidar循环是否正在运行
                let is_color_running = perple.is_color_running();
                let is_lidar_running = perple.is_lidar_running();
                
                println!("  Color模块运行状态: {}", is_color_running);
                println!("  Lidar模块运行状态: {}", is_lidar_running);
                
                // 停止循环
                perple.stop_color_loop();
                perple.stop_lidar_loop();
                
                // 等待线程结束
                let _ = perple.join_color_thread();
                // TODO: 添加join_lidar_thread方法后启用
                // let _ = perple.join_lidar_thread();
            },
            Err(e) => {
                eprintln!("  Perple run方法执行失败: {}", e);
            }
        }
    }
    
    fn test_image_processing(perple: &mut Perple) {
        println!("开始测试图像处理功能...");
        
        // 加载测试图像
        let image_paths = [
            "data/test/images/1562400315184.jpg",
            "data/test/images/IMG_20191030_211644.jpg",
            "data/test/images/IMG_20191031_164949.jpg"
        ];
        
        for (i, image_path) in image_paths.iter().enumerate() {
            println!("处理第{}张图像: {}", i + 1, image_path);
            
            match load_image(image_path) {
                Ok(image) => {
                    println!("  成功加载图像，尺寸: {}x{}", image.width(), image.height());
                    
                    // 更新图像到流中
                    perple.update_image(image.clone());
                    
                    // 启动一次处理循环模式
                    match perple.start_color_loop_count(1) {
                        Ok(_) => {
                            // 等待处理完成或超时
                            if perple.wait_for_result(5000) {
                                println!("  图像处理完成");
                            } else {
                                println!("  图像处理超时");
                                perple.stop_color_loop();
                            }
                            
                            // 等待线程结束
                            if let Err(e) = perple.join_color_thread() {
                                eprintln!("  等待线程结束时出错: {}", e);
                            }
                            
                            // 获取检测结果
                            let bounds = {
                                let mut bounds_stream = perple.clr_bud_stream.lock().unwrap();
                                bounds_stream.read().unwrap_or_else(|| Vec::new())
                            };
                            
                            println!("  检测到 {} 个目标", bounds.len());
                            
                            // 显示部分检测结果
                            for (j, detection) in bounds.iter().take(3).enumerate() {
                                println!("    目标 {}: {} - 置信度: {:.2}", 
                                        j + 1, 
                                        detection.class_name, 
                                        detection.confidence);
                            }
                        },
                        Err(e) => {
                            eprintln!("  启动图像处理循环时出错: {}", e);
                        }
                    }
                },
                Err(e) => {
                    eprintln!("  加载图像时出错: {}", e);
                }
            }
        }
    }
    
    fn test_lidar_processing(perple: &mut Perple) {
        println!("开始测试点云处理功能...");
        
        // 加载测试点云数据
        let lidar_path = "data/test/lidars/000000.pcd";
        
        match DynReader::open(lidar_path) {
            Ok(mut reader) => {
                println!("  成功打开点云文件: {}", lidar_path);
                
                // 获取点云基本信息
                let header = reader.meta();
                println!("  点云包含 {:?} 个点", header.num_points);
                
                // TODO: 后续需要进行功能迭代 - 实现完整的点云数据处理测试
                // 目前只是简单读取文件信息，尚未真正测试lidar模块
                
                // 可以在这里添加更多点云处理测试
                // 例如：读取点云数据并发送到perple.lid_stream中进行处理
                
                println!("  点云文件读取测试完成");
            },
            Err(e) => {
                eprintln!("  打开点云文件时出错: {}", e);
            }
        }
    }
    
    #[test]
    fn test_perple_concurrent_processing() {
        // TODO: 后续需要进行功能迭代 - 添加并发处理测试
        // 测试同时处理图像和点云数据的能力
        println!("并发处理测试占位符");
    }
    
    #[test]
    fn test_perple_performance() {
        // TODO: 后续需要进行功能迭代 - 添加性能测试
        // 测试处理速度和资源消耗
        println!("性能测试占位符");
    }
}