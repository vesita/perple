use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;
use std::io::{self, Write};

use perple::optional::data_loader::DataLoader;
use perple::perple::Perple;
use perple::swapl::Swapl;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Perple与DataLoader集成演示");
    println!("========================");
    
    // 创建数据交换中枢
    let swapl = Arc::new(Mutex::new(Swapl::new()));
    println!("✓ 创建数据交换中枢");
    
    // 创建DataLoader实例
    let mut data_loader = DataLoader::new(
        Arc::clone(&swapl),
        "./data/test".to_string(),
    );
    println!("✓ 创建DataLoader实例");
    
    // 列出要处理的文件
    let files = data_loader.list_files()?;
    println!("✓ 找到 {} 对文件", files.len());
    
    for (i, file_pair) in files.iter().enumerate() {
        println!("  文件对 {}: camera={} lidar={}", i, file_pair[0], file_pair[1]);
    }
    
    print!("按回车键继续加载数据...");
    io::stdout().flush()?;
    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    
    // 加载数据
    println!("正在加载数据...");
    data_loader.load()?;
    println!("✓ 数据加载完成");
    
    // 显示流状态
    show_stream_status(&swapl)?;
    
    print!("按回车键启动Perple处理...");
    io::stdout().flush()?;
    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    
    // 创建Perple实例（使用占位符路径）
    println!("正在创建Perple实例...");
    let mut perple = Perple::new(
        &swapl.lock().unwrap(),
        "./module/color/yolo11n.onnx", // 使用正确的模型路径
        "./config/camera.toml", // 占位符路径
    );
    println!("✓ Perple实例创建完成");
    
    // 启动Perple的各个模块
    println!("正在启动Perple模块...");
    perple.start_color_loop()?;
    perple.start_lidar_loop()?;
    perple.start_tracker_loop()?;
    println!("✓ Perple模块启动完成");
    
    // 等待一段时间让处理完成
    println!("等待处理完成 (5秒)...");
    thread::sleep(Duration::from_secs(5));
    
    // 显示处理后的流状态
    show_stream_status(&swapl)?;
    
    // 停止Perple模块
    println!("正在停止Perple模块...");
    perple.stop_color_loop()?;
    perple.stop_lidar_loop()?;
    perple.stop_tracker_loop()?;
    println!("✓ Perple模块已停止");
    
    println!("\n演示完成!");
    Ok(())
}

fn show_stream_status(swapl: &Arc<Mutex<Swapl>>) -> Result<(), Box<dyn std::error::Error>> {
    let swapl_guard = swapl.lock().unwrap();
    
    // 检查各数据流状态
    let colors_stream = swapl_guard.colors.lock().unwrap();
    println!("  图像数据流大小: {}", colors_stream.len());
    
    let clouds_stream = swapl_guard.clouds.lock().unwrap();
    println!("  点云数据流大小: {}", clouds_stream.len());
    
    let clr_objs_stream = swapl_guard.clr_objs.lock().unwrap();
    println!("  2D检测结果流大小: {}", clr_objs_stream.len());
    
    let cld_objs_stream = swapl_guard.cld_objs.lock().unwrap();
    println!("  3D检测结果流大小: {}", cld_objs_stream.len());
    
    let sights_stream = swapl_guard.sights.lock().unwrap();
    println!("  投影结果流大小: {}", sights_stream.len());
    
    let targets_stream = swapl_guard.targets.lock().unwrap();
    println!("  跟踪结果流大小: {}", targets_stream.len());
    
    Ok(())
}