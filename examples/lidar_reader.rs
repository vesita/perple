use pcd_rs::DynReader;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let file_path = "data/lidar/000000.pcd";
    
    // 检查文件是否存在
    if !Path::new(file_path).exists() {
        println!("错误: 文件 {} 不存在", file_path);
        return Ok(());
    }

    // 打开PCD文件
    let mut reader = DynReader::open(file_path)?;
    
    println!("正在读取点云文件: {}", file_path);
    println!("点云数据格式信息:");
    
    // 获取点云的基本信息
    let header = reader.meta();
    println!("  字段: {:?}", header.field_defs.fields);
    println!("  数量: {:?}", header.num_points);
    println!("  宽度: {}", header.width);
    println!("  高度: {}", header.height);
    println!("  视点信息: {:?}", header.viewpoint);
    println!("  数据存储类型: {:?}", header.data);
    
    // 读取前几个点以查看数据结构
    println!("\n前10个点的数据:");
    for i in 0..10 {
        match reader.next() {
            Some(Ok(point)) => {
                println!("  点 {}: {:?}", i + 1, point);
            }
            Some(Err(e)) => {
                println!("  读取点 {} 时出错: {}", i + 1, e);
                break;
            }
            None => {
                println!("  点云文件中只有 {} 个点", i);
                break;
            }
        }
    }

    // 统计总点数
    let mut count = 0;
    while let Some(_) = reader.next() {
        count += 1;
        // 为了效率，我们只计算剩余点的数量，而不实际读取点的内容
        if count % 10000 == 0 {
            println!("已读取 {} 个点...", count);
        }
    }
    println!("点云文件总共有 {} 个点", 10 + count);

    Ok(())
}