use pcd_rs::DynReader;
use perple::lidar::squeeze::Squeeze;
use perple::lidar::lifra::Lifra;
use perple::config::{POINTS_CAPACITY, RESOLUTION};
use std::path::Path;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let file_path = "data/lidar/000000.pcd";
    
    // 检查文件是否存在
    if !Path::new(file_path).exists() {
        println!("错误: 文件 {} 不存在", file_path);
        return Ok(());
    }

    // 创建Squeeze实例
    // 使用默认配置：分辨率为RESOLUTION，区域大小为POINTS_CAPACITY的一半
    let mut squeeze = Squeeze::new(RESOLUTION, POINTS_CAPACITY / 2, POINTS_CAPACITY / 2);
    
    // 打开PCD文件
    let mut reader = DynReader::open(file_path)?;
    
    println!("正在读取点云文件: {}", file_path);
    let header = reader.meta();
    println!("点云数据数量: {:?}", header.num_points);
    
    // 更新Squeeze中的点云数据
    squeeze.records_mut().update(&mut reader);
    
    println!("成功加载 {} 个点到Squeeze中", squeeze.len());
    
    // 执行聚类操作
    println!("开始执行聚类...");
    let start_time = Instant::now();
    squeeze.claster();
    let duration = start_time.elapsed();
    println!("聚类耗时: {:?}", duration);
    // 输出聚类结果
    let clusters = squeeze.targets().objects();
    println!("发现 {} 个聚类对象", clusters.len());
    
    // 显示每个聚类的边界框信息
    for (i, cluster) in clusters.iter().enumerate() {
        println!("聚类 {}: min({:.3}, {:.3}, {:.3}), max({:.3}, {:.3}, {:.3})", 
                 i+1,
                 cluster.x_min, cluster.y_min, cluster.z_min,
                 cluster.x_max, cluster.y_max, cluster.z_max);
    }

    Ok(())
}