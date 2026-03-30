/// 数据加载器独立路径模式示例
///
/// 此示例展示如何使用 DataLoader 的新方法从独立路径加载图像和点云数据
/// 适用于以下目录结构:
/// - data/integration/image/set1/ - 存放图像文件
/// - data/integration/pcd/set1/ - 存放点云文件
///
/// 两个目录相互独立，不要求文件名匹配

use perple::optional::data_loader::DataLoader;
use std::io;

#[tokio::main]
async fn main() -> io::Result<()> {
    println!("DataLoader 独立路径模式示例");
    println!("========================\n");

    // 示例 1: 使用旧格式 (target_path/camera/ 和 target_path/lidar/)
    println!("1. 旧格式示例 (注释掉的需要相应路径存在):");
    // let mut loader_old = DataLoader::new("data/old_format".to_string());
    // loader_old.load().await?;

    // 示例 2: 使用新格式 - 独立路径模式
    println!("\n2. 新格式 - 独立路径模式:");
    println!("   图像路径：data/integration/image/set1/");
    println!("   点云路径：data/integration/pcd/set1/");
    
    // 创建数据加载器，指定独立的图像和点云路径
    let mut loader = DataLoader::new_independent(
        "data/integration/image/set1".to_string(),
        "data/integration/pcd/set1".to_string(),
    );

    // 列出可加载的文件
    match loader.list_files() {
        Ok(files) => {
            println!("\n找到 {} 个文件对:", files.len());
            for (i, file_pair) in files.iter().enumerate() {
                println!("  [{}] 图像：{} | 点云：{}", i + 1, file_pair[0], file_pair[1]);
            }
        }
        Err(e) => {
            eprintln!("列出文件时出错：{}", e);
            return Err(e);
        }
    }

    // 加载单次数据
    println!("\n开始加载单轮数据...");
    if let Err(e) = loader.load().await {
        eprintln!("加载数据失败：{}", e);
        return Err(e);
    }
    println!("单轮数据加载完成!");

    // 如果需要循环加载，可以使用 load_loop()
    // 注意：这将无限循环加载，直到手动中断
    // println!("\n开始循环加载数据 (Ctrl+C 停止)...");
    // if let Err(e) = loader.load_loop().await {
    //     eprintln!("循环加载失败：{}", e);
    //     return Err(e);
    // }

    println!("\n示例完成!");
    Ok(())
}
