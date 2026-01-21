use std::time::Duration;

use perple::optional::data_loader::DataLoader;
use perple::perple::Perple;
use perple::swapl::global_swapl;

use redra::client::*;
use tokio;
use tokio::time::sleep;

use log::info;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)  // 设置默认日志级别为Info
        .init();
    info!("Perple可视化演示");
    let mut data_loader = DataLoader::new("./data/test".to_string());

    // 首先加载数据
    let _ = data_loader.load().await;

    let mut perple = Perple::new();

    // 启动Perple处理流程
    let _ = perple.run().await;

    // 增加等待时间，让数据处理流程有足够时间运行
    info!("等待数据处理完成...");
    // 给数据处理模块足够的时间来处理数据
    let _ = sleep(Duration::from_secs(5)).await;

    // 显示流状态并等待发送完成
    show_stream_status().await?;

    // 再等待一段时间，确保所有数据都已处理
    info!("再次等待数据处理完成...");
    let _ = sleep(Duration::from_secs(5)).await;
    
    // 再次检查流状态
    show_stream_status().await?;

    Ok(())
}

async fn send_points_async(points: Vec<[f32; 3]>) {
    for point in points {
        let _ = send_point(point[0], point[2], point[1]).await;
    }
}

async fn send_boxes_async(boxes: Vec<perple::cloud::CldBud>) {
    for bound in boxes {
        let edges = bound.the_box.edges_z_up();
        for edge in edges {
            let _ = send_segment(edge[0], edge[1]).await;
        }
    }
}

async fn show_stream_status() -> Result<(), Box<dyn std::error::Error>> {
    let swapl = global_swapl();

    let cloud_in_world_stream = swapl.cloud_in_world.lock().await;

    let cld_objs_stream = swapl.cld_objs.lock().await;

    // 准备异步任务
    let point_task = if let Some(frame) = cloud_in_world_stream.get_at(0) {
        println!("  点云数据对象数量: {}", frame.len());
        let points = frame.clone();
        drop(cloud_in_world_stream); // 释放锁
        Some(tokio::spawn(async move {
            send_points_async(points).await;
        }))
    } else {
        drop(cloud_in_world_stream);
        None
    };

    // 准备3D框发送任务
    let box_task = if let Some(bounds) = cld_objs_stream.get_at(0) {
        println!("  3D检测结果对象数量: {}", bounds.len());
        let bounds_data = bounds.clone();
        drop(cld_objs_stream); // 释放锁
        Some(tokio::spawn(async move {
            send_boxes_async(bounds_data).await;
        }))
    } else {
        drop(cld_objs_stream);
        None
    };

    // 等待所有异步任务完成
    if let Some(task) = point_task {
        let _ = task.await;
    }

    if let Some(task) = box_task {
        let _ = task.await;
    }

    Ok(())
}