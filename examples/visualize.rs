use std::sync::{Arc, Mutex};

use perple::cloud::core::Lidar;
use perple::optional::data_loader::DataLoader;
use perple::tracker::core::Tracker;
use perple::swapl::global_swapl;
use perple::tracker::output::Target;

use expto::rdmp::auto::unit::generate_unit;

use log::info;
use redra_client::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();
    info!("Perple可视化演示");

    let mut data_loader = DataLoader::new("./data/test".to_string());
    let _ = data_loader.load().await;

    let lidar = Arc::new(Mutex::new(Lidar::new()));
    let tracker = Arc::new(Mutex::new(Tracker::new()));

    // 逐帧同步处理（在 blocking 线程中执行同步的 act/run）
    for i in 0..2 {
        info!("处理第 {} 帧...", i + 1);

        let l = Arc::clone(&lidar);
        tokio::task::spawn_blocking(move || {
            let _ = l.lock().unwrap().act();
        })
        .await
        .map_err(|e| format!("lidar任务失败: {}", e))?;

        let t = Arc::clone(&tracker);
        tokio::task::spawn_blocking(move || {
            let _ = t.lock().unwrap().run();
        })
        .await
        .map_err(|e| format!("tracker任务失败: {}", e))?;

        send_visualization().await?;
    }

    Ok(())
}

async fn send_visualization() -> Result<(), Box<dyn std::error::Error>> {
    let swapl = global_swapl();

    // ── 点云 ──
    let cloud_stream = swapl.clouds_out.lock().await;
    if let Some(frame) = cloud_stream.peek_latest() {
        println!("  点云数据对象数量: {}", frame.len());
        let _ = send_point_cloud(&frame).await;
    }
    drop(cloud_stream);

    // ── 跟踪目标 ──
    let target_stream = swapl.targets.lock().await;
    if let Some(targets) = target_stream.peek_latest() {
        println!("  跟踪目标数量: {}", targets.len());
        send_target_boxes(&targets).await?;
        send_speed_arrows(&targets).await?;
    }
    drop(target_stream);

    Ok(())
}

/// 发送带分类颜色的目标包围盒
async fn send_target_boxes(targets: &[Target]) -> Result<(), Box<dyn std::error::Error>> {
    let mut unit = generate_unit();
    const BASE_ID: u64 = 1_000_000;

    for (i, target) in targets.iter().enumerate() {
        let entity_id = BASE_ID + (i as u64) * 4;

        let verts = target.the_box.vertices();
        let points: Vec<Point> = verts.iter()
            .map(|v| Point { x: v.x, y: v.y, z: v.z })
            .collect();

        // AABB 中心
        let mut min = [f32::MAX, f32::MAX, f32::MAX];
        let mut max = [f32::MIN, f32::MIN, f32::MIN];
        for p in &points {
            min[0] = min[0].min(p.x); min[1] = min[1].min(p.y); min[2] = min[2].min(p.z);
            max[0] = max[0].max(p.x); max[1] = max[1].max(p.y); max[2] = max[2].max(p.z);
        }
        if min[0] == f32::MAX {
            continue;
        }
        let cx = (min[0] + max[0]) / 2.0;
        let cy = (min[1] + max[1]) / 2.0;
        let cz = (min[2] + max[2]) / 2.0;

        let material_id = match target.classification.as_str() {
            "dynamic" => "red",
            "static" => "green",
            "movable" => "yellow",
            _ => "metal",
        };

        unit.objects.extend(vec![
            ExObject::from(entity_id),
            ExObject::from(ExMesh::from(Cube { vertices: points })),
            ExObject::from(ExTransform {
                x: cx, y: cy, z: cz,
                rx: 0.0, ry: 0.0, rz: 0.0,
                sx: 1.0, sy: 1.0, sz: 1.0,
            }),
            ExObject { u_object: Some(ex_object::UObject::MaterialId(material_id.to_string())) },
        ]);
    }

    if !unit.objects.is_empty() {
        unit.send().await?;
    }
    Ok(())
}

/// 为高速目标绘制速度方向线
async fn send_speed_arrows(targets: &[Target]) -> Result<(), Box<dyn std::error::Error>> {
    for target in targets {
        if target.speed > 0.5 {
            let center = target.the_box.center();
            let scale = (target.speed * 2.0).min(10.0);
            let dx = target.velocity[0] / target.speed * scale;
            let dy = target.velocity[1] / target.speed * scale;
            let dz = target.velocity[2] / target.speed * scale;

            send_line(
                center.x, center.y, center.z,
                center.x + dx, center.y + dy, center.z + dz,
            ).await?;
        }
    }
    Ok(())
}
