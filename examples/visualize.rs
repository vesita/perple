use std::sync::{Arc, Mutex};

use perple::cloud::core::Lidar;
use perple::optional::data_loader::DataLoader;
use perple::tracker::core::Tracker;
use perple::swapl::global_swapl;
use perple::tracker::output::Target;

use expto::rdmp::auto::unit::generate_unit;
use expto::rdmp::proto::command::{CommandType, ExCommand};
use expto::rdmp::*;

use log::info;
use redra_client::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();
    info!("Perple 检测流程可视化（14 帧）");

    let mut data_loader = DataLoader::new("./data/test".to_string());
    data_loader.set_frame_limit(14);
    info!("开始加载数据...");
    let load_start = std::time::Instant::now();
    let _ = data_loader.load().await;
    info!("数据加载完成，耗时 {}ms", load_start.elapsed().as_millis());

    let lidar = Arc::new(Mutex::new(Lidar::new()));
    let tracker = Arc::new(Mutex::new(Tracker::new()));

    let n_frames = 14;
    for i in 0..n_frames {
        info!("─── 第 {}/{} 帧 ───", i + 1, n_frames);

        // ── LiDAR 处理（点云 → 地面检测 → 聚类） ──
        {
            let l = Arc::clone(&lidar);
            tokio::task::spawn_blocking(move || {
                let _ = l.lock().unwrap().act();
            })
            .await
            .map_err(|e| format!("Lidar 任务失败: {}", e))?;
        }

        // ── 跟踪（检测关联 → Kalman → 速度分类） ──
        {
            let t = Arc::clone(&tracker);
            tokio::task::spawn_blocking(move || {
                let _ = t.lock().unwrap().run();
            })
            .await
            .map_err(|e| format!("Tracker 任务失败: {}", e))?;
        }

        // ── 可视化 ──
        send_frame(i, n_frames).await?;
    }

    info!("所有帧处理完成");
    Ok(())
}

async fn send_frame(frame: usize, total: usize) -> Result<(), Box<dyn std::error::Error>> {
    let swapl = global_swapl();

    // ── 点云（灰色，不区分地面/非地面） ──
    let cloud_stream = swapl.clouds_out.lock().await;
    if let Some(cloud) = cloud_stream.peek_latest() {
        println!("  帧 {}/{} | 点云: {} points", frame + 1, total, cloud.len());
        send_colored_cloud(&cloud, "white", 1_000_000).await?;
    } else {
        println!("  帧 {}/{} | 点云: 无数据", frame + 1, total);
    }
    drop(cloud_stream);

    // ── 跟踪目标（框 + 标签 + 颜色） ──
    let target_stream = swapl.targets.lock().await;
    if let Some(targets) = target_stream.peek_latest() {
        println!("  帧 {}/{} | 目标: {} 个", frame + 1, total, targets.len());
        for t in targets.iter() {
            let dyn_str = if t.is_dynamic { "动态" } else { "静态" };
            println!("    id={} type={} class={} {} speed={:.2}",
                t.id, t.class_type, t.classification, dyn_str, t.speed);
        }
        send_targets(&targets).await?;
        send_speed_arrows(&targets).await?;
    } else {
        println!("  帧 {}/{} | 目标: 无", frame + 1, total);
    }
    drop(target_stream);

    // ── FRAMEEND：告知 redra 当前帧结束 ──
    let mut unit = generate_unit();
    unit.command = Some(ExCommand { u_command: CommandType::Frameend as i32 });
    unit.send().await?;

    Ok(())
}

/// 发送带颜色的点云
async fn send_colored_cloud(points: &[[f32; 3]], color: &str, base_id: u64) -> Result<(), Box<dyn std::error::Error>> {
    if points.is_empty() { return Ok(()); }
    let mut unit = generate_unit();
    for (i, p) in points.iter().enumerate() {
        let eid = base_id + (i as u64) * 4;
        unit.objects.extend(vec![
            ExObject::from(eid),
            ExObject::from(ExMesh::from(Point { x: 0.0, y: 0.0, z: 0.0 })),
            ExObject::from(ExTransform {
                x: p[0], y: p[1], z: p[2],
                rx: 0.0, ry: 0.0, rz: 0.0,
                sx: 1.0, sy: 1.0, sz: 1.0,
            }),
            ExObject { u_object: Some(ex_object::UObject::MaterialId(color.to_string())) },
        ]);
    }
    unit.send().await?;
    Ok(())
}

/// 发送目标（包围盒 + 标签 + 颜色）
async fn send_targets(targets: &[Target]) -> Result<(), Box<dyn std::error::Error>> {
    const BASE_ID: u64 = 2_000_000;

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

        // 颜色：地面→蓝，person(融合确认)→青，动态→红，静态→绿，可移动→黄
        let is_ground = target.class_type == "ground";
        let is_person = target.class_type == "person";
        let material_id = if is_ground {
            "blue"
        } else if is_person {
            "cyan"
        } else {
            match target.classification.as_str() {
                "dynamic" => "red",
                "static" => "green",
                "movable" => "yellow",
                _ => "white",
            }
        };

        let mut unit = generate_unit();
        unit.objects.extend(vec![
            ExObject::from(entity_id),
            ExObject::from(ExMesh::from(Cube { vertices: points })),
            ExObject::from(ExTransform {
                x: cx, y: cy, z: cz,
                rx: 0.0, ry: 0.0, rz: 0.0,
                sx: 1.0, sy: 1.0, sz: 1.0,
            }),
            ExObject { u_object: Some(ex_object::UObject::MaterialId(material_id.to_string())) },
            // 标签：id | classification | class_type | speed
            ExObject::from(Tag::new(format!(
                "{} | {} | {:.1}m/s",
                target.id, target.classification, target.speed,
            )).with_offset(ExTransform {
                x: cx, y: cy + 0.5, z: cz,
                rx: 0.0, ry: 0.0, rz: 0.0,
                sx: 1.0, sy: 1.0, sz: 1.0,
            })),
        ]);
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
