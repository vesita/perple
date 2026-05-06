use std::sync::{Arc, Mutex};

use perple::cloud::core::Lidar;
use perple::optional::data_loader::DataLoader;
use perple::tracker::core::Tracker;
use perple::swapl::global_swapl;
use perple::tracker::output::Target;

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
    let mut writer = RdraWriter::new();

    let n_frames = 14;
    for i in 0..n_frames {
        info!("─── 第 {}/{} 帧 ───", i + 1, n_frames);

        // ── LiDAR 处理 ──
        {
            let l = Arc::clone(&lidar);
            tokio::task::spawn_blocking(move || {
                let _ = l.lock().unwrap().act();
            })
            .await
            .map_err(|e| format!("Lidar 任务失败: {}", e))?;
        }

        // ── 跟踪 ──
        {
            let t = Arc::clone(&tracker);
            tokio::task::spawn_blocking(move || {
                let _ = t.lock().unwrap().run();
            })
            .await
            .map_err(|e| format!("Tracker 任务失败: {}", e))?;
        }

        // ── 写入帧 ──
        write_frame(&mut writer, i, n_frames).await?;
    }

    info!("所有帧处理完成，保存文件...");
    writer.save("output/visualize.rdra")?;
    info!("已保存到 output/visualize.rdra");
    Ok(())
}

async fn write_frame(writer: &mut RdraWriter, frame: usize, total: usize) -> Result<(), Box<dyn std::error::Error>> {
    writer.destroy_all();

    let swapl = global_swapl();

    // ── 点云（白色） ──
    let cloud_stream = swapl.clouds_out.lock().await;
    if let Some(cloud) = cloud_stream.peek_latest() {
        println!("  帧 {}/{} | 点云: {} points", frame + 1, total, cloud.len());
        let step = (cloud.len() / 5000).max(1);
        for (i, p) in cloud.iter().enumerate() {
            if i % step == 0 {
                writer.spawn(spawn_point(*p, "white").id(1_000_000 + i as u64 * 4));
            }
        }
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
        write_targets(writer, &targets);
        write_speed_arrows(writer, &targets);
    } else {
        println!("  帧 {}/{} | 目标: 无", frame + 1, total);
    }
    drop(target_stream);

    writer.end_frame();
    Ok(())
}

fn write_targets(writer: &mut RdraWriter, targets: &[Target]) {
    for (i, target) in targets.iter().enumerate() {
        let verts: Vec<(f32, f32, f32)> = target.the_box.vertices().iter()
            .map(|v| (v.x, v.y, v.z))
            .collect();

        let is_ground = target.class_type == "ground";
        let is_person = target.class_type == "person";
        let material = if is_ground {
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

        let tag = format!("{} | {} | {:.1}m/s", target.id, target.classification, target.speed);
        writer.spawn(
            spawn_cube(verts, material)
                .id(2_000_000 + i as u64 * 4)
                .tag(tag)
        );
    }
}

fn write_speed_arrows(writer: &mut RdraWriter, targets: &[Target]) {
    for target in targets {
        if target.speed > 0.5 {
            let center = target.the_box.center();
            let scale = (target.speed * 2.0).min(10.0);
            let dx = target.velocity[0] / target.speed * scale;
            let dy = target.velocity[1] / target.speed * scale;
            let dz = target.velocity[2] / target.speed * scale;

            writer.spawn(spawn_line(
                [center.x, center.y, center.z],
                [center.x + dx, center.y + dy, center.z + dz],
                "yellow",
            ));
        }
    }
}
