use std::sync::{Arc, Mutex};

use perple::cloud::core::Lidar;
use perple::optional::data_loader::DataLoader;
use perple::tracker::core::Tracker;
use perple::swapl::global_swapl;
use perple::tracker::output::Target;
use perple::utils::rdra::FrameWriter;

use log::info;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();
    info!("Perple 检测流程可视化（14 帧）");

    let mut data_loader = DataLoader::new("./data/cloud".to_string());
    data_loader.set_frame_limit(14);
    info!("开始加载数据...");
    let load_start = std::time::Instant::now();
    let _ = data_loader.load().await;
    info!("数据加载完成，耗时 {}ms", load_start.elapsed().as_millis());

    let lidar = Arc::new(Mutex::new(Lidar::new()));
    let tracker = Arc::new(Mutex::new(Tracker::new()));
    let mut writer = FrameWriter::new("output/visualize.db")?;

    let n_frames = 14;
    for i in 0..n_frames {
        info!("─── 第 {}/{} 帧 ───", i + 1, n_frames);

        // ── LiDAR 处理 ──
        {
            let l = Arc::clone(&lidar);
            let handle = tokio::runtime::Handle::current();
            tokio::task::spawn_blocking(move || {
                let mut lidar = l.lock().unwrap();
                handle.block_on(lidar.act())
                    .map_err(|e| format!("Lidar 处理错误: {:?}", e))
            })
            .await
            .map_err(|e| format!("Lidar 任务失败: {:?}", e))??;
        }

        // ── 交换 DualBuf ──
        global_swapl().swap_pipeline();

        // ── 跟踪 ──
        {
            let t = Arc::clone(&tracker);
            let handle = tokio::runtime::Handle::current();
            tokio::task::spawn_blocking(move || {
                let mut trk = t.lock().unwrap();
                handle.block_on(trk.run())
                    .map_err(|e| format!("Tracker 处理错误: {:?}", e))
            })
            .await
            .map_err(|e| format!("Tracker 任务失败: {:?}", e))??;
        }

        // ── 写入帧 ──
        write_frame(&mut writer, i, n_frames).await?;
    }

    info!("所有帧处理完成，保存文件...");
    writer.save()?;
    info!("已保存到 output/visualize.db");
    Ok(())
}

async fn write_frame(writer: &mut FrameWriter, frame: usize, total: usize) -> Result<(), Box<dyn std::error::Error>> {
    writer.begin_frame(frame);

    let swapl = global_swapl();

    // ── 点云 ──
    let cloud_stream = swapl.clouds_out.lock().unwrap();
    if let Some(cloud) = cloud_stream.peek_latest() {
        println!("  帧 {}/{} | 点云: {} points", frame + 1, total, cloud.len());
        writer.write_cloud(&cloud, "point_cloud", 5000);
    }
    drop(cloud_stream);

    writer.end_frame();
    Ok(())
}

#[allow(unused)]
fn write_targets(writer: &mut FrameWriter, targets: &[Target]) {
    for target in targets.iter() {
        let tag = format!("{} | {} | {} | {:.1}m/s",
            target.id, target.class_type, target.classification, target.speed);
        writer.write_box(&target.the_box, "disabled", &tag);
    }
}

#[allow(unused)]
fn write_speed_arrows(writer: &mut FrameWriter, targets: &[Target]) {
    for target in targets {
        if target.speed > 0.5 {
            let center = target.the_box.center();
            let scale = (target.speed * 2.0).min(10.0);
            let dx = target.velocity[0] / target.speed * scale;
            let dy = target.velocity[1] / target.speed * scale;
            let dz = target.velocity[2] / target.speed * scale;

            writer.write_line(
                [center.x, center.y, center.z],
                [center.x + dx, center.y + dy, center.z + dz],
                "trajectory",
            );
        }
    }
}
