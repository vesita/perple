//! 全流程测试 example
//!
//! 运行完整的 Perple 检测跟踪管线，逐帧输出统计信息并保存为 .rdra 文件。
//!
//! 用法：
//!   cargo run --example pipeline_test
//!   cargo run --example pipeline_test -- --frames 50

use std::time::Instant;

use perple::cloud::core::Lidar;
use perple::cloud::output::CldBud;
use perple::color::core::Camera;
use perple::fuse::Fuse;
use perple::optional::data_loader::DataLoader;
use perple::swapl::global_swapl;
use perple::tracker::core::Tracker;
use perple::tracker::output::Target;

use log::info;
use redra_client::{RdraWriter, ShapeBuilder};

// ─── .rdra 输出 ───────────────────────────────────────────────────────────
fn write_targets(writer: &mut RdraWriter, targets: &[Target]) {
    for (i, target) in targets.iter().enumerate() {
        let verts: Vec<(f32, f32, f32)> = target.the_box.vertices().iter()
            .map(|v| (v.x, v.y, v.z))
            .collect();
        let tag = format!("{} | {} | {:.1}m/s", target.id, target.classification, target.speed);
        writer.spawn(
            ShapeBuilder::cube(verts)
                .id(2_000_000 + i as u64 * 4)
                .material("glass")
                .tag(tag)
        );
    }
}

fn write_cldbuds(writer: &mut RdraWriter, buds: &[CldBud]) {
    for (i, bud) in buds.iter().enumerate() {
        let verts: Vec<(f32, f32, f32)> = bud.the_box.vertices().iter()
            .map(|v| (v.x, v.y, v.z))
            .collect();
        let tag = format!("{} | {:.2}", bud.class_name, bud.confidence);
        writer.spawn(
            ShapeBuilder::cube(verts)
                .id(1_000_000 + i as u64 * 4)
                .material("point_cloud")
                .tag(tag)
        );
    }
}

// ─── 统计信息 ──────────────────────────────────────────────────────────────
fn print_stats(frame: usize, total: usize, elapsed_ms: f64,
               n_ground: usize, n_cloud: usize, n_clusters: usize,
               n_targets: usize, targets: &[Target]) {
    println!("━━━ 帧 {:3}/{} ━━━ 耗时 {:5.0}ms ━━━", frame + 1, total, elapsed_ms);
    println!("  地面: {:5}点 | 非地面: {:5}点 | 聚类: {:3}个", n_ground, n_cloud, n_clusters);
    println!("  跟踪目标: {} 个", n_targets);

    let mut n_moving = 0usize;
    let mut n_static = 0usize;
    let mut n_movable = 0usize;
    let mut n_floating = 0usize;
    let mut total_speed = 0.0f32;

    for t in targets {
        match t.classification.as_str() {
            "moving" => n_moving += 1,
            "static" => n_static += 1,
            "movable" => n_movable += 1,
            "floating" => n_floating += 1,
            _ => {}
        }
        total_speed += t.speed;
        println!("    id={:3} | {:8} | {:>8} | {:.2}m/s",
            t.id, t.classification, t.class_type, t.speed);
    }

    let avg_speed = if targets.is_empty() { 0.0 } else { total_speed / targets.len() as f32 };
    println!("  分类: {} 运动中 / {} 静态 / {} 可运动 / {} 待定 | 平均速度 {:.2}m/s",
        n_moving, n_static, n_movable, n_floating, avg_speed);
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();

    // ─── 解析命令行 ──────────────────────────────────────────────────────
    let args: Vec<String> = std::env::args().collect();
    let n_frames_limit: Option<usize> = args.iter()
        .position(|a| a == "--frames")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok());

    // ─── 检查 YOLO 模型 ─────────────────────────────────────────────────────
    let config = perple::config::fixif();
    let model_path = &config.model_path;
    if std::path::Path::new(model_path).exists() {
        info!("YOLO 模型存在（{}）", model_path);
    } else {
        eprintln!("YOLO 模型不存在（{}），无法启动 Camera 模块", model_path);
        std::process::exit(1);
    }

    // ─── 初始化数据加载器 ─────────────────────────────────────────────────
    let mut data_loader = DataLoader::new("./data/test".to_string());
    data_loader.load().await?;
    let n_frames = n_frames_limit
        .map(|n| n.min(data_loader.frame_count()))
        .unwrap_or(data_loader.frame_count());
    info!("数据目录：{} 帧可用（已预加载）", n_frames);

    // ─── 初始化模块 ──────────────────────────────────────────────────────
    let mut lidar = Lidar::new();
    let mut camera = Camera::new();
    let mut fuse = Fuse::new();
    let mut tracker = Tracker::new();
    let mut writer = RdraWriter::new();
    let mut cluster_writer = RdraWriter::new();

    // ─── 两级流水 ────────────────────────────────────────────────────────
    let total_start = Instant::now();

    if !data_loader.load_next().await? {
        info!("数据为空");
        return Ok(());
    }
    if n_frames > 1 {
        data_loader.load_next().await?;
    }

    let mut l_handle = Some(tokio::spawn(async move {
        let _ = lidar.act().await;
        lidar
    }));
    let mut c_handle = Some(tokio::spawn(async move {
        let _ = camera.act().await;
        camera
    }));

    for i in 0..n_frames {
        let (l_res, c_res) = tokio::join!(
            l_handle.take().unwrap(),
            c_handle.take().unwrap(),
        );
        lidar = l_res.unwrap();
        camera = c_res.unwrap();

        let frame_start = Instant::now();

        fuse.act().await;
        let t2 = frame_start.elapsed().as_secs_f64() * 1000.0;

        // ─── 聚类输出（跟踪前） ──────────────────────────────────────────
        {
            let swapl = global_swapl();
            let buds_guard = swapl.cld_buds_raw.lock().await;
            if let Some(buds) = buds_guard.peek_latest() {
                write_cldbuds(&mut cluster_writer, &buds);
            }
        }
        cluster_writer.end_frame();

        if i + 2 < n_frames {
            data_loader.load_next().await?;
        }

        if i + 1 < n_frames {
            l_handle = Some(tokio::spawn(async move {
                let _ = lidar.act().await;
                lidar
            }));
            c_handle = Some(tokio::spawn(async move {
                let _ = camera.act().await;
                camera
            }));
        }

        let _ = tracker.run().await;
        let t3 = frame_start.elapsed().as_secs_f64() * 1000.0;

        // ─── 读取输出 ────────────────────────────────────────────────────
        let swapl = global_swapl();
        let (n_ground_pts, n_filtered_pts, n_cld_buds) = {
            let cloud = swapl.clouds_out.lock().await;
            let n_raw = cloud.peek_latest().as_ref().map(|c| c.len()).unwrap_or(0);
            let filtered = swapl.clouds_filtered.lock().await;
            let n_filt = filtered.peek_latest().as_ref().map(|c| c.len()).unwrap_or(0);
            let buds = swapl.cld_buds_raw.lock().await;
            let n_buds = buds.peek_latest().as_ref().map(|c| c.len()).unwrap_or(0);
            (n_raw.saturating_sub(n_filt), n_filt, n_buds)
        };

        let targets: Vec<Target> = {
            let mut t_stream = swapl.targets.lock().await;
            t_stream.read().unwrap_or_default()
        };
        let t4 = frame_start.elapsed().as_secs_f64() * 1000.0;

        if i % 50 == 0 || i == n_frames - 1 {
            println!("  ⏱  fuse={:.0}  tracker={:.0}  overhead={:.0}  total={:.0}ms",
                t2, t3 - t2, t4 - t3, t4);
        }

        print_stats(i, n_frames, t4, n_ground_pts, n_filtered_pts, n_cld_buds,
                    targets.len(), &targets);

        // ─── 写入 .rdra ──────────────────────────────────────────────────
        write_targets(&mut writer, &targets);
        writer.end_frame();
    }

    let total_elapsed = total_start.elapsed().as_secs_f64();
    println!();
    println!("══════════════════════════════════════════");
    println!("全流程测试完成 | {} 帧 | 总耗时 {:.1}s | 平均 {:.0}ms/帧",
        n_frames, total_elapsed, total_elapsed * 1000.0 / n_frames as f64);
    println!("══════════════════════════════════════════");

    // ─── 保存 .rdra 文件 ──────────────────────────────────────────────────
    cluster_writer.save("output/cluster_result.rdra")?;
    writer.save("output/tracker_result.rdra")?;

    Ok(())
}
