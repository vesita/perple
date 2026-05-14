//! 全流程测试 example
//!
//! 运行完整的 Perple 检测跟踪管线，逐帧输出统计信息并保存为 .rdra 文件。
//! 输出包含语义标签（ground / wall / person / obstacle）和运动标签（moving/static/movable/floating）。
//!
//! 用法：
//!   cargo run --example pipeline_test
//!   cargo run --example pipeline_test -- --frames 50

use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use perple::cloud::core::Lidar;
use perple::cloud::output::CldBud;
use perple::color::core::Camera;
use perple::fuse::Fuse;
use perple::optional::data_loader::DataLoader;
use perple::swapl::global_swapl;
use perple::tracker::core::Tracker;
use perple::tracker::output::Target;
use perple::utils::rdra::FrameWriter;

use log::info;

const MAT_GROUND: &str = "ground";
const MAT_WALL: &str = "wall";
const MAT_PERSON: &str = "person";
const MAT_OBSTACLE: &str = "disabled";

// ─── .rdra 输出 ───────────────────────────────────────────────────────────
fn write_targets(writer: &mut FrameWriter, buds: &[CldBud]) {
    // 目标输出：用 class_name 作语义标签
    for bud in buds.iter() {
        let mat = match bud.class_name.as_str() {
            "ground" => MAT_GROUND,
            "wall" => MAT_WALL,
            "person" => MAT_PERSON,
            _ => MAT_OBSTACLE,
        };
        let tag = format!("{} | {:.2}", bud.class_name, bud.confidence);
        writer.write_box(&bud.the_box, mat, &tag);
    }
}

fn write_tracker_targets(writer: &mut FrameWriter, targets: &[Target]) {
    for target in targets.iter() {
        let tag = format!("{} | {} | {} | {:.1}m/s", target.id, target.classification, target.class_type, target.speed);
        let mat = match target.class_type.as_str() {
            "person" => MAT_PERSON,
            _ => MAT_OBSTACLE,
        };
        writer.write_box(&target.the_box, mat, &tag);
    }
}

fn write_semantic_buds(writer: &mut FrameWriter, buds: &[CldBud], mat: &str) {
    for bud in buds.iter() {
        let tag = format!("{} | {:.2}", bud.class_name, bud.confidence);
        writer.write_box(&bud.the_box, mat, &tag);
    }
}

// ─── 统计信息 ──────────────────────────────────────────────────────────────
fn print_stats(frame: usize, total: usize, elapsed_ms: f64,
               n_cloud: usize, n_clusters: usize,
               n_targets: usize, targets: &[Target]) {
    println!("━━━ 帧 {:3}/{} ━━━ 耗时 {:5.0}ms ━━━", frame + 1, total, elapsed_ms);
    println!("  非地面点: {:5} | 聚类: {:3}个", n_cloud, n_clusters);
    println!("  跟踪目标: {} 个", n_targets);

    let mut n_moving = 0usize;
    let mut n_static = 0usize;
    let mut n_movable = 0usize;
    let mut n_floating = 0usize;
    let mut n_person = 0usize;
    let mut total_speed = 0.0f32;

    for t in targets {
        match t.classification.as_str() {
            "moving" => n_moving += 1,
            "static" => n_static += 1,
            "movable" => n_movable += 1,
            "floating" => n_floating += 1,
            _ => {}
        }
        if t.class_type == "person" { n_person += 1; }
        total_speed += t.speed;
        println!("    id={:3} | {:8} | {:>8} | {:.2}m/s",
            t.id, t.classification, t.class_type, t.speed);
    }

    let avg_speed = if targets.is_empty() { 0.0 } else { total_speed / targets.len() as f32 };
    println!("  {}运动中/{}静态/{}可运动/{}待定 | 行人:{} | 均速{:.2}m/s",
        n_moving, n_static, n_movable, n_floating, n_person, avg_speed);
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
    let _wall_strategy: Option<String> = args.iter()
        .position(|a| a == "--wall" || a.starts_with("--wall="))
        .and_then(|i| {
            if args[i] == "--wall" {
                args.get(i + 1).cloned()
            } else {
                args[i].split_once('=').map(|(_, v)| v.to_string())
            }
        });

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
    let mut data_loader = DataLoader::new("./data/cloud".to_string());
    data_loader.load().await?;
    let n_frames = n_frames_limit
        .map(|n| n.min(data_loader.frame_count()))
        .unwrap_or(data_loader.frame_count());
    info!("数据目录：{} 帧可用（已预加载）", n_frames);

    // ─── 初始化模块 ──────────────────────────────────────────────────────
    let mut lidar = {
        let cfg = perple::config::fixif();
        info!("墙体策略: {} (config 默认)", cfg.wall_strategy);
        Lidar::new()
    };
    let mut camera = Camera::new();
    let mut fuse = Fuse::new();
    let mut tracker = Tracker::new();

    // 四个独立输出流（每次运行独立子目录）
    let out_dir = {
        let secs = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
        PathBuf::from(format!("output/pipeline_test_{}", secs))
    };
    let mut writer_ground = FrameWriter::new(out_dir.join("ground_result.db"))?;
    let mut writer_wall = FrameWriter::new(out_dir.join("wall_result.db"))?;
    let mut writer_cluster = FrameWriter::new(out_dir.join("cluster_result.db"))?;
    let mut writer_tracker = FrameWriter::new(out_dir.join("tracker_result.db"))?;

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
        let iter_start = Instant::now();

        let (l_res, c_res) = tokio::join!(
            l_handle.take().unwrap(),
            c_handle.take().unwrap(),
        );
        lidar = l_res.unwrap();
        camera = c_res.unwrap();

        let t_join = iter_start.elapsed().as_secs_f64() * 1000.0;

        // ─── 交换 DualBuf：检测阶段 → 后融合阶段 ────────────────────────
        let swapl = global_swapl();
        swapl.cld_buds_raw.swap();
        swapl.clr_objs.swap();
        swapl.clouds_filtered.swap();

        // ─── 提前启动下一帧检测（与当前帧后融合并行） ──────────────────
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

        // ─── 后融合（与下一帧检测并行执行） ─────────────────────────────
        fuse.act().await;
        let t_fuse = iter_start.elapsed().as_secs_f64() * 1000.0;

        // ─── 读取各语义流 ────────────────────────────────────────────────
        let ground_buds: Vec<CldBud> = {
            let gb = swapl.ground_buds.lock().unwrap();
            gb.peek_latest().unwrap_or_default()
        };
        let wall_buds: Vec<CldBud> = {
            let wb = swapl.wall_buds.lock().unwrap();
            wb.peek_latest().unwrap_or_default()
        };
        let cluster_buds: Vec<CldBud> = swapl.cld_buds_raw.consumer().lock().unwrap().clone();

        // ─── 写入 .rdra ──────────────────────────────────────────────────
        writer_ground.begin_frame(i);
        write_semantic_buds(&mut writer_ground, &ground_buds, MAT_GROUND);
        writer_ground.end_frame();

        writer_wall.begin_frame(i);
        write_semantic_buds(&mut writer_wall, &wall_buds, MAT_WALL);
        writer_wall.end_frame();

        writer_cluster.begin_frame(i);
        write_targets(&mut writer_cluster, &cluster_buds);
        writer_cluster.end_frame();

        if i + 2 < n_frames {
            data_loader.load_next().await?;
        }

        let _ = tracker.run().await;
        let t_tracker = iter_start.elapsed().as_secs_f64() * 1000.0;

        // ─── 读取跟踪输出 ────────────────────────────────────────────────
        let n_filtered_pts = swapl.clouds_filtered.consumer().lock().unwrap().len();

        let targets: Vec<Target> = {
            let mut t_stream = swapl.targets.lock().unwrap();
            t_stream.read().unwrap_or_default()
        };
        let t_end = iter_start.elapsed().as_secs_f64() * 1000.0;

        if i % 50 == 0 || i == n_frames - 1 || i < 5 {
            println!("  join={:.0}  fuse={:.0}  tracker={:.0}  seq={:.0}  iter={:.0}ms",
                t_join, t_fuse - t_join, t_tracker - t_fuse, t_end - t_tracker, t_end);
        }

        print_stats(i, n_frames, t_end, n_filtered_pts, cluster_buds.len(),
                    targets.len(), &targets);

        writer_tracker.begin_frame(i);
        write_tracker_targets(&mut writer_tracker, &targets);
        writer_tracker.end_frame();
    }

    let total_elapsed = total_start.elapsed().as_secs_f64();
    println!();
    println!("══════════════════════════════════════════");
    println!("全流程测试完成 | {} 帧 | 总耗时 {:.1}s | 平均 {:.0}ms/帧",
        n_frames, total_elapsed, total_elapsed * 1000.0 / n_frames as f64);
    println!("══════════════════════════════════════════");

    // ─── VACUUM 压缩 ─────────────────────────────────────────────────────
    writer_ground.save()?;
    writer_wall.save()?;
    writer_cluster.save()?;
    writer_tracker.save()?;
    println!("输出保存至 {}/ 目录", out_dir.display());

    Ok(())
}
