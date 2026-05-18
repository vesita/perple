/// Perple 检测跟踪管线入口
///
/// ROS 模式： cargo run --features ros1
/// 离线评测： cargo run [-- --frames N] [--skip N] [--output DIR]

// ─── ROS 模式 ────────────────────────────────────────────────────────────────

#[cfg(feature = "ros1")]
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(feature = "ros1")]
use std::sync::Arc;

#[cfg(feature = "ros1")]
use perple::ros_bridge::{RosBridge, RosBridgeConfig};
#[cfg(feature = "ros1")]
use perple::Perple;

#[cfg(feature = "ros1")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();

    log::info!("Perple ROS 节点启动中...");

    let mut bridge = RosBridge::new(RosBridgeConfig::default());
    bridge.init()?;
    log::info!("ROS 桥接初始化完成，发布器/订阅器已创建");

    let rt = tokio::runtime::Runtime::new()?;

    let _perple_handle = rt.spawn(async {
        let mut perple = Perple::new();
        if let Err(e) = perple.run().await {
            log::error!("Perple 管线错误: {}", e);
        }
    });
    log::info!("Perple 检测管线已启动");

    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    std::thread::spawn(move || {
        if let Ok(rt) = tokio::runtime::Runtime::new() {
            rt.block_on(async {
                tokio::signal::ctrl_c().await.ok();
            });
        }
        log::info!("收到 Ctrl+C，正在关闭...");
        r.store(false, Ordering::SeqCst);
    });

    std::thread::spawn(|| {
        rosrust::spin();
    });

    let rate = std::time::Duration::from_millis(50);
    log::info!("进入主循环 (20 Hz)");

    while running.load(Ordering::SeqCst) {
        bridge.publish_all();
        std::thread::sleep(rate);
    }

    drop(rt);
    log::info!("Perple ROS 节点已关闭");
    Ok(())
}

// ─── 离线评测模式（完整管线 + JSONL 输出供 Python 分析） ────────────────────

#[cfg(not(feature = "ros1"))]
use std::path::PathBuf;
#[cfg(not(feature = "ros1"))]
use std::time::{Instant, SystemTime, UNIX_EPOCH};

#[cfg(not(feature = "ros1"))]
use perple::cloud::core::Lidar;
#[cfg(not(feature = "ros1"))]
use perple::cloud::output::CldBud;
#[cfg(not(feature = "ros1"))]
use perple::color::core::Camera;
#[cfg(not(feature = "ros1"))]
use perple::fuse::Fuse;
#[cfg(not(feature = "ros1"))]
use perple::optional::data_loader::DataLoader;
#[cfg(not(feature = "ros1"))]
use perple::swapl::global_swapl;
#[cfg(not(feature = "ros1"))]
use perple::tracker::core::Tracker;
#[cfg(not(feature = "ros1"))]
use perple::tracker::output::Target;
#[cfg(not(feature = "ros1"))]
use perple::utils::rdra::FrameWriter;
#[cfg(not(feature = "ros1"))]
use perple::yolo_smooth::YoloSmoother;

#[cfg(not(feature = "ros1"))]
use log::info;

#[cfg(not(feature = "ros1"))]
const MAT_GROUND: &str = "ground";
#[cfg(not(feature = "ros1"))]
const MAT_WALL: &str = "wall";
#[cfg(not(feature = "ros1"))]
const MAT_PERSON: &str = "person";
#[cfg(not(feature = "ros1"))]
const MAT_OBSTACLE: &str = "disabled";

// ─── .rdra 写入辅助 ──────────────────────────────────────────────────────────

#[cfg(not(feature = "ros1"))]
fn write_rdra_buds(writer: &mut FrameWriter, buds: &[CldBud], mat: &str) {
    for bud in buds {
        let tag = format!("{} | {:.2}", bud.class_name, bud.confidence);
        writer.write_box(&bud.the_box, mat, &tag);
    }
}

#[cfg(not(feature = "ros1"))]
fn write_rdra_targets(writer: &mut FrameWriter, targets: &[Target]) {
    for t in targets {
        let tag = format!("{} | {} | {} | {:.1}m/s", t.id, t.classification, t.class_type, t.speed);
        let mat = match t.class_type.as_str() {
            "person" => MAT_PERSON,
            _ => MAT_OBSTACLE,
        };
        writer.write_box(&t.the_box, mat, &tag);
    }
}

// ─── JSONL 行构建（纯手动序列化，不修改任何现有类型） ─────────────────────

#[cfg(not(feature = "ros1"))]
fn make_jsonl_line(
    frame: usize,
    n_frames: usize,
    elapsed_frame_ms: f64,
    t_join: f64,
    t_fuse: f64,
    t_io: f64,
    t_tracker: f64,
    n_ground: usize,
    n_wall: usize,
    n_cloud_filtered: usize,
    n_clusters: usize,
    targets: &[Target],
) -> String {
    use serde_json::json;

    let n_targets = targets.len();
    let mut n_moving = 0usize;
    let mut n_static = 0usize;
    let mut n_movable = 0usize;
    let mut n_floating = 0usize;
    let mut n_person = 0usize;
    let mut total_speed = 0.0f32;

    let target_list: Vec<serde_json::Value> = targets.iter().map(|t| {
        match t.classification.as_str() {
            "moving" => n_moving += 1,
            "static" => n_static += 1,
            "movable" => n_movable += 1,
            "floating" => n_floating += 1,
            _ => {}
        }
        if t.class_type == "person" { n_person += 1; }
        total_speed += t.speed;

        let center = t.the_box.center();
        json!({
            "id": t.id,
            "x": (center.x * 100.0).round() / 100.0,
            "y": (center.y * 100.0).round() / 100.0,
            "z": (center.z * 100.0).round() / 100.0,
            "vx": (t.velocity[0] * 100.0).round() / 100.0,
            "vy": (t.velocity[1] * 100.0).round() / 100.0,
            "speed": (t.speed * 100.0).round() / 100.0,
            "classification": t.classification,
            "class_type": t.class_type,
        })
    }).collect();

    let avg_speed = if n_targets > 0 {
        (total_speed / n_targets as f32 * 100.0).round() / 100.0
    } else {
        0.0
    };

    json!({
        "frame": frame,
        "n_frames": n_frames,
        "elapsed_ms": (elapsed_frame_ms * 10.0).round() / 10.0,
        "stages_ms": {
            "join": (t_join * 10.0).round() / 10.0,
            "fuse": (t_fuse * 10.0).round() / 10.0,
            "io": (t_io * 10.0).round() / 10.0,
            "tracker": (t_tracker * 10.0).round() / 10.0,
        },
        "stats": {
            "n_ground": n_ground,
            "n_wall": n_wall,
            "n_cloud_filtered": n_cloud_filtered,
            "n_clusters": n_clusters,
            "n_targets": n_targets,
            "n_moving": n_moving,
            "n_static": n_static,
            "n_movable": n_movable,
            "n_floating": n_floating,
            "n_person": n_person,
            "avg_speed": avg_speed,
        },
        "targets": target_list,
    }).to_string()
}

// ─── 终端统计输出 ────────────────────────────────────────────────────────────

#[cfg(not(feature = "ros1"))]
fn print_stats(frame: usize, n_frames: usize, elapsed_ms: f64,
               n_cloud: usize, n_clusters: usize,
               n_targets: usize, targets: &[Target]) {
    println!("━━━ 帧 {:3}/{} ━━━ 耗时 {:5.0}ms ━━━", frame + 1, n_frames, elapsed_ms);
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

// ─── 离线主流程 ─────────────────────────────────────────────────────────────

#[cfg(not(feature = "ros1"))]
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
    let skip_frames: usize = args.iter()
        .position(|a| a == "--skip")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let out_prefix: String = args.iter()
        .position(|a| a == "--output")
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_default();

    // ─── 检查 YOLO 模型 ──────────────────────────────────────────────────
    let config = perple::config::fixif();
    let model_path = &config.model_path;
    if !std::path::Path::new(model_path).exists() {
        eprintln!("YOLO 模型不存在（{}），无法启动 Camera 模块", model_path);
        std::process::exit(1);
    }
    info!("YOLO 模型存在（{}）", model_path);
    info!("墙体策略: {} | 聚类策略: {} | 跟踪器已启用", config.wall_strategy, config.cluster.strategy);

    // ─── 初始化数据加载器 ────────────────────────────────────────────────
    let mut data_loader = DataLoader::new("./data/cloud".to_string());
    data_loader.load().await?;
    let total_available = data_loader.frame_count().saturating_sub(skip_frames);
    let n_frames = n_frames_limit
        .map(|n| n.min(total_available))
        .unwrap_or(total_available);
    info!("数据目录: {} 帧可用 → 将处理 {} 帧", data_loader.frame_count(), n_frames);

    if n_frames == 0 {
        info!("没有帧需要处理");
        return Ok(());
    }

    if skip_frames > 0 {
        info!("跳过前 {} 帧", skip_frames);
        for _ in 0..skip_frames { data_loader.load_next().await?; }
        let swapl = global_swapl();
        for _ in 0..skip_frames {
            swapl.clouds.lock().unwrap().read();
            swapl.colors.lock().unwrap().read();
        }
    }

    // ─── 初始化模块 ──────────────────────────────────────────────────────
    let mut lidar = Lidar::new();
    let mut camera = Camera::new();
    let mut fuse = Fuse::new();
    let mut tracker = Tracker::new();

    // ─── 输出目录 ────────────────────────────────────────────────────────
    let out_dir = if out_prefix.is_empty() {
        let secs = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
        PathBuf::from(format!("output/pipeline_{}", secs))
    } else {
        PathBuf::from(&out_prefix)
    };
    std::fs::create_dir_all(&out_dir)?;

    // .rdra 文件（供 Bevy/egui 可视化回放）
    let mut writer_ground = FrameWriter::new(out_dir.join("ground.db"))?;
    let mut writer_wall   = FrameWriter::new(out_dir.join("wall.db"))?;
    let mut writer_cluster = FrameWriter::new(out_dir.join("cluster.db"))?;
    let mut writer_tracker = FrameWriter::new(out_dir.join("tracker.db"))?;

    // JSONL 文件（供 Python 分析）
    let jsonl_path = out_dir.join("pipeline.jsonl");
    let mut jsonl_file = std::fs::File::create(&jsonl_path)?;
    use std::io::Write;

    // ─── 预处理: 加载前两帧 ──────────────────────────────────────────────
    if !data_loader.load_next().await? {
        info!("数据为空");
        return Ok(());
    }
    if n_frames > 1 {
        data_loader.load_next().await?;
    }

    let total_start = Instant::now();
    let mut yolo_smoother = YoloSmoother::new();

    // 启动第一帧的检测
    let mut l_handle = Some(tokio::spawn(async move {
        let _ = lidar.act().await;
        lidar
    }));
    let mut c_handle = Some(tokio::spawn(async move {
        let _ = camera.act().await;
        camera
    }));

    // ─── 主循环 ──────────────────────────────────────────────────────────
    for i in 0..n_frames {
        let iter_start = Instant::now();

        // ── 等待检测完成 ─────────────────────────────────────────────────
        let (l_res, c_res) = tokio::join!(
            l_handle.take().unwrap(),
            c_handle.take().unwrap(),
        );
        lidar = l_res.unwrap();
        camera = c_res.unwrap();
        let t_join = iter_start.elapsed().as_secs_f64() * 1000.0;

        // ── DualBuf 交换：检测阶段 → 后融合阶段 ─────────────────────────
        let swapl = global_swapl();
        swapl.swap_pipeline();
        // YOLO 标签平滑（在 Camera→Fuse 之间）
        yolo_smoother.smooth(&mut *swapl.clr_objs.consumer().lock().unwrap());

        // ── 提前启动下一帧检测（与后融合并行） ───────────────────────────
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

        // ── 后融合 ───────────────────────────────────────────────────────
        fuse.act().await;
        let t_fuse = iter_start.elapsed().as_secs_f64() * 1000.0;

        // ── 读取各语义流 ─────────────────────────────────────────────────
        let ground_buds: Vec<CldBud> = swapl.ground_buds.consumer().lock().unwrap().clone();
        let wall_buds:   Vec<CldBud> = swapl.wall_buds.consumer().lock().unwrap().clone();
        let cluster_buds: Vec<CldBud> = swapl.cld_buds_raw.consumer().lock().unwrap().clone();

        // ── 写入 .rdra ───────────────────────────────────────────────────
        writer_ground.begin_frame(i);
        write_rdra_buds(&mut writer_ground, &ground_buds, MAT_GROUND);
        writer_ground.end_frame();

        writer_wall.begin_frame(i);
        write_rdra_buds(&mut writer_wall, &wall_buds, MAT_WALL);
        writer_wall.end_frame();

        writer_cluster.begin_frame(i);
        write_rdra_buds(&mut writer_cluster, &cluster_buds, MAT_PERSON); // 用 person mat 显示聚类
        writer_cluster.end_frame();

        // ── 异步加载下一帧 ───────────────────────────────────────────────
        if i + 2 < n_frames {
            data_loader.load_next().await?;
        }

        let t_io = iter_start.elapsed().as_secs_f64() * 1000.0;

        // ── 跟踪 ─────────────────────────────────────────────────────────
        if let Err(e) = tracker.run().await {
            eprintln!("Tracker 错误：{:?}", e);
        }
        let t_tracker = iter_start.elapsed().as_secs_f64() * 1000.0;

        // ── 读取跟踪输出 ─────────────────────────────────────────────────
        let n_filtered_pts = swapl.clouds_filtered.consumer().lock().unwrap().len();
        let targets: Vec<Target> = swapl.targets.lock().unwrap().read().unwrap_or_default();
        let t_total = iter_start.elapsed().as_secs_f64() * 1000.0;

        // ── 输出终端统计 ─────────────────────────────────────────────────
        let n_clusters = cluster_buds.len();
        let n_targets = targets.len();

        if i % 20 == 0 || i == n_frames - 1 || i < 5 {
            println!("  join={:.0}  fuse={:.0}  io={:.0}  tracker={:.0}  total={:.0}ms",
                t_join, t_fuse - t_join, t_io - t_fuse, t_tracker - t_io, t_total);
        }

        print_stats(i, n_frames, t_total, n_filtered_pts, n_clusters, n_targets, &targets);

        // ── 写入 JSONL ──────────────────────────────────────────────────
        let json_line = make_jsonl_line(
            i, n_frames, t_total,
            t_join, t_fuse, t_io, t_tracker,
            ground_buds.len(), wall_buds.len(),
            n_filtered_pts, n_clusters,
            &targets,
        );
        writeln!(jsonl_file, "{}", json_line)?;

        // ── 写入 .rdra 跟踪结果 ──────────────────────────────────────────
        writer_tracker.begin_frame(i);
        write_rdra_targets(&mut writer_tracker, &targets);
        writer_tracker.end_frame();
    }

    // ─── 清理与总结 ──────────────────────────────────────────────────────
    writer_ground.save()?;
    writer_wall.save()?;
    writer_cluster.save()?;
    writer_tracker.save()?;

    let total_elapsed = total_start.elapsed().as_secs_f64();
    let avg_ms_per_frame = total_elapsed * 1000.0 / n_frames as f64;

    println!();
    println!("══════════════════════════════════════════");
    println!("全流程测试完成");
    println!("  帧数:     {} 帧", n_frames);
    println!("  总耗时:   {:.1}s", total_elapsed);
    println!("  平均:     {:.0}ms/帧 ({:.1} FPS)", avg_ms_per_frame, 1000.0 / avg_ms_per_frame);
    println!("  输出目录: {}", out_dir.display());
    println!("  数据文件: pipeline.jsonl, *.db (.rdra)");
    println!("══════════════════════════════════════════");

    Ok(())
}
