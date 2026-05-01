//! 全流程测试 example
//!
//! 运行完整的 Perple 检测跟踪管线，逐帧输出统计信息。
//! 如果 YOLO 模型存在则自动启用 Camera + Fuse，否则仅运行 Lidar → Tracker。
//!
//! 用法：
//!   cargo run --example pipeline_test
//!   cargo run --example pipeline_test -- --redra   （启用 redra 可视化）
//!   cargo run --example pipeline_test -- --frames 20

use std::time::Instant;

use perple::cloud::core::Lidar;
use perple::color::core::Camera;
use perple::fuse::Fuse;
use perple::optional::data_loader::DataLoader;
use perple::swapl::global_swapl;
use perple::tracker::core::Tracker;
use perple::tracker::output::Target;

use log::info;

// ─── redra 可视化 ─────────────────────────────────────────────────────────
mod viz {
    use expto::rdmp::auto::unit::generate_unit;
    use expto::rdmp::proto::command::{CommandType, ExCommand};
    use expto::rdmp::*;
    use perple::tracker::output::Target;
    use redra_client::*;

    pub async fn send_cloud(cloud: &[[f32; 3]], _frame: usize, _total: usize) -> Result<(), Box<dyn std::error::Error>> {
        if cloud.is_empty() {
            return Ok(());
        }
        let mut unit = generate_unit();
        for (i, p) in cloud.iter().enumerate() {
            let eid = 1_000_000 + (i as u64) * 4;
            unit.objects.extend(vec![
                ExObject::from(eid),
                ExObject::from(ExMesh::from(Point { x: 0.0, y: 0.0, z: 0.0 })),
                ExObject::from(ExTransform {
                    x: p[0], y: p[1], z: p[2],
                    rx: 0.0, ry: 0.0, rz: 0.0,
                    sx: 1.0, sy: 1.0, sz: 1.0,
                }),
                ExObject { u_object: Some(ex_object::UObject::MaterialId("white".to_string())) },
            ]);
        }
        unit.send().await?;
        Ok(())
    }

    pub async fn send_targets(targets: &[Target]) -> Result<(), Box<dyn std::error::Error>> {
        const BASE_ID: u64 = 2_000_000;
        for (i, target) in targets.iter().enumerate() {
            let entity_id = BASE_ID + (i as u64) * 4;
            let verts = target.the_box.vertices();
            let points: Vec<Point> = verts.iter()
                .map(|v| Point { x: v.x, y: v.y, z: v.z })
                .collect();

            let mut min = [f32::MAX, f32::MAX, f32::MAX];
            let mut max = [f32::MIN, f32::MIN, f32::MIN];
            for p in &points {
                min[0] = min[0].min(p.x); min[1] = min[1].min(p.y); min[2] = min[2].min(p.z);
                max[0] = max[0].max(p.x); max[1] = max[1].max(p.y); max[2] = max[2].max(p.z);
            }
            let cx = (min[0] + max[0]) / 2.0;
            let cy = (min[1] + max[1]) / 2.0;
            let cz = (min[2] + max[2]) / 2.0;

            let color = match target.classification.as_str() {
                "moving" => "red",
                "static" => "green",
                "movable" => "yellow",
                "floating" => "blue",
                _ => "white",
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
                ExObject { u_object: Some(ex_object::UObject::MaterialId(color.to_string())) },
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

    pub async fn send_frameend() -> Result<(), Box<dyn std::error::Error>> {
        let mut unit = generate_unit();
        unit.command = Some(ExCommand { u_command: CommandType::Frameend as i32 });
        unit.send().await?;
        Ok(())
    }
}

// ─── 统计信息 ──────────────────────────────────────────────────────────────
fn print_stats(frame: usize, total: usize, elapsed_ms: f64,
               n_ground: usize, n_cloud: usize, n_clusters: usize,
               n_targets: usize, targets: &[Target],
               csv_writer: &mut Option<std::fs::File>,
               csv_detail: &mut Option<std::fs::File>) {
    use std::io::Write;

    println!("━━━ 帧 {:3}/{} ━━━ 耗时 {:5.0}ms ━━━", frame + 1, total, elapsed_ms);
    println!("  地面: {:5}点 | 非地面: {:5}点 | 聚类: {:3}个", n_ground, n_cloud, n_clusters);
    println!("  跟踪目标: {} 个", n_targets);

    let mut n_moving = 0usize;
    let mut n_static = 0usize;
    let mut n_movable = 0usize;
    let mut n_floating = 0usize;
    let mut total_speed = 0.0f32;
    let mut speed_strs = Vec::new();

    for t in targets {
        match t.classification.as_str() {
            "moving" => n_moving += 1,
            "static" => n_static += 1,
            "movable" => n_movable += 1,
            "floating" => n_floating += 1,
            _ => {}
        }
        total_speed += t.speed;
        speed_strs.push(format!("    id={:3} | {:8} | {:>8} | {:.2}m/s",
            t.id, t.classification, t.class_type, t.speed));
    }

    let avg_speed = if targets.is_empty() { 0.0 } else { total_speed / targets.len() as f32 };
    println!("  分类: {} 运动中 / {} 静态 / {} 可运动 / {} 待定 | 平均速度 {:.2}m/s",
        n_moving, n_static, n_movable, n_floating, avg_speed);

    for s in &speed_strs {
        println!("{}", s);
    }

    // ─── CSV 日志 ──────────────────────────────────────────────────────
    if let Some(f) = csv_writer {
        writeln!(f, "{},{},{},{},{},{:.4},{},{},{}",
            frame + 1, n_targets, n_moving, n_static, n_movable, avg_speed,
            n_ground, n_cloud, n_clusters).ok();
    }
    if let Some(f) = csv_detail {
        for t in targets {
            writeln!(f, "{},{},{},{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4}",
                frame + 1, t.id, t.classification, t.class_type, t.speed,
                t.velocity[0], t.velocity[1], t.velocity[2],
                t.the_box.length, t.the_box.width, t.the_box.height).ok();
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();

    // ─── 解析命令行 ──────────────────────────────────────────────────────
    let args: Vec<String> = std::env::args().collect();
    if args.contains(&"--help".to_string()) || args.contains(&"-h".to_string()) {
        println!("全流程测试 example");
        println!("  cargo run --example pipeline_test                           (默认全部帧)");
        println!("  cargo run --example pipeline_test -- --frames 50            (指定帧数)");
        println!("  cargo run --example pipeline_test -- --csv result.csv       (输出 CSV 日志)");
        println!("  cargo run --example pipeline_test -- --redra                (启用 redra 可视化)");
        return Ok(());
    }
    let use_redra = args.contains(&"--redra".to_string());
    let csv_path = args.iter()
        .position(|a| a == "--csv")
        .and_then(|i| args.get(i + 1));
    let n_frames: Option<usize> = args.iter()
        .position(|a| a == "--frames")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok());
    let total_available = std::fs::read_dir("./data/test/lidar")
        .map(|e| e.count()).unwrap_or(0);
    let n_frames = n_frames.unwrap_or(total_available);

    // ─── 初始化 CSV 日志 ─────────────────────────────────────────────────
    let mut csv_writer: Option<std::fs::File> = csv_path
        .map(|p| std::fs::File::create(p).expect("无法创建 CSV 文件"));
    if let Some(f) = csv_writer.as_mut() {
        use std::io::Write;
        writeln!(f, "frame,targets,moving,static,movable,avg_speed,ground_pts,cloud_pts,clusters").ok();
    }
    let mut csv_detail: Option<std::fs::File> = csv_path
        .map(|p| {
            let p = p.replace(".csv", "_detail.csv");
            std::fs::File::create(&p).expect("无法创建详细 CSV 文件")
        });
    if let Some(f) = csv_detail.as_mut() {
        use std::io::Write;
        writeln!(f, "frame,id,classification,class_type,speed,vx,vy,vz,length,width,height").ok();
    }

    info!("Perple 全流程测试 | {} 帧 | redra={}", n_frames, use_redra);

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
    // 预加载：一次性将所有数据读入内存（53s 一次成本，之后无 I/O）
    data_loader.load().await?;
    info!("数据目录：{} 帧可用（已预加载）", n_frames);

    // ─── 初始化模块（直接值，无 Arc/Mutex 开销） ───────────────────────────
    let mut lidar = Lidar::new();
    let mut camera = Camera::new();
    let mut fuse = Fuse::new();
    let mut tracker = Tracker::new();

    // ─── 两级流水：帧 i+1 的 lidar|cam 与帧 i 的 fuse+tracker 并行 ────────
    //
    // 时序:
    //   pre-load(0), pre-load(1) → spawn lidar|cam(0)
    //   Frame 0: join lidar|cam(0) → pre-load(2) → spawn lidar|cam(1) → fuse→tracker
    //   Frame 1: join lidar|cam(1) → pre-load(3) → spawn lidar|cam(2) → fuse→tracker
    //   ...
    // 稳态: 帧耗时 = max(lidar|cam, fuse+tracker) ≈ ~40ms (25 FPS)
    // ──────────────────────────────────────────────────────────────────────
    let total_start = Instant::now();

    // 预加载帧 0 和 1
    if !data_loader.load_next().await? {
        info!("数据为空");
        return Ok(());
    }
    if n_frames > 1 {
        data_loader.load_next().await?;
    }

    // 启动 lidar|cam(0)
    let mut l_handle = Some(tokio::spawn(async move {
        let _ = lidar.act().await;
        lidar
    }));
    let mut c_handle = Some(tokio::spawn(async move {
        let _ = camera.act().await;
        camera
    }));

    for i in 0..n_frames {
        // Step 1: 收回 lidar|cam(i)（管道化下等待时间 ~0）
        let (l_res, c_res) = tokio::join!(
            l_handle.take().unwrap(),
            c_handle.take().unwrap(),
        );
        lidar = l_res.unwrap();
        camera = c_res.unwrap();

        let frame_start = Instant::now();

        // Step 2: 融合（先于 spawn，确保 peek_latest 读到当前帧数据）
        fuse.act().await;
        let t2 = frame_start.elapsed().as_secs_f64() * 1000.0;

        // 预加载帧 i+2（提前写入输入流）
        if i + 2 < n_frames {
            data_loader.load_next().await?;
        }

        // 启动 lidar|cam(i+1)（与 tracker 并行）
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

        // Step 3: 跟踪（与 lidar|cam(i+1) 并行）
        let _ = tracker.run().await;
        let t3 = frame_start.elapsed().as_secs_f64() * 1000.0;

        // ─── 读取输出并统计 ──────────────────────────────────────────────
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
            let d_fuse = t2;
            let d_tracker = t3 - t2;
            let d_overhead = t4 - t3;
            println!("  ⏱  fuse={d_fuse:.0}  tracker={d_tracker:.0}  overhead={d_overhead:.0}  total={t4:.0}ms");
        }

        print_stats(i, n_frames, t4, n_ground_pts, n_filtered_pts, n_cld_buds,
                    targets.len(), &targets,
                    &mut csv_writer, &mut csv_detail);

        // ─── redra 可视化 ───────────────────────────────────────────────
        if use_redra {
            if let Some(cloud) = swapl.clouds_out.lock().await.peek_latest() {
                viz::send_cloud(&cloud, i, n_frames).await?;
            }
            viz::send_targets(&targets).await?;
            viz::send_frameend().await?;
        }
    }

    let total_elapsed = total_start.elapsed().as_secs_f64();
    println!();
    println!("══════════════════════════════════════════");
    println!("全流程测试完成 | {} 帧 | 总耗时 {:.1}s | 平均 {:.0}ms/帧",
        n_frames, total_elapsed, total_elapsed * 1000.0 / n_frames as f64);
    println!("══════════════════════════════════════════");

    Ok(())
}
