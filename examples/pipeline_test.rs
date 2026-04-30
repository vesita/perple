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
                "dynamic" => "red",
                "static" => "green",
                "movable" => "yellow",
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

    let mut n_dynamic = 0usize;
    let mut n_static = 0usize;
    let mut n_movable = 0usize;
    let mut total_speed = 0.0f32;
    let mut speed_strs = Vec::new();

    for t in targets {
        match t.classification.as_str() {
            "dynamic" => n_dynamic += 1,
            "static" => n_static += 1,
            "movable" => n_movable += 1,
            _ => {}
        }
        total_speed += t.speed;
        speed_strs.push(format!("    id={:3} | {:8} | {:>6} | {:>6} | speed={:.2}m/s",
            t.id, t.classification, t.class_type, if t.is_dynamic { "DYNAMIC" } else { "STATIC" }, t.speed));
    }

    let avg_speed = if targets.is_empty() { 0.0 } else { total_speed / targets.len() as f32 };
    println!("  分类: {} 动态 / {} 静态 / {} 可移动 | 平均速度 {:.2}m/s",
        n_dynamic, n_static, n_movable, avg_speed);

    for s in &speed_strs {
        println!("{}", s);
    }

    // ─── CSV 日志 ──────────────────────────────────────────────────────
    if let Some(f) = csv_writer {
        writeln!(f, "{},{},{},{},{},{:.4},{},{},{}",
            frame + 1, n_targets, n_dynamic, n_static, n_movable, avg_speed,
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
        writeln!(f, "frame,targets,dynamic,static,movable,avg_speed,ground_pts,cloud_pts,clusters").ok();
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
    data_loader.set_frame_limit(n_frames);
    info!("加载数据...");
    let load_start = Instant::now();
    let _ = data_loader.load().await;
    info!("数据加载完成，耗时 {}ms", load_start.elapsed().as_millis());

    // ─── 初始化模块（直接值，无 Arc/Mutex 开销） ───────────────────────────
    let mut lidar = Lidar::new();
    let mut camera = Camera::new();
    let mut fuse = Fuse::new();
    let mut tracker = Tracker::new();

    // ─── 逐帧处理 ─────────────────────────────────────────────────────────
    let total_start = Instant::now();

    for i in 0..n_frames {
        let frame_start = Instant::now();

        // Step 1: 并行 2D 检测 + 3D 聚类
        let _ = tokio::join!(lidar.act(), camera.act());

        // 读取处理结果统计
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

        // Step 2: 融合（需要 2D + 3D 都完成）
        fuse.act().await;

        // Step 3: 跟踪（融合完成后）
        let _ = tracker.run().await;

        let elapsed = frame_start.elapsed().as_secs_f64() * 1000.0;

        // ─── 读取输出并统计 ──────────────────────────────────────────────
        let targets: Vec<Target> = {
            let mut t_stream = swapl.targets.lock().await;
            t_stream.read().unwrap_or_default()
        };

        print_stats(i, n_frames, elapsed, n_ground_pts, n_filtered_pts, n_cld_buds,
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
