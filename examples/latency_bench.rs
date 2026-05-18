//! 同帧跟踪延迟测试 (End-to-end per-frame latency)
//!
//! 测量从一帧原始数据到跟踪器输出的端到端耗时。
//! 包含：检测 (LiDAR + Camera) → 融合 → 跟踪 全链路。
//!
//! 模式:
//!   默认（并行）: 检测和跟踪 async 流水线重叠，测吞吐
//!   --serial:     逐帧串行，检测→后处理→融合→跟踪全部完成才启动下一帧，测真实延迟
//!
//! 用法:
//!   cargo run --release --example latency_bench                    # 并行（吞吐）
//!   cargo run --release --example latency_bench -- --serial        # 串行（真延迟）
//!   cargo run --release --example latency_bench -- --frames 50
//!   cargo run --release --example latency_bench -- --serial --frames 50

use std::path::PathBuf;
use std::time::Instant;

use perple::cloud::core::Lidar;
use perple::color::core::Camera;
use perple::fuse::Fuse;
use perple::optional::data_loader::DataLoader;
use perple::swapl::global_swapl;
use perple::tracker::core::Tracker;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Warn)
        .init();

    // ─── CLI ───────────────────────────────────────────────────────────────
    let args: Vec<String> = std::env::args().collect();
    let n_frames: usize = args.iter()
        .position(|a| a == "--frames")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(408);
    let out_prefix: String = args.iter()
        .position(|a| a == "--output")
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_default();
    let serial: bool = args.iter().any(|a| a == "--serial");

    // ─── 检查 YOLO 模型 ──────────────────────────────────────────────────
    let config = perple::config::fixif();
    if !std::path::Path::new(&config.model_path).exists() {
        eprintln!("YOLO 模型不存在（{}）", config.model_path);
        std::process::exit(1);
    }

    // ─── 初始化 ───────────────────────────────────────────────────────────
    let mut lidar = Lidar::new();
    let mut camera = Camera::new();
    let mut fuse = Fuse::new();
    let mut tracker = Tracker::new();

    let mut data_loader = DataLoader::new_independent(
        "data/labeled/camera/image".to_string(),
        "data/labeled/lidar".to_string(),
    );
    data_loader.load().await?;
    let n_total = data_loader.frame_count().min(n_frames);
    if n_total == 0 {
        eprintln!("没有数据");
        return Ok(());
    }

    // ─── 输出目录 ─────────────────────────────────────────────────────────
    let out_dir = if out_prefix.is_empty() {
        let secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();
        let mode = if serial { "serial" } else { "pipeline" };
        PathBuf::from(format!("output/latency_{}_{}", mode, secs))
    } else {
        PathBuf::from(&out_prefix)
    };
    std::fs::create_dir_all(&out_dir)?;

    // 延迟记录 (微秒)
    let mut latencies_us: Vec<u64> = Vec::with_capacity(n_total);

    let mode_label = if serial { "串行" } else { "并行(流水线)" };
    println!("═══ 同帧跟踪延迟测试 [{}] ({} 帧) ═══\n", mode_label, n_total);
    println!("{:-<90}", "");
    println!("{:>8} {:>12} {:>12} {:>10} {:>10} {:>10} {:>8}",
        "帧号", "检测(ms)", "后处理(ms)", "融合(ms)", "跟踪(ms)", "总计(ms)", "FPS");
    println!("{:-<90}", "");

    let total_start = Instant::now();

    if serial {
        // ═════════════════════════════════════════════════════════════════
        //  串行模式：逐帧处理，检测→后处理→融合→跟踪完全串行
        // ═════════════════════════════════════════════════════════════════
        for i in 0..n_total {
            // ── 加载当前帧 ──────────────────────────────────────────────
            if i == 0 {
                data_loader.load_next().await?;
            }

            // ── 阶段 1: LiDAR + Camera 检测（串行等待） ──────────────────
            let t0 = Instant::now();
            let l_h = tokio::spawn(async move { let _ = lidar.act().await; lidar });
            let c_h = tokio::spawn(async move { let _ = camera.act().await; camera });
            let (l_res, c_res) = tokio::join!(l_h, c_h);
            lidar = l_res.unwrap();
            camera = c_res.unwrap();
            let detect_us = t0.elapsed().as_secs_f64() * 1000.0;

            // ── 阶段 2: Swap 后处理 ────────────────────────────────────
            let t1 = Instant::now();
            let swapl = global_swapl();
            swapl.swap_pipeline();
            let swap_us = t1.elapsed().as_secs_f64() * 1000.0;

            // ── 阶段 3: 融合 ───────────────────────────────────────────
            let t2 = Instant::now();
            fuse.act().await;
            let fuse_us = t2.elapsed().as_secs_f64() * 1000.0;

            // ── 阶段 4: 加载下一帧 + 跟踪 ──────────────────────────────
            let t3 = Instant::now();
            if i + 1 < n_total { data_loader.load_next().await?; }
            let _ = tracker.run().await;
            let track_us = t3.elapsed().as_secs_f64() * 1000.0;

            // ── 总延迟 ────────────────────────────────────────────────
            let total_us = t0.elapsed().as_secs_f64() * 1000.0;
            latencies_us.push((total_us * 1000.0) as u64);

            let fps = if total_us > 0.0 { 1000.0 / total_us } else { 0.0 };

            if i % 10 == 0 || i == 0 || i == n_total - 1 {
                println!("{:>8} {:>10.2} {:>10.2} {:>10.2} {:>10.2} {:>10.2} {:>7.1}",
                    i, detect_us, swap_us, fuse_us, track_us, total_us, fps);
            }
        }
    } else {
        // ═════════════════════════════════════════════════════════════════
        //  并行模式：检测和跟踪流水线重叠，测吞吐
        // ═════════════════════════════════════════════════════════════════
        // 预加载前两帧
        if !data_loader.load_next().await? { return Ok(()); }
        if n_total > 1 { data_loader.load_next().await?; }

        let mut l_handle = Some(tokio::spawn(async move { let _ = lidar.act().await; lidar }));
        let mut c_handle = Some(tokio::spawn(async move { let _ = camera.act().await; camera }));

        for i in 0..n_total {
            // ── 阶段 1: 等待检测完成 ────────────────────────────────────
            let t0 = Instant::now();
            let (l_res, c_res) = tokio::join!(l_handle.take().unwrap(), c_handle.take().unwrap());
            lidar = l_res.unwrap();
            camera = c_res.unwrap();
            let detect_us = t0.elapsed().as_secs_f64() * 1000.0;

            // ── 阶段 2: Swap 后处理 ────────────────────────────────────
            let t1 = Instant::now();
            let swapl = global_swapl();
            swapl.swap_pipeline();

            if i + 1 < n_total {
                l_handle = Some(tokio::spawn(async move { let _ = lidar.act().await; lidar }));
                c_handle = Some(tokio::spawn(async move { let _ = camera.act().await; camera }));
            }
            let swap_us = t1.elapsed().as_secs_f64() * 1000.0;

            // ── 阶段 3: 融合 ───────────────────────────────────────────
            let t2 = Instant::now();
            fuse.act().await;
            let fuse_us = t2.elapsed().as_secs_f64() * 1000.0;

            // ── 阶段 4: 加载下一帧 + 跟踪 ──────────────────────────────
            let t3 = Instant::now();
            if i + 2 < n_total { data_loader.load_next().await?; }
            let _ = tracker.run().await;
            let track_us = t3.elapsed().as_secs_f64() * 1000.0;

            // ── 总延迟 ────────────────────────────────────────────────
            let total_us = t0.elapsed().as_secs_f64() * 1000.0;
            latencies_us.push((total_us * 1000.0) as u64);

            let fps = if total_us > 0.0 { 1000.0 / total_us } else { 0.0 };

            if i % 10 == 0 || i == 0 || i == n_total - 1 {
                println!("{:>8} {:>10.2} {:>10.2} {:>10.2} {:>10.2} {:>10.2} {:>7.1}",
                    i, detect_us, swap_us, fuse_us, track_us, total_us, fps);
            }
        }
    }

    let total_elapsed = total_start.elapsed().as_secs_f64();

    // ═════════════════════════════════════════════════════════════════════
    //  统计
    // ═════════════════════════════════════════════════════════════════════
    latencies_us.sort_unstable();
    let avg_us = latencies_us.iter().sum::<u64>() as f64 / latencies_us.len() as f64;
    let p50 = latencies_us[(latencies_us.len() as f64 * 0.50) as usize];
    let p90 = latencies_us[(latencies_us.len() as f64 * 0.90) as usize];
    let p95 = latencies_us[(latencies_us.len() as f64 * 0.95) as usize];
    let p99 = latencies_us[(latencies_us.len() as f64 * 0.99) as usize];
    let min = latencies_us[0];
    let max = latencies_us[latencies_us.len() - 1];

    let avg_ms = avg_us / 1000.0;

    println!();
    println!("═══ {} 延迟统计 ({} 帧, {:>6.1}s 总耗时) ═══\n",
        mode_label, n_total, total_elapsed);
    println!("  {:-<55}", "");
    println!("  {:>15} {:>10} {:>10} {:>10}", "", "μs", "ms", "FPS");
    println!("  {:-<55}", "");
    println!("  {:>15} {:>10} {:>10.2} {:>10.1}", "平均", avg_us as u64, avg_ms, 1000.0 / avg_ms);
    println!("  {:>15} {:>10} {:>10.2} {:>10.1}", "P50", p50, p50 as f64 / 1000.0, 1_000_000.0 / p50 as f64);
    println!("  {:>15} {:>10} {:>10.2} {:>10.1}", "P90", p90, p90 as f64 / 1000.0, 1_000_000.0 / p90 as f64);
    println!("  {:>15} {:>10} {:>10.2} {:>10.1}", "P95", p95, p95 as f64 / 1000.0, 1_000_000.0 / p95 as f64);
    println!("  {:>15} {:>10} {:>10.2} {:>10.1}", "P99", p99, p99 as f64 / 1000.0, 1_000_000.0 / p99 as f64);
    println!("  {:>15} {:>10} {:>10.2} {:>10.1}", "最小", min, min as f64 / 1000.0, 1_000_000.0 / min as f64);
    println!("  {:>15} {:>10} {:>10.2} {:>10.1}", "最大", max, max as f64 / 1000.0, 1_000_000.0 / max as f64);
    println!("  {:-<55}", "");

    // ─── 保存 CSV ─────────────────────────────────────────────────────────
    {
        use std::io::Write;
        let csv_path = out_dir.join("latency.csv");
        let mut f = std::fs::File::create(&csv_path)?;
        writeln!(f, "frame,latency_us")?;
        for (i, &lat) in latencies_us.iter().enumerate() {
            writeln!(f, "{},{}", i, lat)?;
        }
        writeln!(f,)?;
        writeln!(f, "n_frames,{}", n_total)?;
        writeln!(f, "avg_us,{:.0}", avg_us)?;
        writeln!(f, "p50_us,{}", p50)?;
        writeln!(f, "p90_us,{}", p90)?;
        writeln!(f, "p95_us,{}", p95)?;
        writeln!(f, "p99_us,{}", p99)?;
        writeln!(f, "min_us,{}", min)?;
        writeln!(f, "max_us,{}", max)?;
        writeln!(f, "total_s,{:.3}", total_elapsed)?;
        println!("  CSV → {}", csv_path.display());
    }

    // JSON
    {
        use serde::Serialize;
        #[derive(Serialize)]
        struct Output {
            mode: String,
            n_frames: usize,
            total_s: f64,
            avg_ms: f64,
            p50_ms: f64,
            p90_ms: f64,
            p95_ms: f64,
            p99_ms: f64,
            min_ms: f64,
            max_ms: f64,
            avg_fps: f64,
        }

        let output = Output {
            mode: if serial { "serial".into() } else { "pipeline".into() },
            n_frames: n_total,
            total_s: total_elapsed,
            avg_ms,
            p50_ms: p50 as f64 / 1000.0,
            p90_ms: p90 as f64 / 1000.0,
            p95_ms: p95 as f64 / 1000.0,
            p99_ms: p99 as f64 / 1000.0,
            min_ms: min as f64 / 1000.0,
            max_ms: max as f64 / 1000.0,
            avg_fps: 1000.0 / avg_ms,
        };

        let json_path = out_dir.join("latency.json");
        std::fs::write(&json_path, serde_json::to_string_pretty(&output)?)?;
        println!("  JSON → {}", json_path.display());
    }

    println!();
    println!("══════════════════════════════════════════");
    println!("  延迟测试完成");
    println!("══════════════════════════════════════════");

    Ok(())
}
