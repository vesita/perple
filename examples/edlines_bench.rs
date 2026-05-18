//! EDLines 对比测试：BevEdLines（当前实现）vs EdLinesRef（原版算法）
//!
//! 相同管线（地面提取 → BEV → EDLines → 墙壁验证），只换 EDLines 核心。
//!
//! 输出:
//!   - output/edlines_bench/results.json — 每帧详细数据
//!   - 终端汇总表
//!
//! 用法:
//!   cargo run --release --example edlines_bench
//!   cargo run --release --example edlines_bench -- --frames 50
//!   cargo run --release --example edlines_bench -- --single  # 只用单帧

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

use perple::cloud::wall::{BevEdLines, EdLinesRef, WallPickStrategy};
use perple::optional::data_loader::DataLoader;
use perple::swapl::global_swapl;

/// 单策略单帧运行结果
#[derive(Default, Clone, serde::Serialize, serde::Deserialize)]
struct PerFrameResult {
    edlines_ms: f64,
    n_wall_pts: usize,
    n_planes: usize,
}

#[derive(Default, serde::Serialize, serde::Deserialize)]
struct BenchmarkOutput {
    config: HashMap<String, f64>,
    frames_total: usize,
    bev_edlines: Vec<PerFrameResult>,
    edlines_ref: Vec<PerFrameResult>,
    bev_edlines_avg: PerFrameResult,
    edlines_ref_avg: PerFrameResult,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Warn)
        .init();

    let args: Vec<String> = std::env::args().collect();
    let n_frames: usize = args.iter()
        .position(|a| a == "--frames")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(100);
    let single_frame = args.iter().any(|a| a == "--single");

    // ─── 初始化两种策略（相同 BEV/墙壁参数，不同 EDLines 核心） ──────────
    let mut bev = BevEdLines::new()
        .with_min_extent(0.5)
        .with_gaussian_blur(0.8)
        .with_anchor_threshold(0.04)
        .with_fit_error_threshold(0.5);
    let mut ref_edlines = EdLinesRef::new()
        .with_min_extent(0.5)
        .with_gaussian_blur(0.8)
        .with_anchor_threshold(0.04)
        .with_nfa(true)
        .with_nfa_epsilon(1.0);

    // ─── 加载数据 ─────────────────────────────────────────────────────────
    let actual_frames = if single_frame { 1 } else { n_frames };
    let mut data_loader = DataLoader::new("./data/cloud".to_string());
    data_loader.set_frame_limit(actual_frames);
    data_loader.load().await?;

    let mut frame_count = 0usize;
    let mut bev_results = Vec::new();
    let mut ref_results = Vec::new();

    println!("\n═══ EDLines 对比测试 (BevEdLines vs EdLinesRef) ═══\n");
    println!("{:-<100}", "");
    println!("| {:>4} | {:>30} | {:>7} | {:>6} | {:>9} |",
        "帧", "策略", "墙壁点", "平面", "耗时ms");
    println!("{:-<100}", "");

    let total_start = Instant::now();

    while data_loader.load_next().await? {
        let cloud: Vec<[f32; 3]> = {
            let mut stream = global_swapl().clouds.lock().unwrap();
            match stream.read() {
                Some(data) => data,
                None => continue,
            }
        };
        if cloud.is_empty() { continue; }

        // ── BevEdLines ──
        let mut bev_buf = cloud.clone();
        let bev_start = Instant::now();
        let (n_wall_bev, planes_bev) = bev.pick(&mut bev_buf);
        let bev_ms = bev_start.elapsed().as_secs_f64() * 1000.0;

        // ── EdLinesRef ──
        let mut ref_buf = cloud;
        let ref_start = Instant::now();
        let (n_wall_ref, planes_ref) = ref_edlines.pick(&mut ref_buf);
        let ref_ms = ref_start.elapsed().as_secs_f64() * 1000.0;

        let marker = if (n_wall_bev as f64 - n_wall_ref as f64).abs() > 300.0 {
            "  ← 显著差异"
        } else {
            ""
        };

        println!("| {:>4} | {:>30} | {:>7} | {:>6} | {:>9.2} |",
            frame_count, "BevEdLines (当前)", n_wall_bev, planes_bev.len(), bev_ms);
        println!("| {:>4} | {:>30} | {:>7} | {:>6} | {:>9.2} |{}",
            "", "EdLinesRef (原版)", n_wall_ref, planes_ref.len(), ref_ms, marker);
        println!("{:-<100}", "");

        bev_results.push(PerFrameResult {
            edlines_ms: bev_ms, n_wall_pts: n_wall_bev, n_planes: planes_bev.len(),
        });
        ref_results.push(PerFrameResult {
            edlines_ms: ref_ms, n_wall_pts: n_wall_ref, n_planes: planes_ref.len(),
        });
        frame_count += 1;

        if single_frame { break; }
    }

    let total_elapsed = total_start.elapsed();
    let n = frame_count.max(1) as f64;

    // ─── 汇总计算（分离：既用于终端输出，也用于 JSON） ──────────────────
    let bev_avg_ms = bev_results.iter().map(|r| r.edlines_ms).sum::<f64>() / n;
    let ref_avg_ms = ref_results.iter().map(|r| r.edlines_ms).sum::<f64>() / n;
    let bev_avg_wall = bev_results.iter().map(|r| r.n_wall_pts).sum::<usize>() as f64 / n;
    let ref_avg_wall = ref_results.iter().map(|r| r.n_wall_pts).sum::<usize>() as f64 / n;
    let speed_ratio = if ref_avg_ms > 0.0 { bev_avg_ms / ref_avg_ms } else { 1.0 };

    let (mean_diff, std_diff) = if frame_count > 1 {
        let wall_diffs: Vec<f64> = bev_results.iter().zip(ref_results.iter())
            .map(|(b, r)| b.n_wall_pts as f64 - r.n_wall_pts as f64)
            .collect();
        let m = wall_diffs.iter().sum::<f64>() / n;
        let s = (wall_diffs.iter().map(|d| (d - m).powi(2)).sum::<f64>() / n).sqrt();
        (m, s)
    } else {
        (0.0, 0.0)
    };

    // ─── 终端输出 ───────────────────────────────────────────────────────
    if frame_count > 1 {
        println!("\n═══ 汇总 ═══\n");

        let bev_avg_planes = bev_results.iter().map(|r| r.n_planes).sum::<usize>() as f64 / n;
        let ref_avg_planes = ref_results.iter().map(|r| r.n_planes).sum::<usize>() as f64 / n;

        println!("{:<35} {:>12} {:>12}", "", "BevEdLines", "EdLinesRef");
        println!("{:-<60}", "");
        println!("{:<35} {:>9.2}ms {:>9.2}ms", "平均耗时", bev_avg_ms, ref_avg_ms);
        println!("{:<35} {:>12.1} {:>12.1}", "平均墙壁点数", bev_avg_wall, ref_avg_wall);
        println!("{:<35} {:>12.1} {:>12.1}", "平均墙壁平面数", bev_avg_planes, ref_avg_planes);
        println!("{:<35} {:>9.2} ± {:.2}", "墙壁点差异 (均±σ)", mean_diff, std_diff);
        println!("{:<35} {:>12.2}x", "速度比 (当前/原版)", speed_ratio);
        println!("\n总帧数: {} (总耗时: {:.1}s)", frame_count, total_elapsed.as_secs_f64());
    }

    // ─── 输出 JSON ────────────────────────────────────────────────────────
    let out_dir = PathBuf::from("output/edlines_bench");
    std::fs::create_dir_all(&out_dir)?;

    let bev_avg = PerFrameResult {
        edlines_ms: bev_results.iter().map(|r| r.edlines_ms).sum::<f64>() / n,
        n_wall_pts: (bev_results.iter().map(|r| r.n_wall_pts).sum::<usize>() as f64 / n) as usize,
        n_planes: (bev_results.iter().map(|r| r.n_planes).sum::<usize>() as f64 / n) as usize,
    };
    let ref_avg = PerFrameResult {
        edlines_ms: ref_results.iter().map(|r| r.edlines_ms).sum::<f64>() / n,
        n_wall_pts: (ref_results.iter().map(|r| r.n_wall_pts).sum::<usize>() as f64 / n) as usize,
        n_planes: (ref_results.iter().map(|r| r.n_planes).sum::<usize>() as f64 / n) as usize,
    };

    let mut config = HashMap::new();
    config.insert("n_frames".into(), n);
    config.insert("resolution".into(), 0.05);
    config.insert("distance".into(), 0.10);
    config.insert("gaussian_sigma".into(), 0.8);
    config.insert("anchor_threshold".into(), 0.04);
    config.insert("bev_avg_ms".into(), bev_avg_ms);
    config.insert("ref_avg_ms".into(), ref_avg_ms);
    config.insert("speed_ratio".into(), speed_ratio);
    config.insert("wall_diff_mean".into(), mean_diff);
    config.insert("wall_diff_std".into(), std_diff);

    let output = BenchmarkOutput {
        config,
        frames_total: frame_count,
        bev_edlines: bev_results,
        edlines_ref: ref_results,
        bev_edlines_avg: bev_avg,
        edlines_ref_avg: ref_avg,
    };

    let json_path = out_dir.join("results.json");
    let json_str = serde_json::to_string_pretty(&output)?;
    std::fs::write(&json_path, json_str)?;
    println!("\n结果已保存到: {}", json_path.display());

    Ok(())
}
