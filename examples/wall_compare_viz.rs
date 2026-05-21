//! 墙体策略对比可视化数据生成
//!
//! 加载一帧数据，分别用 BevLsd / BevEdLines / BevHough 做墙体检测，
//! 输出 BEV 密度网格 + 各类墙体策略的分类结果到 JSON，供 Python 绘图。
//!
//! 用法：
//!   cargo run --example wall_compare_viz
//!   cargo run --example wall_compare_viz -- --output output/wall_compare_viz
use std::path::PathBuf;
use std::time::Instant;

use serde::Serialize;

use perple::cloud::ground::{GroundPickStrategy, PeakScan};
use perple::cloud::wall::{WallPickStrategy, BevLsd, BevEdLines, BevHough};
use perple::optional::data_loader::DataLoader;
use perple::swapl::global_swapl;

#[derive(Serialize)]
struct VizData {
    bev: BevGrid,
    strategies: Vec<StrategyOutput>,
}

#[derive(Serialize)]
struct BevGrid {
    size: usize,
    max_range: f32,
    resolution: f32,
    density: Vec<f32>,
}

#[derive(Serialize)]
struct StrategyOutput {
    name: String,
    elapsed_ms: f64,
    n_wall: usize,
    n_non_wall: usize,
    /// Wall 点在原非地面点云中的索引
    wall_indices: Vec<usize>,
    /// Non-wall 点在原非地面点云中的索引
    non_wall_indices: Vec<usize>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Warn)
        .init();

    let args: Vec<String> = std::env::args().collect();
    let out_dir: PathBuf = args.iter()
        .position(|a| a == "--output")
        .and_then(|i| args.get(i + 1))
        .map(|s| PathBuf::from(s))
        .unwrap_or_else(|| PathBuf::from("output/wall_compare_viz"));
    std::fs::create_dir_all(&out_dir)?;

    // ─── 加载一帧数据 ──────────────────────────────────────────────────
    let mut loader = DataLoader::new("./data/cloud".into());
    loader.set_frame_limit(1);
    loader.load().await?;
    loader.load_next().await?;

    let cloud: Vec<[f32; 3]> = {
        let swapl = global_swapl();
        let mut stream = swapl.clouds.lock().unwrap();
        stream.read().unwrap_or_default()
    };
    eprintln!("加载点云 {} 点", cloud.len());
    if cloud.is_empty() {
        eprintln!("点云为空");
        return Ok(());
    }

    // ─── 地面检测 ──────────────────────────────────────────────────────
    let mut buf = cloud.clone();
    let (n_ground, _, _) = PeakScan::new().pick(&mut buf);
    let non_ground = buf[n_ground..].to_vec();
    eprintln!("地面检测: {} 地面 / {} 非地面", n_ground, non_ground.len());

    // ─── 构建 BEV 密度网格 ──────────────────────────────────────────────
    let max_range = 10.0f32;
    let resolution = 0.05f32;
    let size = (2.0 * max_range / resolution) as usize;
    let mut bev_counts = vec![0u32; size * size];
    for p in &non_ground {
        if p[0].abs() >= max_range || p[1].abs() >= max_range { continue; }
        let x = ((p[0] + max_range) / resolution) as isize;
        let y = ((p[1] + max_range) / resolution) as isize;
        if x >= 0 && (x as usize) < size && y >= 0 && (y as usize) < size {
            bev_counts[y as usize * size + x as usize] += 1;
        }
    }

    let mut density = vec![0.0f32; size * size];
    let mut max_val = 0.0f32;
    for i in 0..bev_counts.len() {
        let l = (bev_counts[i] as f32 + 1.0).ln();
        density[i] = l;
        if l > max_val { max_val = l; }
    }
    if max_val > 1e-6 {
        for v in &mut density { *v /= max_val; }
    }

    let bev = BevGrid { size, max_range, resolution, density };

    // ─── 三种墙体策略 ──────────────────────────────────────────────────
    let cfg = perple::config::fixif();
    let strategies: Vec<Box<dyn WallPickStrategy>> = vec![
        Box::new(BevLsd::with_params(cfg.wall_distance, 20)
            .with_grad_threshold(0.08)
            .with_angle_tolerance(cfg.wall_angle_tolerance)
            .with_min_extent(0.5)),
        Box::new(BevEdLines::with_params(cfg.wall_distance, 20)
            .with_min_extent(0.5)),
        Box::new(BevHough::with_params(cfg.wall_distance, 20)),
    ];
    let names = ["bev_lsd", "bev_edlines", "bev_hough"];

    let mut outputs = Vec::new();
    for (i, mut strategy) in strategies.into_iter().enumerate() {
        let mut pts = non_ground.clone();
        let start = Instant::now();
        let (n_wall, _planes) = strategy.pick(&mut pts);
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

        let n = non_ground.len();
        let mut wall_indices = Vec::with_capacity(n_wall);
        let mut non_wall_indices = Vec::with_capacity(n - n_wall);
        for j in 0..n_wall { wall_indices.push(j); }
        for j in n_wall..n { non_wall_indices.push(j); }

        outputs.push(StrategyOutput {
            name: names[i].to_string(),
            elapsed_ms,
            n_wall,
            n_non_wall: n - n_wall,
            wall_indices,
            non_wall_indices,
        });

        eprintln!("  {}: {} walls / {} non-walls [{:.1}ms]",
            names[i], n_wall, n - n_wall, elapsed_ms);
    }

    // ─── 保存 JSON ────────────────────────────────────────────────────
    let data = VizData { bev, strategies: outputs };
    let json_path = out_dir.join("wall_compare.json");
    let json_str = serde_json::to_string(&data)?;
    std::fs::write(&json_path, &json_str)?;
    eprintln!("\n可视化数据 → {}", json_path.display());

    // 保存非地面点云供 Python 绘图
    let pts_path = out_dir.join("non_ground.json");
    let pts_json = serde_json::to_string(&non_ground)?;
    std::fs::write(&pts_path, &pts_json)?;

    Ok(())
}
