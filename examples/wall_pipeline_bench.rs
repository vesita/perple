//! 墙体管线对比测试
//!
//! 比较不同墙体检测策略对聚类结果的影响。
//!
//! 管线：地面 → 墙体检测 → 后聚类
//!
//! 固定参数：
//! - 地面检测: PeakScan（默认）
//! - 后聚类: XYGridDBSCAN eps=0.15, min_pts=3, cell=0.30 (with_pre_extracted_wall)
//!
//! 墙体策略从 config/bench/wall/*.toml 读取。
//!
//! 用法:
//!   cargo run --example wall_pipeline_bench -- --mode=quick
//!   cargo run --example wall_pipeline_bench -- --mode=full

use std::time::Instant;

use perple::bench::{
    CliArgs, BenchMode, to_cluster_result,
    config::{load_task_strategies, param_dirname, get_f32, get_i64},
};
use perple::cloud::wall::{
    WallPickStrategy, BevLsd, BevEdLines,
};
use perple::cloud::classify::strategy::{ClusteringStrategy, XYGridDBSCAN};
use perple::cloud::ground::{GroundPickStrategy, PeakScan};

const WARN_THRESHOLD_MS: f64 = 100.0;

struct WallPipelineCase {
    name: String,
    wall: Box<dyn WallPickStrategy>,
    cluster: XYGridDBSCAN,
    frame_count: usize,
    total_pp_ms: f64,
    total_cluster_ms: f64,
    total_cloud_pts: usize,
    total_wall_pts: usize,
    total_non_wall_pts: usize,
    total_clusters: usize,
    total_noise: usize,
    total_cluster_pts: usize,
    frame_times: Vec<f64>,
    min_clusters: usize, max_clusters: usize,
    min_noise: usize, max_noise: usize,
    sq_clusters: u64, sq_noise: u64,
    skipped: bool,
}

impl WallPipelineCase {
    fn new(name: &str, wall: Box<dyn WallPickStrategy>, cluster: XYGridDBSCAN) -> Self {
        Self {
            name: name.to_string(), wall, cluster,
            frame_count: 0,
            total_pp_ms: 0.0, total_cluster_ms: 0.0,
            total_cloud_pts: 0, total_wall_pts: 0,
            total_non_wall_pts: 0,
            total_clusters: 0, total_noise: 0, total_cluster_pts: 0,
            frame_times: Vec::new(),
            min_clusters: usize::MAX, max_clusters: 0,
            min_noise: usize::MAX, max_noise: 0,
            sq_clusters: 0, sq_noise: 0,
            skipped: false,
        }
    }

    fn run_on(&mut self, non_ground: &[[f32; 3]], n_cloud: usize) -> f64 {
        let frame_start = Instant::now();

        let mut wall_buf = non_ground.to_vec();
        let (n_wall, _) = self.wall.pick(&mut wall_buf);
        let non_wall = wall_buf[n_wall..].to_vec();
        let pp_ms = frame_start.elapsed().as_secs_f64() * 1000.0;

        let cluster_start = Instant::now();
        let (sampled, objects) = self.cluster.run(&non_wall);
        let cluster_ms = cluster_start.elapsed().as_secs_f64() * 1000.0;
        let (clusters, noise) = to_cluster_result(&sampled, &objects);
        let n_clusters = clusters.len();
        let frame_ms = pp_ms + cluster_ms;

        self.frame_count += 1;
        self.frame_times.push(frame_ms);
        self.total_pp_ms += pp_ms;
        self.total_cluster_ms += cluster_ms;
        self.total_cloud_pts += n_cloud;
        self.total_wall_pts += n_wall;
        self.total_non_wall_pts += non_wall.len();
        self.total_clusters += n_clusters;
        self.total_noise += noise;
        self.total_cluster_pts += objects.iter().map(|c| c.len()).sum::<usize>();
        self.min_clusters = self.min_clusters.min(n_clusters);
        self.max_clusters = self.max_clusters.max(n_clusters);
        self.min_noise = self.min_noise.min(noise);
        self.max_noise = self.max_noise.max(noise);
        self.sq_clusters += (n_clusters as u64).pow(2);
        self.sq_noise += (noise as u64).pow(2);

        println!("[{}] 云={} 墙={} 非墙={} 簇={} 噪={} | {:.1}+{:.1}={:.1}ms",
            self.name, n_cloud, n_wall, non_wall.len(),
            n_clusters, noise, pp_ms, cluster_ms, frame_ms);

        frame_ms
    }
}

fn build_wall_from_toml(strategy_type: &str, p: &toml::Table) -> Box<dyn WallPickStrategy> {
    match strategy_type {
        "bev_edlines" => {
            let mut s = BevEdLines::with_params(get_f32(p, "distance"), get_i64(p, "min_wall_pts") as usize);
            if let Some(ext) = p.get("min_extent").and_then(|v| v.as_float()) {
                s = s.with_min_extent(ext as f32);
            }
            if let Some(gt) = p.get("grad_threshold").and_then(|v| v.as_float()) {
                s = s.with_grad_threshold(gt as f32);
            }
            if let Some(at) = p.get("anchor_threshold").and_then(|v| v.as_float()) {
                s = s.with_anchor_threshold(at as f32);
            }
            Box::new(s)
        }
        _ => {
            let mut s = BevLsd::with_params(get_f32(p, "distance"), get_i64(p, "min_wall_pts") as usize);
            if let Some(ext) = p.get("min_extent").and_then(|v| v.as_float()) {
                s = s.with_min_extent(ext as f32);
            }
            if let Some(gt) = p.get("grad_threshold").and_then(|v| v.as_float()) {
                s = s.with_grad_threshold(gt as f32);
            }
            if let Some(at) = p.get("angle_tolerance").and_then(|v| v.as_float()) {
                s = s.with_angle_tolerance(at as f32);
            }
            Box::new(s)
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_default_env().filter_level(log::LevelFilter::Warn).init();
    let args: Vec<String> = std::env::args().collect();
    let cli = CliArgs::parse(&args[1..]);
    let mode = cli.mode();
    let _json = cli.has("json");

    let families = load_task_strategies("wall");
    if families.is_empty() {
        eprintln!("WARN: no wall strategy TOML files found in config/bench/wall/");
        return Ok(());
    }

    let params_cfg = match mode {
        BenchMode::Quick => &families[0].quick,
        _ => &families[0].full,
    };
    let expected_frames = params_cfg.frames;

    struct Entry {
        strategy_type: String,
        params: toml::Table,
        dirname: String,
    }
    let entries: Vec<Entry> = families.iter().flat_map(|f| {
        let p = match mode {
            BenchMode::Quick => &f.quick.params,
            _ => &f.full.params,
        };
        p.iter().map(|p| Entry {
            strategy_type: f.strategy_type.clone(),
            params: p.clone(),
            dirname: param_dirname(&f.strategy_type, p),
        }).collect::<Vec<_>>()
    }).collect();

    println!("\n═══ 墙体管线对比 ({}, {} 策略, {} 帧, post-cluster: xy_grid_dbscan e0.15_m3) ═══\n",
        if mode == BenchMode::Quick { "quick" } else { "full" }, entries.len(), expected_frames);

    let mut strategies: Vec<WallPipelineCase> = Vec::with_capacity(entries.len());
    for entry in &entries {
        let wall = build_wall_from_toml(&entry.strategy_type, &entry.params);
        let name = format!("{}_{}", entry.strategy_type, entry.dirname);

        let dummy_wall = Box::new(BevLsd::with_params(0.05, 20));
        let cluster = XYGridDBSCAN::with_params(dummy_wall, 0.30, 3, 12.0, 0.15, 3)
            .with_pre_extracted_wall();

        strategies.push(WallPipelineCase::new(&name, wall, cluster));
    }

    let mut ground = PeakScan::new();

    let mut data_loader = perple::optional::data_loader::DataLoader::new("./data/cloud".to_string());
    data_loader.set_frame_limit(expected_frames);
    data_loader.load().await?;

    let mut frame_idx = 0usize;
    let total_start = Instant::now();

    while data_loader.load_next().await? {
        let cloud: Vec<[f32; 3]> = {
            use perple::swapl::global_swapl;
            let mut stream = global_swapl().clouds.lock().unwrap();
            match stream.read() {
                Some(data) => data,
                None => continue,
            }
        };
        if cloud.is_empty() { frame_idx += 1; continue; }
        let n_cloud = cloud.len();

        let mut buf = cloud.to_vec();
        let (n_ground, _, _) = ground.pick(&mut buf);
        let non_ground = buf[n_ground..].to_vec();

        if frame_idx == 0 {
            let mut first_times: Vec<f64> = Vec::with_capacity(strategies.len());
            for s in strategies.iter_mut() {
                let ms = s.run_on(&non_ground, n_cloud);
                first_times.push(ms);
            }
            let mut skipped_count = 0;
            for (i, ms) in first_times.iter().enumerate() {
                if *ms > WARN_THRESHOLD_MS {
                    println!("  >>> 跳过 {} ({:.1}ms > {:.0}ms)", strategies[i].name, ms, WARN_THRESHOLD_MS);
                    strategies[i].skipped = true;
                    skipped_count += 1;
                }
            }
            if skipped_count > 0 {
                println!("  >>> 跳过 {} 个慢策略，剩余 {} 个\n", skipped_count, strategies.len() - skipped_count);
            }
        } else {
            for s in strategies.iter_mut() {
                if s.skipped { continue; }
                s.run_on(&non_ground, n_cloud);
            }
        }

        frame_idx += 1;
    }

    let total_elapsed = total_start.elapsed();
    println!("\n共 {} 帧，总耗时: {:.1}s\n", frame_idx, total_elapsed.as_secs_f64());

    println!("\n=== 结果汇总（按墙面点数降序） ===");
    println!("{:-<105}", "");
    println!("| {:<40} | {:>7} | {:>5} | {:>5} | {:>7} | {:>7} | {:>7} |",
        "策略", "墙面点", "簇", "噪声", "管线ms", "聚类ms", "总计ms");
    println!("{:-<105}", "");

    let mut sorted: Vec<_> = strategies.iter().collect();
    sorted.sort_by(|a, b| {
        let wa = if a.frame_count > 0 { a.total_wall_pts as f64 / a.frame_count as f64 } else { 0.0 };
        let wb = if b.frame_count > 0 { b.total_wall_pts as f64 / b.frame_count as f64 } else { 0.0 };
        wb.partial_cmp(&wa).unwrap_or(std::cmp::Ordering::Equal)
    });
    for s in &sorted {
        let n = s.frame_count.max(1) as f64;
        let cls_range = if s.min_clusters <= s.max_clusters {
            format!("{}~{}", s.min_clusters, s.max_clusters)
        } else {
            format!("{:.0}", s.total_clusters as f64 / n)
        };
        let noi_range = if s.min_noise <= s.max_noise {
            format!("{}~{}", s.min_noise, s.max_noise)
        } else {
            format!("{:.0}", s.total_noise as f64 / n)
        };
        let avg_total = if s.frame_count > 0 {
            s.frame_times.iter().sum::<f64>() / s.frame_count as f64
        } else { 0.0 };
        println!("  {:<40} | {:>7.0} | {:>5} | {:>5} | {:>7.1} | {:>7.1} | {:>7.1} [{}帧]",
            if s.skipped { format!("{} [跳过]", s.name) } else { s.name.clone() },
            s.total_wall_pts as f64 / n,
            cls_range, noi_range,
            s.total_pp_ms / n, s.total_cluster_ms / n, avg_total,
            s.frame_count,
        );
    }
    println!("{:-<105}", "");
    let active_count = strategies.iter().filter(|s| !s.skipped).count();
    println!("  地面检测: PeakScan (默认, 共享)");
    println!("  后聚类: XYGridDBSCAN cell=0.30 eps=0.15 min_pts=3 max_range=12.0");
    println!("  活跃策略: {}/{} (第一帧 >{}ms 的被跳过)", active_count, strategies.len(), WARN_THRESHOLD_MS);

    Ok(())
}
