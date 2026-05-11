//! 墙体管线对比测试
//!
//! 在完整五级管线（地面 → 降噪 → 墙体 → 降噪 → 聚类）下，
//! 比较不同墙体提取策略对聚类结果的影响。
//!
//! 固定参数：
//! - 地面提取: PeakScan（默认）
//! - 预处理降噪: RadiusOutlierRemoval r=0.30, m=3
//! - 后处理降噪: RadiusOutlierRemoval r=0.20, m=3
//! - 后聚类: XYGridDBSCAN eps=0.15, min_pts=3, cell=0.30 (with_pre_extracted_wall)
//!
//! 墙体策略从 config/bench/wall/*.toml 读取。
//!
//! 用法:
//!   cargo run --example wall_pipeline_bench -- --mode=quick
//!   cargo run --example wall_pipeline_bench -- --mode=full
//!   cargo run --example wall_pipeline_bench -- --mode=full --json

use std::time::{Duration, Instant};

use perple::bench::{
    BenchStrategy, BenchStats, BenchHarness, BenchRecorder, FrameData, Preprocessed,
    PassthroughPreprocessor, WallPreprocessor, DenoisePreprocessor, Preprocessor,
    CliArgs, BenchMode, to_cluster_result,
    config::{load_task_strategies, param_dirname, get_f32, get_i64},
    mats, CLUSTER_PALETTE,
};
use perple::cloud::wall::{
    WallPickStrategy, TopDownCluster, XYRansacWall, NormalWall, QuadtreeWall,
    AdaptiveDBSCANWall, SequentialFit, XYDBSCANWall, Downsampler,
};
use perple::cloud::denoise::RadiusOutlierRemoval;
use perple::cloud::classify::strategy::{ClusteringStrategy, XYGridDBSCAN};
use perple::cloud::ground::PeakScan;
use perple::utils::boxes::Box3D;
use redra_client::spawn_point;

// ── 墙体管线对比用例 ──────────────────────────────────────────

struct WallPipelineCase {
    name: String,
    pp: DenoisePreprocessor,
    cluster: XYGridDBSCAN,
    // 累计统计
    frame_count: usize,
    total_pp_ms: f64,
    total_cluster_ms: f64,
    total_cloud_pts: usize,
    total_wall_pts: usize,
    total_non_wall_pts: usize,
    total_denoised_pts: usize,
    total_clusters: usize,
    total_noise: usize,
    total_cluster_pts: usize,
    frame_times: Vec<f64>,
    min_clusters: usize, max_clusters: usize,
    min_noise: usize, max_noise: usize,
    sq_clusters: u64, sq_noise: u64,
    // 写帧缓存
    last_sampled: Vec<[f32; 3]>,
    last_assignment: Vec<Option<usize>>,
    last_clusters: Vec<Vec<[f32; 3]>>,
}

impl WallPipelineCase {
    fn new(name: &str, pp: DenoisePreprocessor, cluster: XYGridDBSCAN) -> Self {
        Self {
            name: name.to_string(), pp, cluster,
            frame_count: 0,
            total_pp_ms: 0.0, total_cluster_ms: 0.0,
            total_cloud_pts: 0, total_wall_pts: 0,
            total_non_wall_pts: 0, total_denoised_pts: 0,
            total_clusters: 0, total_noise: 0, total_cluster_pts: 0,
            frame_times: Vec::new(),
            min_clusters: usize::MAX, max_clusters: 0,
            min_noise: usize::MAX, max_noise: 0,
            sq_clusters: 0, sq_noise: 0,
            last_sampled: Vec::new(), last_assignment: Vec::new(),
            last_clusters: Vec::new(),
        }
    }
}

impl BenchStrategy for WallPipelineCase {
    fn name(&self) -> &str { &self.name }

    fn run(&mut self, frame: &FrameData) -> Duration {
        let frame_start = Instant::now();

        // ── 完整管线预处理（地面 → 降噪 → 墙体 → 降噪）──
        let preprocessed = self.pp.preprocess(frame.cloud);
        let (n_non_ground, n_non_wall, denoised) = match &preprocessed {
            Preprocessed::Denoise { non_ground, non_wall, denoised } =>
                (non_ground.len(), non_wall.len(), denoised.clone()),
            _ => unreachable!(),
        };
        let pp_ms = frame_start.elapsed().as_secs_f64() * 1000.0;
        let n_wall = n_non_ground.saturating_sub(n_non_wall);
        let n_cloud = frame.cloud.len();

        // ── 后聚类（xy_grid_dbscan e0.15 m3）──
        let cluster_start = Instant::now();
        let (sampled, objects) = self.cluster.run(&denoised);
        let cluster_ms = cluster_start.elapsed().as_secs_f64() * 1000.0;
        let (clusters, noise) = to_cluster_result(&sampled, &objects);
        let n_clusters = clusters.len();
        let frame_ms = pp_ms + cluster_ms;

        // ── 累计 ──
        self.frame_count += 1;
        self.frame_times.push(frame_ms);
        self.total_pp_ms += pp_ms;
        self.total_cluster_ms += cluster_ms;
        self.total_cloud_pts += n_cloud;
        self.total_wall_pts += n_wall;
        self.total_non_wall_pts += n_non_wall;
        self.total_denoised_pts += denoised.len();
        self.total_clusters += n_clusters;
        self.total_noise += noise;
        self.total_cluster_pts += objects.iter().map(|c| c.len()).sum::<usize>();
        self.min_clusters = self.min_clusters.min(n_clusters);
        self.max_clusters = self.max_clusters.max(n_clusters);
        self.min_noise = self.min_noise.min(noise);
        self.max_noise = self.max_noise.max(noise);
        self.sq_clusters += (n_clusters as u64).pow(2);
        self.sq_noise += (noise as u64).pow(2);

        // ── 写帧缓存 ──
        self.last_sampled = sampled;
        self.last_clusters = clusters;
        let mut assignment = vec![None; self.last_sampled.len()];
        for (ci, obj) in objects.iter().enumerate() {
            for &i in obj { assignment[i] = Some(ci); }
        }
        self.last_assignment = assignment;

        println!("[{}] 云={} 墙={} 非墙={} 降噪后={} 簇={} 噪={} | {:.1}+{:.1}={:.1}ms",
            self.name, n_cloud, n_wall, n_non_wall, denoised.len(),
            n_clusters, noise, pp_ms, cluster_ms, frame_ms);

        Duration::from_secs_f64(frame_ms / 1000.0)
    }

    fn write_frame(&mut self, recorder: &mut BenchRecorder, frame: &FrameData) {
        recorder.begin_frame(frame.frame_idx);
        for (i, pt) in self.last_sampled.iter().enumerate() {
            recorder.spawn(spawn_point(*pt, mats::BG).id(i as u64));
        }
        for (i, assignment) in self.last_assignment.iter().enumerate() {
            let id = i as u64;
            match assignment {
                Some(ci) => {
                    let color = CLUSTER_PALETTE[ci % CLUSTER_PALETTE.len()];
                    recorder.set_material(id, color);
                }
                None => recorder.set_material(id, mats::NOISE),
            }
        }
        let boxes: Vec<Box3D> = self.last_clusters.iter()
            .map(|c| Box3D::from_cloud_aabb(c, 0.05)).collect();
        if !boxes.is_empty() {
            let tagged: Vec<(Box3D, String)> = boxes.iter().enumerate()
                .map(|(i, b)| (b.clone(), format!("c{}", i)))
                .collect();
            recorder.write_boxes(&tagged, mats::CLUSTER_BOX);
        }
        recorder.end_frame();
    }

    fn summarize(&self) {
        let n = self.frame_count.max(1) as f64;
        let avg_ms = self.total_pp_ms / n + self.total_cluster_ms / n;
        let cls_range = if self.min_clusters <= self.max_clusters {
            format!("{}~{}", self.min_clusters, self.max_clusters)
        } else {
            format!("{:.0}", self.total_clusters as f64 / n)
        };
        let noi_range = if self.min_noise <= self.max_noise {
            format!("{}~{}", self.min_noise, self.max_noise)
        } else {
            format!("{:.0}", self.total_noise as f64 / n)
        };
        println!("  {:<40} | {:>7.0} | {:>5} | {:>5} | {:>7.1} | {:>7.1} | {:>7.1} [{}帧]",
            self.name,
            self.total_wall_pts as f64 / n,
            cls_range,
            noi_range,
            self.total_pp_ms / n,
            self.total_cluster_ms / n,
            avg_ms,
            self.frame_count,
        );
    }

    fn stats(&self) -> BenchStats {
        BenchStats {
            name: self.name.clone(),
            frame_count: self.frame_count,
            total_ms: self.total_pp_ms + self.total_cluster_ms,
            frame_times: self.frame_times.clone(),
        }
    }

    fn extra_metrics(&self) -> Vec<(String, f64)> {
        let n = self.frame_count.max(1) as f64;
        let avg_cls = self.total_clusters as f64 / n;
        let avg_noi = self.total_noise as f64 / n;
        let var_cls = (self.sq_clusters as f64 / n) - avg_cls * avg_cls;
        let var_noi = (self.sq_noise as f64 / n) - avg_noi * avg_noi;
        let avg_wall_ratio = if self.total_cloud_pts > 0 {
            self.total_wall_pts as f64 / self.total_cloud_pts as f64 * 100.0
        } else { 0.0 };
        vec![
            ("avg_cloud_pts".into(), self.total_cloud_pts as f64 / n),
            ("avg_wall_pts".into(), self.total_wall_pts as f64 / n),
            ("avg_non_wall_pts".into(), self.total_non_wall_pts as f64 / n),
            ("avg_denoised_pts".into(), self.total_denoised_pts as f64 / n),
            ("wall_ratio_pct".into(), avg_wall_ratio),
            ("avg_clusters".into(), avg_cls),
            ("std_clusters".into(), var_cls.max(0.0).sqrt()),
            ("min_clusters".into(), self.min_clusters as f64),
            ("max_clusters".into(), self.max_clusters as f64),
            ("avg_noise".into(), avg_noi),
            ("std_noise".into(), var_noi.max(0.0).sqrt()),
            ("avg_pp_ms".into(), self.total_pp_ms / n),
            ("avg_cluster_ms".into(), self.total_cluster_ms / n),
        ]
    }
}

// ── TOML 策略工厂（从 wall_bench 移植） ─────────────────────

fn build_wall_from_toml(strategy_type: &str, p: &toml::Table) -> Box<dyn WallPickStrategy> {
    match strategy_type {
        "top_down" => Box::new(TopDownCluster::with_params(f(p, "cell_size"), i(p, "min_density") as usize, 2)
            .with_width_ratio(f(p, "width_ratio"))),
        "xy_ransac" => Box::new(XYRansacWall::with_params(f(p, "distance"), i(p, "iterations") as usize, 30).with_seed(42)),
        "normal_wall" => Box::new(NormalWall::with_params(f(p, "cell_size"), i(p, "min_pts") as usize, 30.0)
            .with_normal_threshold(f(p, "normal_threshold"))),
        "quadtree" => Box::new(QuadtreeWall::with_params(f(p, "cell_size"), i(p, "min_pts") as usize, 0.5)
            .with_width_ratio(f(p, "width_ratio"))),
        "seq_fit" => Box::new(SequentialFit::with_params(f(p, "distance"), f(p, "normal_threshold"), i(p, "max_walls") as usize)),
        "adaptive_dbscan" => {
            let ds = p.get("downsampler").and_then(|v| v.as_str()).unwrap_or("grid");
            let down = if ds == "fps" { Downsampler::FPS } else { Downsampler::Grid };
            Box::new(AdaptiveDBSCANWall::with_params(f(p, "base_eps"), f(p, "scale_factor"), i(p, "min_pts") as usize)
                .with_downsampler(down))
        }
        "xy_dbscan_wall" => Box::new(XYDBSCANWall::with_params(f(p, "eps"), i(p, "min_pts") as usize, f(p, "min_z_span"))),
        _ => panic!("未知墙体策略类型: {}", strategy_type),
    }
}

fn f(t: &toml::Table, key: &str) -> f32 { get_f32(t, key) }
fn i(t: &toml::Table, key: &str) -> i64 { get_i64(t, key) }

// ── 主入口 ──────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_default_env().filter_level(log::LevelFilter::Warn).init();
    let args: Vec<String> = std::env::args().collect();
    let cli = CliArgs::parse(&args[1..]);
    let mode = cli.mode();
    let json = cli.has("json");

    // 加载墙体策略 TOML
    let families = load_task_strategies("wall");
    if families.is_empty() {
        eprintln!("WARN: no wall strategy TOML files found in config/bench/wall/");
        return Ok(());
    }

    let (mode_label, expected_frames) = match mode {
        BenchMode::Quick => ("quick", families[0].quick.frames),
        _ => ("full", families[0].full.frames),
    };

    // 展开为平坦参数列表
    struct Entry {
        strategy_type: String,
        params: toml::Table,
        dirname: String,
    }
    let entries: Vec<Entry> = families.iter().flat_map(|f| {
        let params = match mode {
            BenchMode::Quick => &f.quick.params,
            _ => &f.full.params,
        };
        params.iter().map(|p| Entry {
            strategy_type: f.strategy_type.clone(),
            params: p.clone(),
            dirname: param_dirname(&f.strategy_type, p),
        }).collect::<Vec<_>>()
    }).collect();

    println!("\n═══ 墙体管线对比 ({}, {} 策略, {} 帧, post-cluster: xy_grid_dbscan e0.15_m3) ═══\n",
        mode_label, entries.len(), expected_frames);

    // 构建策略 + recorder
    let mut strategies: Vec<Box<dyn BenchStrategy>> = Vec::with_capacity(entries.len());
    let mut recorders: Vec<BenchRecorder> = Vec::with_capacity(entries.len());
    let out_dir = "output/bench/wall_pipeline";
    std::fs::create_dir_all(out_dir)?;

    for entry in &entries {
        let wall = build_wall_from_toml(&entry.strategy_type, &entry.params);
        let name = format!("{}_{}", entry.strategy_type, entry.dirname);

        let wall_pp = WallPreprocessor::new(
            Box::new(PeakScan::new()),
            Box::new(RadiusOutlierRemoval::new(0.30, 3)),
            wall,
        );
        let pp = DenoisePreprocessor::new(
            wall_pp,
            Box::new(RadiusOutlierRemoval::new(0.20, 3)),
        );

        let dummy_wall = Box::new(XYRansacWall::with_params(0.05, 50, 30));
        let cluster = XYGridDBSCAN::with_params(dummy_wall, 0.30, 3, 12.0, 0.15, 3)
            .with_pre_extracted_wall();

        strategies.push(Box::new(WallPipelineCase::new(&name, pp, cluster)));

        let db_path = format!("{}/{}.db", out_dir, name);
        let _ = std::fs::remove_file(&db_path);
        recorders.push(BenchRecorder::new(&db_path)?);
    }

    // 执行（PassthroughPreprocessor 让每个策略自行处理原始点云）
    let harness = BenchHarness::new("./data/cloud", expected_frames);
    let mut pp = PassthroughPreprocessor;
    let _all_stats = harness.run(&mut pp, &mut strategies, &mut recorders).await?;

    // VACUUM
    for rec in &recorders {
        rec.save()?;
    }

    // 汇总输出
    if json {
        println!("\n=== JSON ===");
        let entries_json: Vec<String> = strategies.iter().map(|s| {
            let st = s.stats();
            let extra = s.extra_metrics();
            let base = format!(r#"{{"name":"{}","frames":{},"total_ms":{},"avg_ms":{:.1}"#,
                st.name, st.frame_count, st.total_ms, st.total_ms / st.frame_count.max(1) as f64);
            let e: String = extra.iter().map(|(k, v)| format!(r#","{}":{}"#, k, v)).collect();
            format!("{}{}}}", base, e)
        }).collect();
        println!("[{}]", entries_json.join(","));
    } else {
        println!("\n=== 结果汇总（按墙面点数降序） ===");
        println!("{:-<100}", "");
        println!("| {:<40} | {:>7} | {:>5} | {:>5} | {:>7} | {:>7} | {:>7} |",
            "策略", "墙面点", "簇", "噪声", "管线ms", "聚类ms", "总计ms");
        println!("{:-<100}", "");

        let mut sorted: Vec<_> = strategies.iter().collect();
        sorted.sort_by(|a, b| {
            let wa = a.extra_metrics().iter().find(|(k,_)| k == "avg_wall_pts")
                .map(|(_,v)| *v).unwrap_or(0.0);
            let wb = b.extra_metrics().iter().find(|(k,_)| k == "avg_wall_pts")
                .map(|(_,v)| *v).unwrap_or(0.0);
            wb.partial_cmp(&wa).unwrap_or(std::cmp::Ordering::Equal)
        });
        for s in &sorted { s.summarize(); }
        println!("{:-<100}", "");
        println!("\n  地面提取: PeakScan (默认)");
        println!("  预处理降噪: RadiusOutlierRemoval r=0.30 m=3");
        println!("  后处理降噪: RadiusOutlierRemoval r=0.20 m=3");
        println!("  后聚类: XYGridDBSCAN cell=0.30 eps=0.15 min_pts=3 max_range=12.0");
    }

    Ok(())
}
