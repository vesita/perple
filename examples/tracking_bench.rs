//! 跟踪精度 Benchmark — 不同后聚类策略对跟踪器输出的影响。
//!
//! 数据流：
//!   FrameData (non_wall 点)
//!     → ClusteringStrategy::run()
//!     → 手动构建 CldBud（复用 Cluster 的过滤逻辑）
//!     → Tracker::run_with_detections()
//!     → 收集跟踪指标
//!
//! 用法：
//!   cargo run --example tracking_bench -- --mode=quick    # 单帧快速对比
//!   cargo run --example tracking_bench -- --mode=full     # 多帧全量对比
//!   cargo run --example tracking_bench -- --frames 50     # 指定帧数
//!   cargo run --example tracking_bench -- --strategy=dbscan_qt --frames 20  # 单策略

use std::collections::HashSet;
use std::time::{Duration, Instant};

use perple::bench::{
    BenchStats, BenchStrategy, BenchHarness, BenchRecorder, FrameData,
    GroundWallPreprocessor,
    CliArgs, BenchMode, mats,
};
use perple::cloud::classify::cluster::clusters_to_cldbuds;
use perple::cloud::classify::strategy::{
    ClusteringStrategy, CcCluster,
    DbscanStrategy, RangeImageStrategy,
    XYGridDBSCAN, LvdotClusterStrategy, PruneQt,
};
use perple::cloud::wall::{WallPickStrategy, BevEdLines};
use perple::cloud::CldBud;
use perple::config::fixif;
use perple::tracker::core::Tracker;
use perple::tracker::output::Target;
use perple::utils::boxes::Box3D;

/// 跟踪 benchmark 用例。
///
/// 每个用例包含一个聚类策略 + 一个 Tracker 实例。
/// Tracker 状态在帧间保持，模拟真实跟踪过程。
struct TrackingBenchCase {
    name: String,
    strategy: Box<dyn ClusteringStrategy>,
    tracker: Tracker,
    // 时序
    total_ms: f64,
    frame_count: usize,
    frame_times: Vec<f64>,
    // 聚类统计
    acc_clusters: usize,
    acc_cluster_input: usize,
    // 跟踪输出统计（累积，除以 frame_count 得均值）
    acc_output: usize,
    acc_static: usize,
    acc_moving: usize,
    acc_floating: usize,
    acc_movable: usize,
    // 已见过的 track ID 集合（用于检测新轨迹）
    seen_ids: HashSet<usize>,
    acc_new_tracks: usize,
    // 每帧活跃跟踪数累积
    acc_tracks_sum: usize,
    // 缓存上一帧结果供 write_frame 写入
    last_cldbuds: Vec<CldBud>,
    last_targets: Vec<Target>,
}

impl TrackingBenchCase {
    fn new(name: &str, strategy: Box<dyn ClusteringStrategy>) -> Self {
        Self {
            name: name.to_string(),
            strategy,
            tracker: Tracker::new(),
            total_ms: 0.0,
            frame_count: 0,
            frame_times: Vec::new(),
            acc_clusters: 0,
            acc_cluster_input: 0,
            acc_output: 0,
            acc_static: 0,
            acc_moving: 0,
            acc_floating: 0,
            acc_movable: 0,
            seen_ids: HashSet::new(),
            acc_new_tracks: 0,
            acc_tracks_sum: 0,
            last_cldbuds: Vec::new(),
            last_targets: Vec::new(),
        }
    }
}

impl BenchStrategy for TrackingBenchCase {
    fn name(&self) -> &str {
        &self.name
    }

    fn run(&mut self, frame: &FrameData) -> Duration {
        // 1. 聚类
        let input = frame.non_wall().to_vec();
        let start = Instant::now();
        let (sampled, objects) = self.strategy.run(&input);
        let elapsed = start.elapsed();

        // 2. 转为 CldBud
        let cldbuds = clusters_to_cldbuds(&sampled, &objects);

        // 3. 跟踪（带点云投票——用 non_wall 点作点云投票）
        let targets = self.tracker.run_with_detections(&cldbuds, frame.non_wall());

        // 4. 收集指标（在缓存前完成，避免所有权问题）
        let cluster_count = cldbuds.len();
        let target_count = targets.len();
        let ms = elapsed.as_secs_f64() * 1000.0;
        self.total_ms += ms;
        self.frame_count += 1;
        self.frame_times.push(ms);
        self.acc_clusters += cluster_count;
        self.acc_cluster_input += input.len();
        self.acc_tracks_sum += self.tracker.get_tracking_count();
        self.acc_output += target_count;
        for t in &targets {
            match t.classification.as_str() {
                "static" => self.acc_static += 1,
                "moving" => self.acc_moving += 1,
                "floating" => self.acc_floating += 1,
                "movable" => self.acc_movable += 1,
                _ => {}
            }
            if self.seen_ids.insert(t.id) {
                self.acc_new_tracks += 1;
            }
        }

        // 缓存供 write_frame 写入 DB 可视化
        self.last_cldbuds = cldbuds;
        self.last_targets = targets;

        // 每帧打日志
        if self.frame_count <= 3 || self.frame_count % 10 == 0 {
            let n_tracks = self.tracker.get_tracking_count();
            println!("  [{:30}] 帧{:3} 簇{:2} 跟踪{:2} 输出{:2} | {:>6.1}ms",
                self.name, self.frame_count, cluster_count,
                n_tracks, target_count, ms);
        }

        elapsed
    }

    fn write_frame(&mut self, recorder: &mut BenchRecorder, frame: &FrameData) {
        recorder.begin_frame(frame.frame_idx);
        // 写入聚类包围盒
        let boxes: Vec<(Box3D, String)> = self.last_cldbuds.iter()
            .enumerate()
            .map(|(i, c)| (c.the_box.clone(), format!("c{}", i)))
            .collect();
        if !boxes.is_empty() {
            recorder.write_boxes(&boxes, mats::CLUSTER_BOX);
        }
        // 写入跟踪目标包围盒（不同分类用不同颜色）
        let color_for = |cls: &str| -> &str {
            match cls {
                "static" => mats::BOX,
                "moving" => mats::ALERT,
                "floating" => mats::FAR_BOX,
                "movable" => mats::SELECTED,
                _ => mats::BOX,
            }
        };
        let tb: Vec<(Box3D, String)> = self.last_targets.iter()
            .map(|t| (t.the_box.clone(), format!("id{}_cls{}", t.id, t.classification)))
            .collect();
        if !tb.is_empty() {
            for (b, label) in &tb {
                let cls = self.last_targets.iter()
                    .find(|t| format!("id{}_cls{}", t.id, t.classification) == *label)
                    .map(|t| t.classification.as_str())
                    .unwrap_or("static");
                recorder.write_boxes(&[(b.clone(), label.clone())], color_for(cls));
            }
        }
        recorder.end_frame();
    }

    fn summarize(&self) {
        // 由主逻辑统一排序输出
    }

    fn stats(&self) -> BenchStats {
        BenchStats {
            name: self.name.clone(),
            frame_count: self.frame_count,
            total_ms: self.total_ms,
            frame_times: self.frame_times.clone(),
        }
    }

    fn extra_metrics(&self) -> Vec<(String, f64)> {
        let n = self.frame_count.max(1) as f64;
        vec![
            ("avg_input".into(), self.acc_cluster_input as f64 / n),
            ("avg_clusters".into(), self.acc_clusters as f64 / n),
            ("avg_output".into(), self.acc_output as f64 / n),
            ("avg_tracks".into(), self.acc_tracks_sum as f64 / n),
            ("avg_new_tracks".into(), self.acc_new_tracks as f64 / n),
            ("static_ratio".into(), self.acc_static as f64 / self.acc_output.max(1) as f64),
            ("moving_ratio".into(), self.acc_moving as f64 / self.acc_output.max(1) as f64),
            ("floating_ratio".into(), self.acc_floating as f64 / self.acc_output.max(1) as f64),
            ("movable_ratio".into(), self.acc_movable as f64 / self.acc_output.max(1) as f64),
        ]
    }
}

// ── 策略工厂 ──────────────────────────────────────────────

fn build_strategies() -> Vec<Box<dyn ClusteringStrategy>> {
    let cfg = fixif();
    let wall: Box<dyn WallPickStrategy> = Box::new(
        BevEdLines::with_params(0.05, 20).with_min_extent(0.0),
    );

    vec![
        // 1. dbscan_qt — 默认基线，四叉树加速
        Box::new(DbscanStrategy::with_params(
            cfg.cluster.merge_patience,
            cfg.cluster.eps_slope,
            cfg.cluster.min_points_per_cluster.unwrap_or(3) as usize,
            20, 10, cfg.cluster.voxel_size,
        )),
        // 2. lvdot — 体素占用过滤
        Box::new(LvdotClusterStrategy::direct(
            cfg.cluster.voxel_size,
            cfg.cluster.min_points_per_cluster.unwrap_or(3) as usize,
            0.30, 5,
        )),
        // 3. prune_qt — 四叉树剪叶过滤
        Box::new(PruneQt::new().with_params(
            cfg.cluster.min_points_per_cluster.unwrap_or(3) as usize,
            0.30, 5,
        )),
        // 4. xy_grid_dbscan — XY 网格预过滤
        Box::new(XYGridDBSCAN::with_params(
            wall, 0.30, 3, 12.0, 0.30, 5,
        ).with_pre_extracted_wall()),
        // 5. range_image — 极速 range image 聚类
        Box::new(RangeImageStrategy::with_params(
            cfg.cluster.azimuth_resolution,
            cfg.cluster.elevation_resolution,
            cfg.cluster.cluster_threshold,
            cfg.cluster.min_points_per_cluster.unwrap_or(3) as usize,
        )),
        // 6. cc_grid — 连通域
        Box::new(CcCluster::new(
            cfg.cluster.voxel_size.max(0.10), 3,
        )),
    ]
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Warn)
        .init();

    let args: Vec<String> = std::env::args().collect();
    let cli = CliArgs::parse(&args[1..]);
    let mode = cli.mode();

    let frame_limit: usize = match mode {
        BenchMode::Quick => 1,
        BenchMode::Full => cli.get("frames", 20usize),
        BenchMode::Single => cli.get("frames", 10usize),
    };

    let strategies: Vec<Box<dyn ClusteringStrategy>> = if mode == BenchMode::Single {
        let name = cli.strategy().unwrap_or_else(|| "dbscan_qt".to_string());
        // 用 name 查预构建列表，找不到则尝试直接构造
        let all = build_strategies();
        let mut found: Vec<_> = all.into_iter().filter(|s| s.strategy_name() == name).collect();
        if found.is_empty() {
            let cfg = fixif();
            let s: Box<dyn ClusteringStrategy> = match name.as_str() {
                "dbscan_qt" => Box::new(DbscanStrategy::with_params(
                    cfg.cluster.merge_patience, cfg.cluster.eps_slope,
                    cfg.cluster.min_points_per_cluster.unwrap_or(3) as usize,
                    20, 10, cfg.cluster.voxel_size,
                )),
                "lvdot" | "lvdot_grid" => Box::new(LvdotClusterStrategy::direct(
                    cfg.cluster.voxel_size,
                    cfg.cluster.min_points_per_cluster.unwrap_or(3) as usize,
                    0.30, 5,
                )),
                "lvdot_qt" | "prune_qt" => Box::new(PruneQt::new().with_params(
                    cfg.cluster.min_points_per_cluster.unwrap_or(3) as usize,
                    0.30, 5,
                )),
                "xy_grid_dbscan" => Box::new(XYGridDBSCAN::new().with_pre_extracted_wall()),
                "range_image" => Box::new(RangeImageStrategy::new()),
                "cc_grid" | "cc" => Box::new(CcCluster::new(cfg.cluster.voxel_size.max(0.10), 3)),
                _ => {
                    eprintln!("未知策略 '{}'，使用 dbscan_qt", name);
                    Box::new(DbscanStrategy::with_params(
                        cfg.cluster.merge_patience, cfg.cluster.eps_slope,
                        cfg.cluster.min_points_per_cluster.unwrap_or(3) as usize,
                        20, 10, cfg.cluster.voxel_size,
                    ))
                }
            };
            found.push(s);
        }
        found
    } else {
        build_strategies()
    };

    println!("═══ 聚类策略 → 跟踪影响对比 ({} 帧, {} 策略) ═══\n", frame_limit, strategies.len());
    for s in &strategies {
        println!("  - {}", s.strategy_name());
    }
    println!();

    let mut bench_cases: Vec<Box<dyn BenchStrategy>> = strategies
        .into_iter()
        .map(|s| {
            let name = s.strategy_name().to_string();
            Box::new(TrackingBenchCase::new(&name, s)) as Box<dyn BenchStrategy>
        })
        .collect();

    let harness = BenchHarness::new("./data/cloud", frame_limit);
    let mut preprocessor = GroundWallPreprocessor::default();

    // ── 输出目录：output/bench/tracking/{strategy}/{strategy}.db ──
    let out_root = "output/bench/tracking";
    let mut recorders: Vec<BenchRecorder> = bench_cases
        .iter()
        .map(|s| {
            let dir = format!("{}/{}", out_root, s.name());
            std::fs::create_dir_all(&dir).ok();
            let path = format!("{}/{}.db", dir, s.name());
            // 删除旧 DB 确保干净
            std::fs::remove_file(&path).ok();
            BenchRecorder::new(&path).expect("创建 recorder 失败")
        })
        .collect();

    let all_stats = harness.run(&mut preprocessor, &mut bench_cases, &mut recorders).await?;

    // VACUUM 并关闭所有 DB
    for rec in &recorders {
        let _ = rec.save();
    }

    // ── 汇总排序 ──
    let mut indexed: Vec<(usize, &Box<dyn BenchStrategy>)> = bench_cases.iter().enumerate().collect();
    indexed.sort_by(|(_, a), (_, b)| {
        let at = a.stats().total_ms / a.stats().frame_count.max(1) as f64;
        let bt = b.stats().total_ms / b.stats().frame_count.max(1) as f64;
        at.partial_cmp(&bt).unwrap_or(std::cmp::Ordering::Equal)
    });

    println!("\n═══ 跟踪质量汇总 (按速度升序) ═══");
    println!("{:-<105}", "");
    println!("| {:<33} | {:>5} | {:>6} | {:>5} | {:>4} {:>4} {:>4} {:>4} | {:>7} |",
        "策略", "簇/帧", "跟踪/帧", "输出/帧",
        "静%", "动%", "浮%", "可%", "ms/帧");
    println!("{:-<105}", "");
    for (_, s) in &indexed {
        let st = s.stats();
        let extra = s.extra_metrics();
        let to_f = |key: &str| -> f64 {
            extra.iter().find(|(k, _)| k == key).map(|(_, v)| *v).unwrap_or(0.0)
        };
        let avg_ms = if st.frame_count > 0 { st.total_ms / st.frame_count as f64 } else { 0.0 };
        println!("  {:<35} | {:>5.0} | {:>6.1} | {:>5.0} | {:>3.0}% {:>3.0}% {:>3.0}% {:>3.0}% | {:>7.1}ms",
            st.name,
            to_f("avg_clusters"),
            to_f("avg_tracks"),
            to_f("avg_output"),
            to_f("static_ratio") * 100.0,
            to_f("moving_ratio") * 100.0,
            to_f("floating_ratio") * 100.0,
            to_f("movable_ratio") * 100.0,
            avg_ms);
    }
    println!("{:-<105}", "");

    // ── 写入 info.json（与 cluster_bench 格式对齐） ──
    let all_times: Vec<f64> = all_stats.iter().flat_map(|st| st.frame_times.iter().copied()).collect();
    let fastest = all_times.iter().cloned().fold(f64::MAX, f64::min);
    let slowest = all_times.iter().cloned().fold(f64::MIN, f64::max);
    let avg_all = if all_times.is_empty() { 0.0 } else { all_times.iter().sum::<f64>() / all_times.len() as f64 };
    let median = perple::bench::compute_median(all_times);

    let results: Vec<serde_json::Value> = indexed.iter().map(|(_, s)| {
        let st = s.stats();
        let extra = s.extra_metrics();
        let extra_map: serde_json::Map<String, serde_json::Value> = extra.iter()
            .map(|(k, v)| (k.clone(), serde_json::Value::Number(
                serde_json::Number::from_f64(*v).unwrap_or(serde_json::Number::from_f64(0.0).unwrap())
            )))
            .collect();
        serde_json::json!({
            "name": st.name,
            "frame_count": st.frame_count,
            "total_ms": st.total_ms,
            "avg_ms": if st.frame_count > 0 { st.total_ms / st.frame_count as f64 } else { 0.0 },
            "extra": serde_json::Value::Object(extra_map),
        })
    }).collect();

    let mode_label = match mode {
        BenchMode::Quick => "quick",
        BenchMode::Full => "full",
        BenchMode::Single => "single",
    };

    let info = serde_json::json!({
        "strategy": "tracking",
        "mode": mode_label,
        "stats": {
            "fastest_ms": fastest,
            "slowest_ms": slowest,
            "avg_ms": avg_all,
            "median_ms": median,
        },
        "results": results,
    });
    let info_path = format!("{}/info.json", out_root);
    std::fs::create_dir_all(out_root).ok();
    std::fs::write(&info_path, serde_json::to_string_pretty(&info)?)?;
    println!("\n输出已保存至: {}/", out_root);

    // ── 可选 stdout JSON ──
    if cli.has("json") {
        println!("\n=== JSON ===");
        println!("{}", serde_json::to_string_pretty(&info)?);
    }

    Ok(())
}
