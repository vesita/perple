use std::time::{Duration, Instant};

use perple::bench::{
    BenchStrategy, BenchStats, BenchHarness, BenchRecorder, FrameData,
    GroundWallPreprocessor, Preprocessor,
    CliArgs, BenchMode, run_toml_bench, StrategyBuilder, to_cluster_result,
    mats, CLUSTER_PALETTE,
};
use perple::cloud::classify::strategy::{
    ClusteringStrategy, CcCluster, RansacCluster, SeqCluster,
    DbscanStrategy, DbscanGrid, RangeImageStrategy, XYGridDBSCAN, LvdotClusterStrategy, PruneQt,
};
use perple::cloud::wall::{WallPickStrategy, BevLsd};
use perple::config::fixif;
use perple::utils::boxes::Box3D;
use redra_client::spawn_point;

/// 输入源策略
enum InputSource {
    /// frame.non_ground() — 仅去地面+预处理降噪，供 lvdot（自带体素过滤+墙提）使用
    NonGround,
    /// frame.denoised() — 去地面+墙体+后处理降噪，供通用聚类使用
    Denoised,
}

struct ClusterBenchCase {
    name: String,
    strategy: Box<dyn ClusteringStrategy>,
    input_source: InputSource,
    total_ms: f64, frame_count: usize,
    total_clusters: usize, total_noise: usize,
    total_input_n: usize, total_cluster_pts: usize,
    frame_times: Vec<f64>,
    // 分布统计（每帧 min/max + 平方和→std）
    min_clusters_per_frame: usize, max_clusters_per_frame: usize,
    min_noise_per_frame: usize, max_noise_per_frame: usize,
    sq_clusters: u64, sq_noise: u64,
    last_clusters: Vec<Vec<[f32; 3]>>, last_noise: usize, last_input_n: usize,
    last_sampled: Vec<[f32; 3]>,
    /// 每个点的聚类分配（Some(ci) = 簇索引，None = 噪点）
    last_assignment: Vec<Option<usize>>,
}

impl ClusterBenchCase {
    fn new(name: &str, strategy: Box<dyn ClusteringStrategy>, input_source: InputSource) -> Self {
        Self {
            name: name.to_string(), strategy, input_source,
            total_ms: 0.0, frame_count: 0,
            total_clusters: 0, total_noise: 0,
            total_input_n: 0, total_cluster_pts: 0,
            frame_times: Vec::new(),
            min_clusters_per_frame: usize::MAX, max_clusters_per_frame: 0,
            min_noise_per_frame: usize::MAX, max_noise_per_frame: 0,
            sq_clusters: 0, sq_noise: 0,
            last_clusters: Vec::new(), last_noise: 0, last_input_n: 0,
            last_sampled: Vec::new(),
            last_assignment: Vec::new(),
        }
    }
}

impl BenchStrategy for ClusterBenchCase {
    fn name(&self) -> &str { &self.name }
    fn run(&mut self, frame: &FrameData) -> Duration {
        // lvdot: 使用 non_ground（内部自带体素占用过滤替代墙提）
        // 其他:  使用 non_wall（去地面+墙体后的纯障碍物聚类）
        let input = match self.input_source {
            InputSource::NonGround => frame.non_ground().to_vec(),
            InputSource::Denoised => frame.denoised().to_vec(),
        };
        let start = Instant::now();
        let (sampled, objects) = self.strategy.run(&input);
        let elapsed = start.elapsed();
        self.last_input_n = input.len();
        let (clusters, noise) = to_cluster_result(&sampled, &objects);
        let n_clusters = clusters.len();
        // 簇大小统计
        let cluster_total_pts: usize = objects.iter().map(|c| c.len()).sum();
        let ms = elapsed.as_secs_f64() * 1000.0;
        self.total_ms += ms; self.frame_count += 1; self.frame_times.push(ms);
        self.total_clusters += n_clusters; self.total_noise += noise;
        self.total_input_n += self.last_input_n;
        self.total_cluster_pts += cluster_total_pts;
        // 每帧分布
        self.min_clusters_per_frame = self.min_clusters_per_frame.min(n_clusters);
        self.max_clusters_per_frame = self.max_clusters_per_frame.max(n_clusters);
        self.min_noise_per_frame = self.min_noise_per_frame.min(noise);
        self.max_noise_per_frame = self.max_noise_per_frame.max(noise);
        self.sq_clusters += (n_clusters as u64).pow(2);
        self.sq_noise += (noise as u64).pow(2);
        self.last_clusters = clusters; self.last_noise = noise;
        // 存储降采样后的点云和每个点的聚类分配
        self.last_sampled = sampled;
        let mut assignment = vec![None; self.last_sampled.len()];
        for (ci, obj) in objects.iter().enumerate() {
            for &i in obj {
                assignment[i] = Some(ci);
            }
        }
        self.last_assignment = assignment;
        elapsed
    }
    fn write_frame(&mut self, recorder: &mut BenchRecorder, frame: &FrameData) {
        recorder.begin_frame(frame.frame_idx);
        // 全量写入降采样后的点云（BG 材质），然后根据分类结果更新材质
        for (i, pt) in self.last_sampled.iter().enumerate() {
            let id = i as u64;
            recorder.spawn(spawn_point(*pt, mats::BG).id(id));
        }
        // 根据聚类分配更新材质
        for (i, assignment) in self.last_assignment.iter().enumerate() {
            let id = i as u64;
            match assignment {
                Some(ci) => {
                    let color = CLUSTER_PALETTE[ci % CLUSTER_PALETTE.len()];
                    recorder.set_material(id, color);
                }
                None => {
                    recorder.set_material(id, mats::NOISE);
                }
            }
        }
        // 仅写入包围盒（点云已通过 BG+update 染色，不重复写入）
        let boxes: Vec<Box3D> = self.last_clusters.iter()
            .filter(|c| cluster_points_good_box(c))
            .map(|c| Box3D::from_cloud_aabb(c, 0.05)).collect();
        if !boxes.is_empty() {
            let tagged: Vec<(Box3D, String)> = boxes.iter().enumerate()
                .map(|(i, b)| (b.clone(), format!("c{}", i)))
                .collect();
            recorder.write_boxes(&tagged, mats::CLUSTER_BOX);
        }
        recorder.end_frame();
        let n = self.frame_count.max(1) as f64;
        println!("[{}] 入{} 簇{} 噪{} | {:.0}ms", self.name, self.last_input_n,
            self.last_clusters.len(), self.last_noise, self.total_ms / n);
    }
    fn summarize(&self) {
        let n = self.frame_count.max(1) as f64;
        let avg = self.total_ms / n;
        let cls_range = if self.min_clusters_per_frame <= self.max_clusters_per_frame {
            format!("{}~{}", self.min_clusters_per_frame, self.max_clusters_per_frame)
        } else {
            format!("{}", self.total_clusters as f64 / n)
        };
        println!("  {:<40} | 入{:>5.0} | 簇{:>5} | {:>6.1}ms | {}{}",
            self.name, self.total_input_n as f64 / n, cls_range,
            avg, n as usize,
            if avg > 100.0 { " [OVER]" } else { "" });
    }
    fn stats(&self) -> BenchStats {
        BenchStats { name: self.name.clone(), frame_count: self.frame_count, total_ms: self.total_ms, frame_times: self.frame_times.clone() }
    }
    fn extra_metrics(&self) -> Vec<(String, f64)> {
        let n = self.frame_count.max(1) as f64;
        let avg_cls = self.total_clusters as f64 / n;
        let avg_noi = self.total_noise as f64 / n;
        // 标准差 = sqrt(E[X²] - E[X]²)
        let var_cls = (self.sq_clusters as f64 / n) - avg_cls * avg_cls;
        let var_noi = (self.sq_noise as f64 / n) - avg_noi * avg_noi;
        let std_cls = var_cls.max(0.0).sqrt();
        let std_noi = var_noi.max(0.0).sqrt();
        let avg_cluster_size = if self.total_clusters > 0 { self.total_cluster_pts as f64 / self.total_clusters as f64 } else { 0.0 };
        vec![
            ("avg_input".into(), self.total_input_n as f64 / n),
            ("avg_clusters".into(), avg_cls),
            ("std_clusters".into(), std_cls),
            ("min_clusters".into(), self.min_clusters_per_frame as f64),
            ("max_clusters".into(), self.max_clusters_per_frame as f64),
            ("avg_noise".into(), avg_noi),
            ("std_noise".into(), std_noi),
            ("avg_cluster_size".into(), avg_cluster_size),
        ]
    }
}

// ── 策略工厂 ──────────────────────────────────────────────

fn build_cluster_strategy(cli: &CliArgs) -> Box<dyn ClusteringStrategy> {
    let cfg = fixif();
    let strat = cli.strategy().unwrap_or(cfg.cluster.strategy.clone());
    let eps = cli.get("eps", 0.20f32);
    let min_pts = cli.get("min-pts", 5usize);
    let voxel_size = cli.get("voxel-size", 0.0f32);
    let min_occ = cli.get("min-occ", 3usize);
    let cell_size = cli.get("cell-size", 0.30f32);
    let wall: Box<dyn WallPickStrategy> = Box::new(
        BevLsd::with_params(0.05, 20).with_min_extent(0.0),
    );
    match strat.as_str() {
        "cc_grid" | "cc" => Box::new(CcCluster::new(cell_size, min_pts).with_denoise(0.20, 3)),
        "ransac" => Box::new(RansacCluster::new(eps, 100, min_pts).with_denoise(0.20, 3)),
        "seq" => Box::new(SeqCluster::new(eps, min_pts).with_denoise(0.20, 3)),
        "xy_grid_dbscan" | "xy_grid_dbscan_grid" => Box::new(XYGridDBSCAN::with_params(wall, cell_size, 3, 12.0, eps, min_pts).with_pre_extracted_wall()),
        "lvdot_grid" | "lvdot" => Box::new(LvdotClusterStrategy::direct(voxel_size, min_occ, eps, min_pts)),
        "lvdot_qt" | "prune_qt" => Box::new(PruneQt::new().with_params(min_occ, eps, min_pts)),
        "xy_dbscan" => Box::new(LvdotClusterStrategy::direct(0.0, 1, eps, min_pts).with_pre_extracted_wall()),
        "dbscan_qt" | "dbscan" | "dbscan_adaptive" => Box::new(DbscanStrategy::with_params(0.10, 0.20, 10, 20, 10, 0.10)),
        "dbscan_grid" => Box::new(DbscanGrid::new(eps, min_pts)),
        "range_image" => Box::new(RangeImageStrategy::new()),
        _ => { eprintln!("未知聚类策略 '{}'，使用 xy_grid_dbscan_grid", strat);
            Box::new(XYGridDBSCAN::with_params(wall, cell_size, 3, 12.0, eps, min_pts)) }
    }
}

fn build_cluster_from_toml(strategy_type: &str, p: &toml::Table) -> Box<dyn ClusteringStrategy> {
    let wall: Box<dyn WallPickStrategy> = Box::new(
        BevLsd::with_params(0.05, 20).with_min_extent(0.0),
    );
    match strategy_type {
        "cc_grid" | "cc" => {
            Box::new(CcCluster::with_params(f32(p, "cell_size"), i(p, "min_pts") as usize,
                p.get("merge_dist").and_then(|v| v.as_integer()).unwrap_or(1) as usize)
                .with_denoise(f32(p, "denoise_radius"), i(p, "denoise_min_pts") as usize))
        },
        "ransac" => {
            let mut r = RansacCluster::new(f32(p, "distance"), i(p, "iterations") as usize, i(p, "min_pts") as usize)
                .with_denoise(f32(p, "denoise_radius"), i(p, "denoise_min_pts") as usize);
            if let Some(mw) = p.get("max_walls").and_then(|v| v.as_integer()) {
                r = r.with_max_clusters(mw as usize);
            }
            Box::new(r)
        },
        "seq" => {
            let mut s = SeqCluster::new(f32(p, "distance"), i(p, "min_pts") as usize)
                .with_denoise(f32(p, "denoise_radius"), i(p, "denoise_min_pts") as usize);
            if let Some(mw) = p.get("max_walls").and_then(|v| v.as_integer()) {
                s = s.with_max_clusters(mw as usize);
            }
            Box::new(s)
        },
        "xy_grid_dbscan" | "xy_grid_dbscan_grid" => Box::new(XYGridDBSCAN::with_params(wall, f32(p, "cell_size"), 3, 12.0, f32(p, "eps"), i(p, "min_pts") as usize).with_pre_extracted_wall()),
        "lvdot_grid" | "lvdot" => Box::new(LvdotClusterStrategy::direct(f32(p, "voxel_size"), i(p, "min_occ") as usize, f32(p, "eps"), i(p, "min_pts") as usize)),
        "lvdot_qt" | "prune_qt" => Box::new(PruneQt::new().with_params(i(p, "min_occ") as usize, f32(p, "eps"), i(p, "min_pts") as usize)),
        "xy_dbscan" => Box::new(LvdotClusterStrategy::direct(0.0, 1, f32(p, "eps"), i(p, "min_pts") as usize).with_pre_extracted_wall()),
        "range_image" => Box::new(RangeImageStrategy::with_params(f32(p, "azimuth"), f32(p, "elevation"), f32(p, "threshold"), i(p, "min_pts") as usize)),
        "dbscan_qt" | "dbscan_adaptive" => Box::new(DbscanStrategy::with_params(f32(p, "patience"), f32(p, "slope"), i(p, "min_pts") as usize, 20, 10, f32(p, "voxel_size"))),
        "dbscan_grid" => Box::new(DbscanGrid::new(f32(p, "eps"), i(p, "min_pts") as usize)),
        _ => panic!("未知聚类策略类型: {}", strategy_type),
    }
}

struct ClusterBuilder;
impl StrategyBuilder for ClusterBuilder {
    fn build(&self, strategy_type: &str, p: &toml::Table) -> Box<dyn BenchStrategy> {
        let strategy = build_cluster_from_toml(strategy_type, p);
        let dirname = perple::bench::param_dirname(strategy_type, p);
        // 墙提已在预处理阶段完成，所有聚类策略的输入统一为去地面+墙体后的点。
        // cc/ransac/seq 等新策略内部自带降噪，只需 non_wall 输入。
        let input_source = match strategy_type {
            "lvdot" | "lvdot_grid" | "lvdot_qt" | "prune_qt" => InputSource::NonGround,
            _ => InputSource::Denoised,
        };
        Box::new(ClusterBenchCase::new(&format!("{}_{}", strategy_type, dirname), strategy, input_source))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_default_env().filter_level(log::LevelFilter::Warn).init();
    let args: Vec<String> = std::env::args().collect();
    let cli = CliArgs::parse(&args[1..]);
    let mode = cli.mode();
    let json = cli.has("json");

    match mode {
        BenchMode::Single => {
            let frame_limit = cli.get("frames", 32usize);
            let name = cli.strategy().unwrap_or_else(|| "xy_grid_dbscan".to_string());
            let single_source = match name.as_str() {
                "lvdot" => InputSource::NonGround,
                _ => InputSource::Denoised,
            };
            let mut strategies: Vec<Box<dyn BenchStrategy>> = vec![
                Box::new(ClusterBenchCase::new(&name, build_cluster_strategy(&cli), single_source))
            ];
            let out = "output/cluster_bench";
            let db_name = name.replace(['=', '.', ' '], "_");
            std::fs::create_dir_all(out)?;
            let rec = BenchRecorder::new(&format!("{}/{}.db", out, db_name))
                .map_err(|e| format!("创建 recorder 失败: {}", e))?;
            let mut recs = vec![rec];
            let harness = BenchHarness::new("./data/cloud", frame_limit);
            let wall_strat = build_wall_strategy(&cli);
            let mut pp: Box<dyn Preprocessor> = Box::new(GroundWallPreprocessor::new(
                Box::new(perple::cloud::ground::PeakScan::new()),
                wall_strat,
            ));
            harness.run(&mut *pp, &mut strategies, &mut recs).await?;
            recs[0].save().map_err(|e| format!("保存失败: {}", e))?;
            output_json_or_table(&strategies, json);
        }
        BenchMode::Quick | BenchMode::Full => {
            let mut pp = GroundWallPreprocessor::default();
            run_toml_bench("cluster", "./data/cloud", mode, &mut pp, &ClusterBuilder).await?;
        }
    }
    Ok(())
}

fn output_json_or_table(strategies: &[Box<dyn BenchStrategy>], json: bool) {
    let mut sorted: Vec<_> = strategies.iter().map(|s| (s.stats(), s.extra_metrics())).collect();
    sorted.sort_by(|a, b| {
        let at = a.0.total_ms / a.0.frame_count.max(1) as f64;
        let bt = b.0.total_ms / b.0.frame_count.max(1) as f64;
        at.partial_cmp(&bt).unwrap_or(std::cmp::Ordering::Equal)
    });
    if json {
        println!("\n=== JSON ===");
        let entries: Vec<String> = sorted.iter().map(|(st, extra)| {
            let base = format!(r#"{{"name":"{}","frames":{},"total_ms":{},"avg_ms":{:.1}"#,
                st.name, st.frame_count, st.total_ms, st.total_ms / st.frame_count.max(1) as f64);
            let e: String = extra.iter().map(|(k, v)| format!(r#","{}":{}"#, k, v)).collect();
            format!("{}{}}}", base, e)
        }).collect();
        println!("[{}]", entries.join(","));
    } else {
        println!("\n=== 按速度升序 ===");
        println!("{:-<90}", "");
        println!("| {:<44} | {:>5} | {:>4} | {:>7} | {:>4} |", "策略", "输入", "簇", "ms/帧", "帧");
        println!("{:-<90}", "");
        println!("{:-<90}", "");
    }
}

/// 从 CLI 参数创建墙体策略。
fn build_wall_strategy(cli: &CliArgs) -> Box<dyn WallPickStrategy> {
    let wall = cli.get("wall", "edlines".to_string());
    match wall.as_str() {
        "edlines" | "bev_lsd" => Box::new(BevLsd::new()),
        "hough" | "bev_hough" => Box::new(perple::cloud::wall::BevHough::new()),
        _ => {
            eprintln!("未知墙体策略 '{}'，使用 bev_lsd", wall);
            Box::new(BevLsd::new())
        }
    }
}

fn cluster_points_good_box(cluster: &[[f32; 3]]) -> bool {
    const FLATNESS_RATIO: f32 = 0.15;
    const MIN_VOLUME: f32 = 0.03;
    if cluster.len() < 3 { return false; }
    let b = Box3D::from_cloud_aabb(cluster, 0.05);
    let w = b.length.max(b.width);
    let h = b.height;
    if h < FLATNESS_RATIO * w { return false; }
    if b.length * b.width * h < MIN_VOLUME { return false; }
    true
}

use perple::bench::get_f32 as f32;
use perple::bench::get_i64 as i;
