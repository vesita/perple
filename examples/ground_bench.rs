use std::time::{Duration, Instant};

use perple::bench::{
    BenchStrategy, BenchStats, BenchHarness, BenchRecorder, FrameData, CliArgs, BenchMode,
    PassthroughPreprocessor,
    run_toml_bench,
    mats,
};
use perple::cloud::ground::*;
use perple::config::fixif;
use perple::utils::boxes::Box3D;
use redra_client::spawn_point;

struct GroundBenchCase {
    name: String,
    strategy: Box<dyn GroundPickStrategy>,
    total_ms: f64, frame_count: usize,
    frame_times: Vec<f64>,
    last_n_ground: usize, last_cloud: Vec<[f32; 3]>,
    last_ground_box: Option<Box3D>,
    total_ground_pts: usize, total_input_pts: usize,
}

impl GroundBenchCase {
    fn new(name: &str, strategy: Box<dyn GroundPickStrategy>) -> Self {
        Self {
            name: name.to_string(), strategy,
            total_ms: 0.0, frame_count: 0, frame_times: Vec::new(),
            last_n_ground: 0, last_cloud: Vec::new(),
            last_ground_box: None,
            total_ground_pts: 0, total_input_pts: 0,
        }
    }
}

impl BenchStrategy for GroundBenchCase {
    fn name(&self) -> &str { &self.name }
    fn run(&mut self, frame: &FrameData) -> Duration {
        let mut cloud = frame.cloud.to_vec();
        let start = Instant::now();
        let (n_ground, _, _) = self.strategy.pick(&mut cloud);
        let elapsed = start.elapsed();

        let ground_box = if n_ground >= 3 {
            Some(Box3D::from_cloud_aabb(&cloud[..n_ground], 0.0))
        } else {
            None
        };

        self.total_ms += elapsed.as_secs_f64() * 1000.0;
        self.frame_count += 1; self.frame_times.push(elapsed.as_secs_f64() * 1000.0);
        self.total_ground_pts += n_ground; self.total_input_pts += cloud.len();
        self.last_n_ground = n_ground; self.last_cloud = cloud;
        self.last_ground_box = ground_box;
        elapsed
    }
    fn write_frame(&mut self, recorder: &mut BenchRecorder, frame: &FrameData) {
        recorder.begin_frame(frame.frame_idx);

        // 全量写入点云（SQLite 无内存累积，不再需要降采样）
        for (i, pt) in self.last_cloud.iter().enumerate() {
            let id = i as u64;
            recorder.spawn(spawn_point(*pt, mats::BG).id(id));
        }

        // 更新地面点材质为绿色
        for i in 0..self.last_n_ground {
            recorder.set_material(i as u64, mats::GROUND);
        }

    // 地面区域包围盒（半透明，不遮挡点云）
    if let Some(ref bx) = self.last_ground_box {
        recorder.write_boxes(&[(bx.clone(), "ground".into())], mats::GROUND_BOX);
    }

        if self.last_n_ground > 0 {
            let z_min = self.last_cloud[..self.last_n_ground].iter().map(|p| p[2]).fold(f32::INFINITY, f32::min);
            let z_max = self.last_cloud[..self.last_n_ground].iter().map(|p| p[2]).fold(f32::NEG_INFINITY, f32::max);
            let ratio = self.last_n_ground as f64 / self.last_cloud.len() as f64 * 100.0;
            let avg_ms = self.total_ms / self.frame_count.max(1) as f64;
            println!("[{}] 地面 {}/{} ({:.0}%) Z=[{:.2},{:.2}] {:.1}ms",
                self.name, self.last_n_ground, self.last_cloud.len(), ratio, z_min, z_max, avg_ms);
        }
        recorder.end_frame();
    }
    fn summarize(&self) {
        let n = self.frame_count.max(1) as f64;
        let avg_ms = self.total_ms / n;
        println!("  {:<36} | 平均 {:>7.1}ms | {} 帧{}", self.name, avg_ms, self.frame_count,
            if avg_ms > 100.0 { " [OVER 100ms]" } else { "" });
    }
    fn stats(&self) -> BenchStats {
        BenchStats { name: self.name.clone(), frame_count: self.frame_count, total_ms: self.total_ms, frame_times: self.frame_times.clone() }
    }
    fn extra_metrics(&self) -> Vec<(String, f64)> {
        let n = self.frame_count.max(1) as f64;
        vec![
            ("avg_ground".into(), self.total_ground_pts as f64 / n),
            ("avg_total".into(), self.total_input_pts as f64 / n),
            ("ground_ratio".into(), self.total_ground_pts as f64 / self.total_input_pts.max(1) as f64 * 100.0),
        ]
    }
}

// ── 策略工厂 ──────────────────────────────────────────────

fn build_ground_strategy(cli: &CliArgs) -> Box<dyn GroundPickStrategy> {
    let cfg = fixif();
    let strat = cli.get::<String>("strategy", cfg.ground_strategy.clone());
    let expand = cli.get("expand", cfg.ground_expand);
    let distance = cli.get("distance", cfg.ground_ransac_distance);
    let iterations = cli.get("iterations", cfg.ground_ransac_iterations);
    let n_lpr = cli.get("n-lpr", 100usize);
    let th_seed = cli.get("th-seed", 0.5f32);
    let th_dist = cli.get("th-dist", 0.3f32);
    match strat.as_str() {
        "histogram" => Box::new(HistogramExpand::with_expand(expand)),
        "peak_scan" => Box::new(PeakScan::with_params(0.10, expand)),
        "ransac" => Box::new(RansacGround::with_params(distance, iterations)),
        "histoseed" => Box::new(HistoseedPlane::with_params(expand, distance, iterations)),
        "gpf" => Box::new(GpfGround::with_params(n_lpr, th_seed, th_dist)),
        _ => { eprintln!("未知地面策略 '{}'，使用 peak_scan", strat); Box::new(PeakScan::with_params(0.10, expand)) }
    }
}

/// 从 TOML 参数字典构建策略。
fn build_ground_from_toml(strategy_type: &str, p: &toml::Table) -> Box<dyn GroundPickStrategy> {
    match strategy_type {
        "histogram" => Box::new(HistogramExpand::with_expand(f(p, "expand"))),
        "peak_scan" => Box::new(PeakScan::with_params(f(p, "threshold"), f(p, "expand"))),
        "ransac" => Box::new(RansacGround::with_params(f(p, "distance"), i(p, "iterations") as usize)),
        "histoseed" => Box::new(HistoseedPlane::with_params(f(p, "expand"), f(p, "distance"), i(p, "iterations") as usize)),
        "gpf" => Box::new(GpfGround::with_params(i(p, "n_lpr") as usize, f(p, "th_seed"), f(p, "th_dist"))),
        _ => panic!("未知地面策略类型: {}", strategy_type),
    }
}

/// TOML 策略构建器：包装 ground 策略为 BenchStrategy。
struct GroundBuilder;
impl perple::bench::StrategyBuilder for GroundBuilder {
    fn build(&self, strategy_type: &str, p: &toml::Table) -> Box<dyn BenchStrategy> {
        let strategy = build_ground_from_toml(strategy_type, p);
        let dirname = perple::bench::param_dirname(strategy_type, p);
        Box::new(GroundBenchCase::new(&format!("{}_{}", strategy_type, dirname), strategy))
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
            let frame_limit = cli.get("frames", 5usize);
            let name = cli.get::<String>("strategy", "peak_scan".to_string());
            let mut strategies: Vec<Box<dyn BenchStrategy>> = vec![
                Box::new(GroundBenchCase::new(&name, build_ground_strategy(&cli)))
            ];
            let out = "output/ground_bench";
            let db_name = name.replace(['=', '.', ' '], "_");
            std::fs::create_dir_all(out)?;
            let rec = BenchRecorder::new(&format!("{}/{}.db", out, db_name))
                .map_err(|e| format!("创建 recorder 失败: {}", e))?;
            let mut recs = vec![rec];
            let harness = BenchHarness::new("./data/test", frame_limit);
            let mut pp = PassthroughPreprocessor;
            harness.run(&mut pp, &mut strategies, &mut recs).await?;
            recs[0].save().map_err(|e| format!("保存失败: {}", e))?;

            output_json_or_table(&strategies, json);
        }
        BenchMode::Quick | BenchMode::Full => {
            let mut pp = PassthroughPreprocessor;
            run_toml_bench("ground", "./data/test", mode, &mut pp, &GroundBuilder).await?;
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
        println!("{:-<80}", "");
        println!("| {:<36} | {:>7} | {:>4} |", "策略", "ms/帧", "帧");
        println!("{:-<80}", "");
        println!("{:-<80}", "");
    }
}

use perple::bench::get_f32 as f;
use perple::bench::get_i64 as i;
