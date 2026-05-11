use std::time::{Duration, Instant};

use perple::bench::{
    BenchStrategy, BenchStats, BenchHarness, BenchRecorder, FrameData, CliArgs, BenchMode,
    WallPreprocessor,
    run_toml_bench, StrategyBuilder,
    mats,
};
use perple::cloud::denoise::{DenoiseStrategy, RadiusOutlierRemoval};
use redra_client::spawn_point;

struct DenoiseBenchCase {
    name: String,
    strategy: Box<dyn DenoiseStrategy>,
    total_ms: f64, frame_count: usize,
    frame_times: Vec<f64>,
    last_input: usize, last_output: usize,
    total_input: usize, total_output: usize,
}

impl DenoiseBenchCase {
    fn new(name: &str, strategy: Box<dyn DenoiseStrategy>) -> Self {
        Self {
            name: name.to_string(), strategy,
            total_ms: 0.0, frame_count: 0, frame_times: Vec::new(),
            last_input: 0, last_output: 0,
            total_input: 0, total_output: 0,
        }
    }
}

impl BenchStrategy for DenoiseBenchCase {
    fn name(&self) -> &str { &self.name }
    fn run(&mut self, frame: &FrameData) -> Duration {
        let input = frame.non_wall();
        self.last_input = input.len();
        let start = Instant::now();
        let (output, _) = self.strategy.denoise(input);
        let elapsed = start.elapsed();
        self.last_output = output.len();
        let ms = elapsed.as_secs_f64() * 1000.0;
        self.total_ms += ms; self.frame_count += 1; self.frame_times.push(ms);
        self.total_input += self.last_input;
        self.total_output += self.last_output;
        elapsed
    }
    fn write_frame(&mut self, recorder: &mut BenchRecorder, frame: &FrameData) {
        recorder.begin_frame(frame.frame_idx);
        let n = self.frame_count.max(1) as f64;

        // 全量写入点云（SQLite 无内存累积，不再需要降采样）
        let cloud = frame.non_wall();
        for (i, pt) in cloud.iter().enumerate() {
            let id = i as u64;
            recorder.spawn(spawn_point(*pt, mats::BG).id(id));
        }

        println!("[{}] 入{} 出{} 保留{:.0}% | {:.0}ms",
            self.name, self.last_input, self.last_output,
            if self.last_input > 0 { self.last_output as f64 / self.last_input as f64 * 100.0 } else { 0.0 },
            self.total_ms / n);
        recorder.end_frame();
    }
    fn summarize(&self) {
        let n = self.frame_count.max(1) as f64;
        let avg_ms = self.total_ms / n;
        let retention = if self.total_input > 0 { self.total_output as f64 / self.total_input as f64 * 100.0 } else { 0.0 };
        println!("  {:<40} | 入{:>5.0} 保留{:>4.0}% | {:>6.1}ms | {} 帧{}",
            self.name, self.total_input as f64 / n, retention, avg_ms, self.frame_count,
            if avg_ms > 100.0 { " [OVER 100ms]" } else { "" });
    }
    fn stats(&self) -> BenchStats {
        BenchStats { name: self.name.clone(), frame_count: self.frame_count, total_ms: self.total_ms, frame_times: self.frame_times.clone() }
    }
    fn extra_metrics(&self) -> Vec<(String, f64)> {
        let n = self.frame_count.max(1) as f64;
        let retention = if self.total_input > 0 { self.total_output as f64 / self.total_input as f64 * 100.0 } else { 0.0 };
        vec![
            ("avg_input".into(), self.total_input as f64 / n),
            ("avg_output".into(), self.total_output as f64 / n),
            ("retention_pct".into(), retention),
        ]
    }
}

// ── 策略工厂 ──────────────────────────────────────────────

fn build_denoise_from_toml(strategy_type: &str, p: &toml::Table) -> Box<dyn DenoiseStrategy> {
    match strategy_type {
        "radius_outlier" => Box::new(RadiusOutlierRemoval::new(f(p, "radius"), i(p, "min_pts") as usize)),
        _ => panic!("未知降噪策略类型: {}", strategy_type),
    }
}

struct DenoiseBuilder;
impl StrategyBuilder for DenoiseBuilder {
    fn build(&self, strategy_type: &str, p: &toml::Table) -> Box<dyn BenchStrategy> {
        let strategy = build_denoise_from_toml(strategy_type, p);
        let dirname = perple::bench::param_dirname(strategy_type, p);
        Box::new(DenoiseBenchCase::new(&format!("{}_{}", strategy_type, dirname), strategy))
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
            let radius = cli.get("radius", 0.20f32);
            let min_pts = cli.get("min-pts", 3usize);
            let strategy: Box<dyn DenoiseStrategy> = Box::new(RadiusOutlierRemoval::new(radius, min_pts));
            let mut strategies: Vec<Box<dyn BenchStrategy>> = vec![
                Box::new(DenoiseBenchCase::new("radius_outlier", strategy))
            ];
            let out = "output/denoise_bench";
            std::fs::create_dir_all(out)?;
            let rec = BenchRecorder::new(&format!("{}/radius_outlier.db", out))
                .map_err(|e| format!("创建 recorder 失败: {}", e))?;
            let mut recs = vec![rec];
            let harness = BenchHarness::new("./data/cloud", frame_limit);
            let mut pp = WallPreprocessor::default();
            harness.run(&mut pp, &mut strategies, &mut recs).await?;
            recs[0].save().map_err(|e| format!("保存失败: {}", e))?;
            output_json_or_table(&strategies, json);
        }
        BenchMode::Quick | BenchMode::Full => {
            let mut pp = WallPreprocessor::default();
            run_toml_bench("denoise", "./data/cloud", mode, &mut pp, &DenoiseBuilder).await?;
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
        println!("| {:<36} | {:>5} | {:>6} | {:>4} |", "策略", "入", "保留%", "帧");
        println!("{:-<80}", "");
        println!("{:-<80}", "");
    }
}

use perple::bench::get_f32 as f;
use perple::bench::get_i64 as i;
