use std::time::{Duration, Instant};

use perple::bench::{
    BenchStrategy, BenchStats, BenchHarness, BenchRecorder, FrameData, CliArgs, BenchMode,
    GroundDenoisePreprocessor, run_toml_bench, StrategyBuilder,
    mats, CLUSTER_PALETTE,
};
use perple::cloud::wall::{
    WallPickStrategy, BevLsd, BevEdLines, BevHough,
    cluster_obstacles_with_indices,
};
use perple::utils::boxes::Box3D;
use redra_client::spawn_point;

struct WallBenchCase {
    name: String,
    strategy: Box<dyn WallPickStrategy>,
    min_box_pts: usize,
    total_ms: f64, frame_count: usize,
    total_wall_ms: f64, total_post_ms: f64,
    total_wall_points: usize, total_input: usize,
    frame_times: Vec<f64>,
    last_n_wall: usize, last_cloud: Vec<[f32; 3]>,
    last_near_boxes: Vec<Box3D>,
    last_far_boxes: Vec<Box3D>,
    last_far_clouds: Vec<Vec<[f32; 3]>>, last_far_distances: Vec<f32>,
    last_n_discarded: usize,
    total_discarded_pts: usize,
    last_near_indices: Vec<Vec<usize>>,
    last_far_indices: Vec<Vec<usize>>,
}

impl WallBenchCase {
    fn new(name: &str, strategy: Box<dyn WallPickStrategy>, min_box_pts: usize) -> Self {
        Self {
            name: name.to_string(), strategy, min_box_pts,
            total_ms: 0.0, frame_count: 0, total_wall_points: 0, total_input: 0,
            total_wall_ms: 0.0, total_post_ms: 0.0,
            frame_times: Vec::new(), last_n_wall: 0,
            last_cloud: Vec::new(),
            last_near_boxes: Vec::new(), last_far_boxes: Vec::new(), last_far_clouds: Vec::new(),
            last_far_distances: Vec::new(), last_n_discarded: 0, total_discarded_pts: 0,
            last_near_indices: Vec::new(), last_far_indices: Vec::new(),
        }
    }
}

impl BenchStrategy for WallBenchCase {
    fn name(&self) -> &str { &self.name }
    fn run(&mut self, frame: &FrameData) -> Duration {
        let mut cloud = frame.non_ground().to_vec();
        self.total_input += cloud.len();

        let wall_start = Instant::now();
        let (n_wall, _planes) = self.strategy.pick(&mut cloud);
        let wall_ms = wall_start.elapsed().as_secs_f64() * 1000.0;

        let post_start = Instant::now();

        let remaining = &cloud[n_wall..];
        let (all_boxes, all_indices) =
            cluster_obstacles_with_indices(remaining, 0.30, 3, 0.05, 0.0);
        let post_ms = post_start.elapsed().as_secs_f64() * 1000.0;

        let elapsed = Duration::from_secs_f64((wall_ms + post_ms) / 1000.0);

        let max_d2 = 12.0f32 * 12.0;
        let mut near_boxes = Vec::new();
        let mut far_boxes = Vec::new();
        let mut far_clouds = Vec::new();
        let mut far_distances = Vec::new();
        let mut n_discarded = 0;
        let mut near_indices = Vec::new();
        let mut far_indices = Vec::new();

        for (b, indices) in all_boxes.into_iter().zip(all_indices.into_iter()) {
            let mut discard = false;
            if indices.len() < self.min_box_pts { discard = true; }
            if discard { n_discarded += indices.len(); continue; }
            let abs_indices: Vec<usize> = indices.iter().map(|ri| ri + n_wall).collect();
            let c = b.center();
            let d2 = c[0] * c[0] + c[1] * c[1];
            if d2 <= max_d2 { near_boxes.push(b); near_indices.push(abs_indices); }
            else {
                let cluster_pts: Vec<[f32; 3]> = indices.iter().map(|&ri| remaining[ri]).collect();
                far_boxes.push(b); far_clouds.push(cluster_pts); far_distances.push(d2.sqrt()); far_indices.push(abs_indices);
            }
        }

        let ms = wall_ms + post_ms;
        self.total_ms += ms; self.frame_count += 1; self.frame_times.push(ms);
        self.total_wall_ms += wall_ms; self.total_post_ms += post_ms;
        self.total_wall_points += n_wall; self.last_n_wall = n_wall;
        self.last_cloud = cloud;
        self.last_near_boxes = near_boxes; self.last_far_boxes = far_boxes; self.last_far_clouds = far_clouds;
        self.last_far_distances = far_distances; self.last_n_discarded = n_discarded;
        self.total_discarded_pts += n_discarded;
        self.last_near_indices = near_indices; self.last_far_indices = far_indices;
        elapsed
    }
    fn write_frame(&mut self, recorder: &mut BenchRecorder, frame: &FrameData) {
        recorder.begin_frame(frame.frame_idx);
        let n = self.frame_count.max(1) as f64;

        for (i, pt) in self.last_cloud.iter().enumerate() {
            let id = i as u64;
            recorder.spawn(spawn_point(*pt, mats::BG).id(id));
        }

        for i in 0..self.last_n_wall {
            recorder.set_material(i as u64, mats::WALL);
        }

        for (ci, indices) in self.last_near_indices.iter().enumerate() {
            let color = CLUSTER_PALETTE[ci % CLUSTER_PALETTE.len()];
            for &abs_idx in indices {
                recorder.set_material(abs_idx as u64, color);
            }
        }

        for indices in &self.last_far_indices {
            for &abs_idx in indices {
                recorder.set_material(abs_idx as u64, mats::FAR_BOX);
            }
        }

        recorder.write_boxes(
            &self.last_near_boxes.iter().enumerate().map(|(i, b)| (b.clone(), format!("n{}", i))).collect::<Vec<_>>(),
            mats::WALL_BOX,
        );
        let far_tagged: Vec<(Box3D, String)> = self.last_far_boxes.iter().enumerate()
            .map(|(i, b)| (b.clone(), format!("far{}_{:.0}m", i,
                self.last_far_distances.get(i).copied().unwrap_or(0.0))))
            .collect();
        recorder.write_boxes(&far_tagged, mats::FAR_BOX);

        println!("[{}] 墙={} 近={} 远={} 弃={} | {:.0}ms",
            self.name, self.last_n_wall, self.last_near_indices.len(),
            self.last_far_clouds.len(), self.last_n_discarded, self.total_ms / n);
        recorder.end_frame();
    }
    fn summarize(&self) {
        let n = self.frame_count.max(1) as f64;
        let avg_ms = self.total_ms / n;
        let avg_near = self.last_near_indices.len() as f64 / n;
        let avg_far = self.last_far_clouds.len() as f64 / n;
        let avg_wall_pts = self.total_wall_points as f64 / n;
        println!("  {:<40} | 墙点 {:>6.0} | 近 {:>4.1} 远 {:>4.1} | {:>6.1}ms | {} 帧{}",
            self.name, avg_wall_pts, avg_near, avg_far, avg_ms, self.frame_count,
            if avg_ms > 100.0 { " [OVER 100ms]" } else { "" });
    }
    fn stats(&self) -> BenchStats {
        BenchStats { name: self.name.clone(), frame_count: self.frame_count, total_ms: self.total_ms, frame_times: self.frame_times.clone() }
    }
    fn extra_metrics(&self) -> Vec<(String, f64)> {
        let n = self.frame_count.max(1) as f64;
        let avg_wall = self.total_wall_points as f64 / n;
        let avg_input = self.total_input as f64 / n;
        let wall_ratio = if avg_input > 0.0 { avg_wall / avg_input * 100.0 } else { 0.0 };
        vec![
            ("avg_input".into(), avg_input),
            ("avg_wall_pts".into(), avg_wall),
            ("wall_ratio".into(), wall_ratio),
            ("avg_wall_ms".into(), self.total_wall_ms / n),
            ("avg_post_ms".into(), self.total_post_ms / n),
            ("avg_obstacles".into(), self.last_near_indices.len() as f64 / n),
            ("avg_far_obstacles".into(), self.last_far_clouds.len() as f64 / n),
            ("avg_discarded".into(), self.total_discarded_pts as f64 / n),
        ]
    }
}

fn build_wall_strategy(cli: &CliArgs) -> Box<dyn WallPickStrategy> {
    let strat = cli.get::<String>("strategy", "bev_lsd".to_string());
    let distance = cli.get("distance", 0.05f32);
    match strat.as_str() {
        "bev_hough" => {
            let mut s = BevHough::with_params(distance, cli.get("min-wall-pts", 30usize));
            let ext = cli.get("min-extent", 0.0f32);
            if ext > 0.0 { s = s.with_min_extent(ext); }
            let ht = cli.get("hough-threshold", 0.0f32);
            if ht > 0.0 { s = s.with_hough_threshold(ht); }
            Box::new(s)
        },
        "bev_edlines" => {
            let mut s = BevEdLines::with_params(distance, cli.get("min-wall-pts", 30usize));
            let ext = cli.get("min-extent", 0.0f32);
            if ext > 0.0 { s = s.with_min_extent(ext); }
            Box::new(s)
        },
        _ => {
            let mut s = BevLsd::with_params(distance, cli.get("min-wall-pts", 30usize));
            let ext = cli.get("min-extent", 0.0f32);
            if ext > 0.0 { s = s.with_min_extent(ext); }
            let gt = cli.get("grad-threshold", 0.0f32);
            if gt > 0.0 { s = s.with_grad_threshold(gt); }
            let at = cli.get("angle-tolerance", 0.0f32);
            if at > 0.0 { s = s.with_angle_tolerance(at); }
            Box::new(s)
        },
    }
}

fn build_wall_from_toml(strategy_type: &str, p: &toml::Table) -> Box<dyn WallPickStrategy> {
    match strategy_type {
        "bev_hough" => {
            let mut s = BevHough::with_params(perple::bench::get_f32(p, "distance"), perple::bench::get_i64(p, "min_wall_pts") as usize);
            if let Some(ext) = p.get("min_extent").and_then(|v| v.as_float()) {
                s = s.with_min_extent(ext as f32);
            }
            if let Some(ht) = p.get("hough_threshold").and_then(|v| v.as_float()) {
                s = s.with_hough_threshold(ht as f32);
            }
            Box::new(s)
        },
        "bev_edlines" => {
            let mut s = BevEdLines::with_params(perple::bench::get_f32(p, "distance"), perple::bench::get_i64(p, "min_wall_pts") as usize);
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
        },
        _ => {
            let mut s = BevLsd::with_params(perple::bench::get_f32(p, "distance"), perple::bench::get_i64(p, "min_wall_pts") as usize);
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
        },
    }
}

struct WallBuilder;
impl StrategyBuilder for WallBuilder {
    fn build(&self, strategy_type: &str, p: &toml::Table) -> Box<dyn BenchStrategy> {
        let strategy = build_wall_from_toml(strategy_type, p);
        let dirname = perple::bench::param_dirname(strategy_type, p);
        Box::new(WallBenchCase::new(&format!("{}_{}", strategy_type, dirname), strategy, 20))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_default_env().filter_level(log::LevelFilter::Warn).init();
    let args: Vec<String> = std::env::args().collect();
    let cli = CliArgs::parse(&args[1..]);
    let mode = cli.mode();
    let sweep = cli.has("sweep");
    let json = cli.has("json");
    let frame_limit = cli.get("frames", 10usize);

    let effective_mode = if sweep { BenchMode::Quick } else { mode };

    match effective_mode {
        BenchMode::Single => {
            let name = cli.get::<String>("strategy", "bev_lsd".to_string());
            let min_box_pts = cli.get("min-box-pts", 20usize);
            let mut strategies: Vec<Box<dyn BenchStrategy>> = vec![
                Box::new(WallBenchCase::new(&name, build_wall_strategy(&cli), min_box_pts))
            ];
            let out = "output/wall_bench";
            let db_name = name.replace(['=', '.', ' '], "_");
            std::fs::create_dir_all(out)?;
            let rec = BenchRecorder::new(&format!("{}/{}.db", out, db_name))
                .map_err(|e| format!("创建 recorder 失败: {}", e))?;
            let mut recs = vec![rec];
            let harness = BenchHarness::new("./data/cloud", frame_limit);
            let mut pp = GroundDenoisePreprocessor::default();
            harness.run(&mut pp, &mut strategies, &mut recs).await?;
            recs[0].save().map_err(|e| format!("保存失败: {}", e))?;
            output_json_or_table(&strategies, json);
        }
        BenchMode::Quick | BenchMode::Full => {
            let mut pp = GroundDenoisePreprocessor::default();
            run_toml_bench("wall", "./data/cloud", effective_mode, &mut pp, &WallBuilder).await?;
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
        println!("| {:<40} | {:>6} | {:>8} | {:>7} | {:>4} |", "策略", "墙点", "近/远", "ms/帧", "帧");
        println!("{:-<90}", "");
        println!("{:-<90}", "");
    }
}
