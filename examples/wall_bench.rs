use std::time::{Duration, Instant};

use perple::bench::{
    BenchStrategy, BenchStats, BenchHarness, BenchRecorder, FrameData, CliArgs, BenchMode,
    WallPreprocessor, run_toml_bench, StrategyBuilder,
    mats, CLUSTER_PALETTE,
};
use perple::cloud::wall::{
    WallPickStrategy, TopDownCluster, XYRansacWall, NormalWall, QuadtreeWall,
    AdaptiveDBSCANWall, SequentialFit, XYDBSCANWall, Downsampler,
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
        let (all_boxes, all_indices) =
            cluster_obstacles_with_indices(&cloud[n_wall..], 0.30, 3, 0.05, 0.0);
        let post_ms = post_start.elapsed().as_secs_f64() * 1000.0;

        let elapsed = Duration::from_secs_f64((wall_ms + post_ms) / 1000.0);

        let wall_nz_threshold: f32 = 0.15;
        let remaining = &cloud[n_wall..];
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
            if !discard {
                if let Some((normal, _)) = fit_plane_3d_wallbench(&indices, remaining, b.height) {
                    if normal[2].abs() < wall_nz_threshold { discard = true; }
                }
            }
            if discard { n_discarded += indices.len(); continue; }
            let abs_indices: Vec<usize> = indices.iter().map(|ri| ri + n_wall).collect();
            let cluster_pts: Vec<[f32; 3]> = indices.iter().map(|&ri| remaining[ri]).collect();
            let c = b.center();
            let d2 = c[0] * c[0] + c[1] * c[1];
            if d2 <= max_d2 { near_boxes.push(b); near_indices.push(abs_indices); }
            else { far_boxes.push(b); far_clouds.push(cluster_pts); far_distances.push(d2.sqrt()); far_indices.push(abs_indices); }
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

        // 全量写入点云（SQLite 无内存累积，不再需要降采样）
        for (i, pt) in self.last_cloud.iter().enumerate() {
            let id = i as u64;
            recorder.spawn(spawn_point(*pt, mats::BG).id(id));
        }

        // 更新墙面点材质
        for i in 0..self.last_n_wall {
            recorder.set_material(i as u64, mats::WALL);
        }

        // 更新近距聚类点材质
        for (ci, indices) in self.last_near_indices.iter().enumerate() {
            let color = CLUSTER_PALETTE[ci % CLUSTER_PALETTE.len()];
            for &abs_idx in indices {
                recorder.set_material(abs_idx as u64, color);
            }
        }

        // 更新远距聚类点材质
        for indices in &self.last_far_indices {
            for &abs_idx in indices {
                recorder.set_material(abs_idx as u64, mats::FAR_BOX);
            }
        }

        // 剩余点云（非墙面点）的障碍物检测框
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

// ── 策略工厂 ──────────────────────────────────────────────

fn build_wall_strategy(cli: &CliArgs) -> Box<dyn WallPickStrategy> {
    let strat = cli.get::<String>("strategy", "xy_ransac".to_string());
    let distance = cli.get("distance", 0.05f32);
    let iterations = cli.get("iterations", 50usize);
    let seed = cli.get("seed", 42u64);
    let cell_size = cli.get("cell-size", 0.10f32);
    let min_density = cli.get("min-density", 5usize);
    let merge_dist = cli.get("merge-dist", 2usize);
    let normal_threshold = cli.get("normal-threshold", 0.17f32);
    let min_pts = cli.get("min-pts", 10usize);
    match strat.as_str() {
        "top_down" => Box::new(TopDownCluster::with_params(cell_size, min_density, merge_dist)
            .with_width_ratio(normal_threshold)),
        "xy_ransac" => Box::new(XYRansacWall::with_params(distance, iterations, 30).with_seed(seed)),
        "normal_wall" => Box::new(NormalWall::with_params(cell_size, min_pts, 30.0)
            .with_normal_threshold(normal_threshold)),
        "quadtree" => Box::new(QuadtreeWall::with_params(cell_size, min_pts, 0.5)),
        "adaptive_dbscan" => Box::new(AdaptiveDBSCANWall::with_params(0.10, 2.0, min_pts)),
        "xy_dbscan" => Box::new(AdaptiveDBSCANWall::with_params(0.0, 1.0, min_pts)),
        "seq_fit" => Box::new(SequentialFit::with_params(distance, normal_threshold, cli.get("max-walls", 5usize))),
        "xy_dbscan_wall" => Box::new(XYDBSCANWall::with_params(cli.get("eps", 0.15f32), cli.get("min-pts", 5usize), cli.get("min-z-span", 1.5f32))),
        _ => { eprintln!("未知墙体策略 '{}'，使用 xy_ransac", strat);
            Box::new(XYRansacWall::with_params(distance, iterations, 30).with_seed(seed)) }
    }
}

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

    // 传统 --sweep 标志兼容（映射到 BenchMode::Quick）
    let effective_mode = if sweep { BenchMode::Quick } else { mode };

    match effective_mode {
        BenchMode::Single => {
            let name = cli.get::<String>("strategy", "xy_ransac".to_string());
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
            let harness = BenchHarness::new("./data/test", frame_limit);
            let mut pp = WallPreprocessor::default();
            harness.run(&mut pp, &mut strategies, &mut recs).await?;
            recs[0].save().map_err(|e| format!("保存失败: {}", e))?;
            output_json_or_table(&strategies, json);
        }
        BenchMode::Quick | BenchMode::Full => {
            let mut pp = WallPreprocessor::default();
            run_toml_bench("wall", "./data/test", effective_mode, &mut pp, &WallBuilder).await?;
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

use perple::bench::get_f32 as f;
use perple::bench::get_i64 as i;

fn fit_plane_3d_wallbench(indices: &[usize], points: &[[f32; 3]], box_h: f32) -> Option<([f32; 3], f32)> {
    let n = indices.len();
    if n < 10 { return None; }
    if box_h < 0.8 { return None; }
    let nf = n as f32;
    let mut cx = 0.0f32; let mut cy = 0.0f32; let mut cz = 0.0f32;
    for &i in indices { let p = &points[i]; cx += p[0]; cy += p[1]; cz += p[2]; }
    cx /= nf; cy /= nf; cz /= nf;
    let mut cov = nalgebra::Matrix3::zeros();
    for &i in indices {
        let p = &points[i];
        let dx = p[0] - cx; let dy = p[1] - cy; let dz = p[2] - cz;
        cov[(0, 0)] += dx * dx; cov[(0, 1)] += dx * dy; cov[(0, 2)] += dx * dz;
        cov[(1, 1)] += dy * dy; cov[(1, 2)] += dy * dz; cov[(2, 2)] += dz * dz;
    }
    cov /= nf;
    cov[(1, 0)] = cov[(0, 1)]; cov[(2, 0)] = cov[(0, 2)]; cov[(2, 1)] = cov[(1, 2)];
    let eig = cov.symmetric_eigen();
    let mut min_idx = 0;
    let mut min_val = eig.eigenvalues[0];
    for i in 1..3 { if eig.eigenvalues[i] < min_val { min_val = eig.eigenvalues[i]; min_idx = i; } }
    let nv = eig.eigenvectors.column(min_idx);
    Some(([nv[0], nv[1], nv[2]], -(nv[0] * cx + nv[1] * cy + nv[2] * cz)))
}
