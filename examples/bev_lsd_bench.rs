/// BevLsd 参数扫描 Bench + 检测框输出。
///
/// 管线：地面检测 → 降噪 → BevLsd 墙体检测 → XY DBSCAN(固定参数) 后聚类
/// 对所有参数组合输出检测框到 SQLite，供可视化查看。
///
/// 用法：
///   cargo run --example bev_lsd_bench
///   cargo run --example bev_lsd_bench -- --frames=5
use std::time::Instant;

use perple::bench::{CliArgs, BenchRecorder, mats, CLUSTER_PALETTE};
use perple::cloud::wall::{BevLsd, WallPickStrategy, cluster_obstacles_with_indices};
use perple::optional::data_loader::DataLoader;
use perple::swapl::global_swapl;
use perple::utils::boxes::Box3D;
use serde::Serialize;

// ── 固定后聚类参数 ───────────────────────────────────────

const CLUSTER_CELL: f32 = 0.30;
const CLUSTER_MIN_PTS: usize = 3;

// ── 参数组合 ──────────────────────────────────────────────

#[derive(Clone)]
struct LsdParams {
    distance: f32,
    min_wall_pts: usize,
    grad_threshold: f32,
    angle_tolerance: f32,
    min_extent: f32,
}

fn label(p: &LsdParams) -> String {
    format!("d{:.2}_g{:.3}", p.distance, p.grad_threshold)
}

fn generate_params() -> Vec<LsdParams> {
    let distances = [0.06f32, 0.08];
    let grad_thresholds = [0.05f32, 0.08, 0.10];
    let mut params = Vec::new();
    for &d in &distances {
        for &g in &grad_thresholds {
            params.push(LsdParams {
                distance: d,
                min_wall_pts: 20,
                grad_threshold: g,
                angle_tolerance: 30.0,
                min_extent: 0.5,
            });
        }
    }
    params
}

// ── 结果 ──────────────────────────────────────────────────

#[derive(Serialize, Clone)]
struct FrameResult {
    label: String,
    wall_pts: usize,
    total_pts: usize,
    n_clusters: usize,
    cluster_pts: usize,
    noise_pts: usize,
    wall_ms: f64,
    cluster_ms: f64,
    total_ms: f64,
}

#[derive(Serialize)]
struct AggregatedResult {
    combo: String,
    distance: f32,
    grad_threshold: f32,
    avg_wall_pts: f64,
    avg_clusters: f64,
    avg_cluster_pts: f64,
    avg_noise: f64,
    avg_wall_ms: f64,
    avg_cluster_ms: f64,
    avg_total_ms: f64,
    frames: Vec<FrameResult>,
}

// ── 预处理 ────────────────────────────────────────────────

fn preprocess(cloud: &[[f32; 3]]) -> Vec<[f32; 3]> {
    use perple::cloud::ground::PeakScan;
    use perple::cloud::denoise::RadiusOutlierRemoval;
    use perple::cloud::ground::GroundPickStrategy;
    use perple::cloud::denoise::DenoiseStrategy;

    let mut buf = cloud.to_vec();
    let (n_ground, _, _) = PeakScan::new().pick(&mut buf);
    let non_ground = buf[n_ground..].to_vec();
    let (denoised, _) = RadiusOutlierRemoval::new(0.30, 3).denoise(&non_ground);
    denoised
}

// ── 写入一帧到 SQLite ────────────────────────────────────

fn write_frame_db(
    recorder: &mut BenchRecorder,
    frame_idx: usize,
    wall_pts: &[[f32; 3]],
    all_remaining: &[[f32; 3]],
    clusters: &[Vec<usize>],
    boxes: &[Box3D],
) {
    recorder.begin_frame(frame_idx);
    recorder.write_point_cloud(all_remaining, mats::BG, 3000);
    recorder.write_point_cloud(wall_pts, mats::WALL, 2000);

    let cluster_clouds: Vec<Vec<[f32; 3]>> = clusters.iter()
        .map(|indices| indices.iter().map(|&i| all_remaining[i]).collect())
        .collect();

    for (i, cloud) in cluster_clouds.iter().enumerate() {
        if cloud.is_empty() { continue; }
        let color = CLUSTER_PALETTE[i % CLUSTER_PALETTE.len()];
        recorder.write_point_cloud(cloud, color, cloud.len());
    }
    if !boxes.is_empty() {
        let tagged: Vec<(Box3D, String)> = boxes.iter().enumerate()
            .map(|(i, b)| (b.clone(), format!("c{}", i)))
            .collect();
        recorder.write_boxes(&tagged, mats::CLUSTER_BOX);
    }
    recorder.end_frame();
}

// ── 主流程 ────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let cli = CliArgs::parse(&args[1..]);
    let frame_limit = cli.get("frames", 3usize);

    let mut loader = DataLoader::new("./data/cloud".into());
    loader.set_frame_limit(frame_limit);
    loader.load().await?;

    let params = generate_params();
    println!("BevLsd 参数扫描 ({} 组合 × {} 帧)", params.len(), frame_limit);
    println!("后聚类: 网格连通域 cell={:.2} min_pts={}", CLUSTER_CELL, CLUSTER_MIN_PTS);
    println!();

    let out_base = "output/bench/bev_lsd_bench";
    std::fs::create_dir_all(out_base)?;
    let mut recorders: Vec<Option<BenchRecorder>> = params.iter().map(|p| {
        let dir = format!("{}/{}", out_base, label(p));
        std::fs::create_dir_all(&dir).ok()?;
        let db_path = format!("{}/{}.db", dir, label(p));
        BenchRecorder::new(&db_path).ok()
    }).collect();

    let mut accum: std::collections::HashMap<String, Vec<FrameResult>> = std::collections::HashMap::new();
    for p in &params {
        accum.entry(label(p)).or_default();
    }

    let mut frame_idx = 0usize;
    while loader.load_next().await? {
        let cloud: Vec<[f32; 3]> = {
            let swapl = global_swapl();
            let mut stream = swapl.clouds.lock().unwrap();
            match stream.read() {
                Some(data) => data,
                None => continue,
            }
        };

        frame_idx += 1;
        println!("─ 帧 {} ({} 点) ─", frame_idx, cloud.len());

        let non_ground = preprocess(&cloud);
        println!("  非地面点: {}", non_ground.len());

        for (pi, p) in params.iter().enumerate() {
            let wall_start = Instant::now();
            let mut cloud_copy = non_ground.clone();
            let mut ed = BevLsd::with_params(p.distance, p.min_wall_pts)
                .with_min_extent(p.min_extent)
                .with_grad_threshold(p.grad_threshold)
                .with_angle_tolerance(p.angle_tolerance);
            let (n_wall, _) = ed.pick(&mut cloud_copy);
            let wall_ms = wall_start.elapsed().as_secs_f64() * 1000.0;

            let remaining = &cloud_copy[n_wall..];

            let cluster_start = Instant::now();
            let (boxes, clusters) = cluster_obstacles_with_indices(remaining, CLUSTER_CELL, CLUSTER_MIN_PTS, 0.05, 0.0);
            let cluster_ms = cluster_start.elapsed().as_secs_f64() * 1000.0;

            let n_clusters = clusters.len();
            let cluster_pts: usize = clusters.iter().map(|c| c.len()).sum();
            let noise_pts = remaining.len().saturating_sub(cluster_pts);

            let wall_pts = &cloud_copy[..n_wall];
            if let Some(ref mut rec) = recorders[pi] {
                write_frame_db(rec, frame_idx, wall_pts, remaining, &clusters, &boxes);
            }

            let lbl = label(p);
            accum.get_mut(&lbl).unwrap().push(FrameResult {
                label: lbl,
                wall_pts: n_wall,
                total_pts: non_ground.len(),
                n_clusters,
                cluster_pts,
                noise_pts,
                wall_ms,
                cluster_ms,
                total_ms: wall_ms + cluster_ms,
            });
        }
    }

    for rec in recorders.iter_mut().flatten() {
        let _ = rec.save();
    }

    // ── 输出汇总 ──
    println!();
    println!("{}", "=".repeat(120));
    println!("{:<20} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "参数", "墙面点", "障碍簇", "簇点数", "噪声点", "墙体ms", "聚类ms", "合计ms");
    println!("{}", "=".repeat(120));

    let mut sorted: Vec<(String, Vec<FrameResult>)> = accum.into_iter().collect();
    sorted.sort_by(|(_, ra), (_, rb)| {
        let ta: f64 = ra.iter().map(|r| r.total_ms).sum::<f64>() / ra.len() as f64;
        let tb: f64 = rb.iter().map(|r| r.total_ms).sum::<f64>() / rb.len() as f64;
        ta.partial_cmp(&tb).unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut aggregated: Vec<AggregatedResult> = Vec::new();
    for (lbl, results) in &sorted {
        let n = results.len() as f64;
        let _avg_wall: f64 = results.iter().map(|r| r.wall_pts as f64).sum::<f64>() / n;
        let avg_clusters: f64 = results.iter().map(|r| r.n_clusters as f64).sum::<f64>() / n;
        let avg_cluster_pts: f64 = results.iter().map(|r| r.cluster_pts as f64).sum::<f64>() / n;
        let avg_noise: f64 = results.iter().map(|r| r.noise_pts as f64).sum::<f64>() / n;
        let avg_wall: f64 = results.iter().map(|r| r.wall_ms).sum::<f64>() / n;
        let avg_cl: f64 = results.iter().map(|r| r.cluster_ms).sum::<f64>() / n;
        let avg_total: f64 = results.iter().map(|r| r.total_ms).sum::<f64>() / n;

        println!("{:<20} {:>7.0} {:>7.1} {:>7.0} {:>7.0} {:>8.2} {:>8.2} {:>8.2}",
            lbl, avg_wall, avg_clusters, avg_cluster_pts, avg_noise,
            avg_wall, avg_cl, avg_total);

        let parts: Vec<&str> = lbl.split('_').collect();
        let d: f32 = parts[0].trim_start_matches('d').parse().unwrap_or(0.0);
        let g: f32 = parts[1].trim_start_matches('g').parse().unwrap_or(0.0);

        aggregated.push(AggregatedResult {
            combo: lbl.clone(),
            distance: d,
            grad_threshold: g,
            avg_wall_pts: avg_wall,
            avg_clusters,
            avg_cluster_pts,
            avg_noise,
            avg_wall_ms: avg_wall,
            avg_cluster_ms: avg_cl,
            avg_total_ms: avg_total,
            frames: results.clone(),
        });
    }

    let json_path = format!("{}/results.json", out_base);
    std::fs::write(&json_path, serde_json::to_string_pretty(&aggregated)?)?;
    println!();
    println!("分析数据: {}", json_path);
    println!("检测框:   {}/{{label}}/{{label}}.db", out_base);

    Ok(())
}
