use std::time::Instant;

use perple::optional::data_loader::DataLoader;
use perple::swapl::global_swapl;
use perple::utils::boxes::Box3D;
use perple::cloud::ground::{GroundPickStrategy, HistogramExpandStrategy, create_ground_strategy};
use perple::cloud::classify::strategy::{DbscanStrategy, RangeImageStrategy};

use redra_client::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();

    println!("=== 室内聚类策略对比测试（全量帧） ===");

    // ── 加载数据（不限帧数） ──
    let mut data_loader = DataLoader::new("./data/test".to_string());
    data_loader.load().await?;

    let mut results: Vec<BenchResult> = Vec::new();
    let mut frame_idx = 0usize;
    let total_start = Instant::now();

    // 保存最后一帧的非地面点（用于可视化）
    let mut last_non_ground: Vec<[f32; 3]> = Vec::new();

    while data_loader.load_next().await? {
        // 读取当前帧点云
        let mut cloud: Vec<[f32; 3]> = {
            let swapl = global_swapl();
            let mut stream = swapl.clouds.lock().await;
            match stream.read() {
                Some(data) => data,
                None => continue,
            }
        };

        if cloud.is_empty() {
            frame_idx += 1;
            continue;
        }

        // ── 提走地面 ──
        let mut ground_strategy = create_ground_strategy();
        let (n_ground, _, _) = ground_strategy.pick(&mut cloud);
        let non_ground = &cloud[n_ground..];

        if frame_idx == 0 {
            println!("点云总数: {}, 地面点: {}, 非地面点: {}\n",
                cloud.len(), n_ground, non_ground.len());
        }

        // 保存最后一帧的非地面点
        if frame_idx >= 0 {
            last_non_ground = non_ground.to_vec();
        }

        // ── 策略 1: 当前默认参数 ──
        {
            let points = non_ground.to_vec();
            let start = Instant::now();
            let mut strat = DbscanStrategy::with_params(0.20, 0.0, 10, 50, 10, 0.10);
            let (processed, objects) = strat.run(&points);
            let (clusters, noise) = to_bench_results(&processed, &objects);
            let elapsed = start.elapsed();
            let n_humans = count_human_like(&clusters);
            accumulate_or_push(&mut results, "默认 eps0.20", clusters, noise, n_humans, elapsed);
        }

        // ── 策略 2: 固定 eps DBSCAN ──
        for &voxel in &[0.05f32, 0.10, 0.20] {
            for &eps in &[0.15f32, 0.25, 0.35, 0.50, 0.80] {
                for &min_pts in &[3usize, 5, 8, 15] {
                    let points = non_ground.to_vec();
                    let start = Instant::now();
                    let mut strat = DbscanStrategy::with_params(eps, 0.0, min_pts, 50, 10, voxel);
                    let (processed, objects) = strat.run(&points);
                    let (clusters, noise) = to_bench_results(&processed, &objects);
                    let elapsed = start.elapsed();
                    let n_humans = count_human_like(&clusters);
                    let label = format!("eps{:.2}_m{}_v{:.2}", eps, min_pts, voxel);
                    accumulate_or_push(&mut results, &label, clusters, noise, n_humans, elapsed);
                }
            }
        }

        // ── 策略 3: 自适应 eps DBSCAN ──
        for &voxel in &[0.05f32, 0.10, 0.20] {
            for &eps_0 in &[0.05f32, 0.10, 0.15] {
                for &slope in &[0.02f32, 0.05, 0.10] {
                    for &min_pts in &[3usize, 5, 8, 15] {
                        let points = non_ground.to_vec();
                        let start = Instant::now();
                        let mut strat = DbscanStrategy::with_params(eps_0, slope, min_pts, 50, 10, voxel);
                        let (processed, objects) = strat.run(&points);
                        let (clusters, noise) = to_bench_results(&processed, &objects);
                        let elapsed = start.elapsed();
                        let n_humans = count_human_like(&clusters);
                        let label = format!("adapt_e{:.2}_s{:.2}_m{}_v{:.2}", eps_0, slope, min_pts, voxel);
                        accumulate_or_push(&mut results, &label, clusters, noise, n_humans, elapsed);
                    }
                }
            }
        }

        // ── 策略 4: 无下采样 ──
        for &(eps_0, slope, min_pts, label) in &[
            (0.35f32, 0.0f32, 5usize, "无体素_eps0.35_m5"),
            (0.05, 0.05, 5, "无体素_adapt_e0.05_s0.05_m5"),
            (0.10, 0.05, 3, "无体素_adapt_e0.10_s0.05_m3"),
        ] {
            let points = non_ground.to_vec();
            let start = Instant::now();
            let mut strat = DbscanStrategy::with_params(eps_0, slope, min_pts, 50, 10, 0.0);
            let (processed, objects) = strat.run(&points);
            let (clusters, noise) = to_bench_results(&processed, &objects);
            let elapsed = start.elapsed();
            let n_humans = count_human_like(&clusters);
            accumulate_or_push(&mut results, label, clusters, noise, n_humans, elapsed);
        }

        // ── 策略 5: Range Image ──
        for &(az, el, thresh, min_pts, label) in &[
            (0.5f32, 0.5f32, 0.5f32, 3usize, "ri_0.5deg_t0.5_m3"),
            (1.0, 1.0, 0.5, 3, "ri_1.0deg_t0.5_m3"),
            (1.0, 1.0, 1.0, 3, "ri_1.0deg_t1.0_m3"),
            (2.0, 2.0, 1.0, 3, "ri_2.0deg_t1.0_m3"),
            (0.5, 0.5, 0.3, 5, "ri_0.5deg_t0.3_m5"),
        ] {
            let points = non_ground.to_vec();
            let start = Instant::now();
            let mut strat = RangeImageStrategy::with_params(az, el, thresh, min_pts);
            let (processed, objects) = strat.run(&points);
            let (clusters, noise) = to_bench_results(&processed, &objects);
            let elapsed = start.elapsed();
            let n_humans = count_human_like(&clusters);
            accumulate_or_push(&mut results, label, clusters, noise, n_humans, elapsed);
        }

        frame_idx += 1;
        if frame_idx % 10 == 0 {
            println!("已处理 {} 帧...", frame_idx);
        }
    }

    let total_elapsed = total_start.elapsed();

    // ── 汇总表（按大目标均值降序） ──
    println!("\n全量 {} 帧，总耗时: {:.1}s\n", frame_idx, total_elapsed.as_secs_f64());
    println!("{:-<110}", "");
    println!("| {:<32} | {:>6} | {:>5} | {:>5} | {:>7} | {:>7} | {:>5} |",
        "策略", "帧均簇", "帧均噪", "帧均人", "帧均ms", "人占比", "帧数");
    println!("{:-<110}", "");

    results.sort_by(|a, b| {
        let a_avg = if a.frame_count > 0 { a.total_humans as f64 / a.frame_count as f64 } else { 0.0 };
        let b_avg = if b.frame_count > 0 { b.total_humans as f64 / b.frame_count as f64 } else { 0.0 };
        b_avg.partial_cmp(&a_avg).unwrap_or(std::cmp::Ordering::Equal)
    });

    for r in &results {
        let n = r.frame_count.max(1) as f64;
        let avg_clusters = r.total_clusters as f64 / n;
        let avg_noise = r.total_noise as f64 / n;
        let avg_humans = r.total_humans as f64 / n;
        let avg_ms = r.total_ms / n;
        let human_ratio = if r.total_clusters > 0 {
            r.total_humans as f64 / r.total_clusters as f64 * 100.0
        } else {
            0.0
        };
        println!("| {:<32} | {:>6.1} | {:>5.0} | {:>5.1} | {:>7.1} | {:>6.0}% | {:>5} |",
            r.label, avg_clusters, avg_noise, avg_humans, avg_ms, human_ratio, r.frame_count);
    }
    println!("{:-<110}", "");

    // ── 每个策略写入独立 .rdra 文件（用最后一帧数据可视化） ──
    let output_dir = "output/cluster_bench";
    std::fs::create_dir_all(output_dir)?;

    let point_step = (last_non_ground.len() / 5000).max(1);
    let colors = ["red", "green", "blue", "yellow", "magenta", "cyan", "orange", "purple"];

    for (idx, r) in results.iter().enumerate() {
        if let Some(ref clusters) = r.last_clusters {
            if clusters.is_empty() {
                continue;
            }

            let mut writer = RdraWriter::new();

            // 非地面点云（白色，稀疏采样）
            for (si, i) in (0..last_non_ground.len()).step_by(point_step).enumerate() {
                let p = last_non_ground[i];
                writer.spawn(
                    spawn_sphere(p, 0.03, "white")
                        .id(1_000_000 + si as u64 * 4)
                );
            }

            // 各簇的包围盒（颜色循环）
            for (ci, cluster) in clusters.iter().enumerate() {
                if cluster.is_empty() { continue; }
                let mut box3d = Box3D::empty_box();
                box3d.cloud2box(cluster);
                let verts: Vec<(f32, f32, f32)> = box3d.vertices().iter()
                    .map(|v| (v.x, v.y, v.z))
                    .collect();
                let color = colors[ci % colors.len()];
                writer.spawn(
                    spawn_cube(verts, color)
                        .id(200_000 + (idx as u64) * 1000 + ci as u64)
                        .tag(format!("{}pts h={:.1}m", cluster.len(), box3d.height))
                );
            }

            // 策略标签
            {
                let n = r.frame_count.max(1) as f64;
                let tag = format!("{} | {:.1}簇 {:.1}人 {:.0}ms/帧 | {}帧",
                    r.label,
                    r.total_clusters as f64 / n,
                    r.total_humans as f64 / n,
                    r.total_ms / n,
                    r.frame_count);
                let dummy: Vec<(f32, f32, f32)> = vec![
                    (-0.1, -0.1, -0.1), ( 0.1, -0.1, -0.1), ( 0.1,  0.1, -0.1), (-0.1,  0.1, -0.1),
                    (-0.1, -0.1,  0.1), ( 0.1, -0.1,  0.1), ( 0.1,  0.1,  0.1), (-0.1,  0.1,  0.1),
                ];
                writer.spawn(
                    spawn_cube(dummy, "glass")
                        .id(900_000 + idx as u64)
                        .tag(tag)
                );
            }

            writer.end_frame();

            let safe_label = r.label.replace(['=', '.', ' ', ':'], "_");
            let path = format!("{}/{}.rdra", output_dir, safe_label);
            writer.save(&path)?;
            println!("  [{}] {} → {}", idx + 1, r.label, path);
        }
    }

    println!("\n共 {} 个 .rdra 文件保存到 {}", results.len(), output_dir);
    Ok(())
}

struct BenchResult {
    label: String,
    total_clusters: usize,
    total_noise: usize,
    total_humans: usize,
    total_ms: f64,
    frame_count: usize,
    /// 最后一帧的簇数据（用于可视化）
    last_clusters: Option<Vec<Vec<[f32; 3]>>>,
}

/// 累加到已有结果，或首次插入
fn accumulate_or_push(
    results: &mut Vec<BenchResult>,
    label: &str,
    clusters: Vec<Vec<[f32; 3]>>,
    noise: usize,
    n_humans: usize,
    elapsed: std::time::Duration,
) {
    let ms = elapsed.as_secs_f64() * 1000.0;
    match results.iter_mut().find(|r| r.label == label) {
        Some(r) => {
            r.total_clusters += clusters.len();
            r.total_noise += noise;
            r.total_humans += n_humans;
            r.total_ms += ms;
            r.frame_count += 1;
            r.last_clusters = Some(clusters);
        }
        None => {
            results.push(BenchResult {
                label: label.to_string(),
                total_clusters: clusters.len(),
                total_noise: noise,
                total_humans: n_humans,
                total_ms: ms,
                frame_count: 1,
                last_clusters: Some(clusters),
            });
        }
    }
}

fn to_bench_results(points: &[[f32; 3]], objects: &[Vec<usize>]) -> (Vec<Vec<[f32; 3]>>, usize) {
    let total: usize = objects.iter().map(|c| c.len()).sum();
    let noise = points.len() - total;
    let clusters: Vec<Vec<[f32; 3]>> = objects.iter()
        .map(|c| c.iter().map(|&i| points[i]).collect())
        .collect();
    (clusters, noise)
}

fn count_human_like(clusters: &[Vec<[f32; 3]>]) -> usize {
    let mut count = 0;
    for cluster in clusters {
        if cluster.len() < 3 { continue; }
        let mut box3d = Box3D::empty_box();
        box3d.cloud2box(&cluster);
        let w = box3d.length.max(box3d.width);
        let h = box3d.height;
        if w < 0.25 || h < 0.5 { continue; }
        if h > w * 0.5 && w < 1.2 && h > 1.0 && h < 2.5 {
            count += 1;
        }
    }
    count
}
