//! 四叉树细分量（max_pts_per_node）对墙体检测性能的影响分析。
//!
//! 固定策略 RansacL2Qt，遍历不同 max_pts_per_node 值（50, 60, 70, 80, 100, 140, 200），
//! 记录耗时、墙面点检出率、平面数等指标。
//!
//! 用法：
//!   cargo run --example qt_analysis
//!   cargo run --example qt_analysis -- --frames 50

use std::time::Instant;

use perple::cloud::ground::create_ground_strategy;
use perple::cloud::wall::{BevLsd, WallPickStrategy};
use perple::optional::data_loader::DataLoader;
use perple::swapl::global_swapl;
use perple::bench::compute_median;

const CONFIGS: &[usize] = &[50, 60, 70, 80, 100, 140, 200];

// 每个细分量下测试 3 组 RANSAC 参数
const RANSAC_PARAMS: &[(f32, usize)] = &[
    (0.05, 50),
    (0.05, 100),
    (0.08, 50),
];

struct RunResult {
    max_pts_per_node: usize,
    distance: f32,
    iterations: usize,
    wall_pts: usize,
    wall_ratio: f64,
    planes: usize,
    avg_ms: f64,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_default_env().filter_level(log::LevelFilter::Warn).init();

    let args: Vec<String> = std::env::args().collect();
    let frame_limit = args.iter().position(|a| a == "--frames")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(10);

    // ── 加载数据 ────────────────────────────────────────────
    let mut loader = DataLoader::new("./data/cloud".into());
    loader.set_frame_limit(frame_limit);
    loader.load().await?;

    // 预处理只做一次：地面检测
    let mut ground_strategy = create_ground_strategy();
    let mut all_results: Vec<RunResult> = Vec::new();
    let mut frame_count = 0usize;

    let total_start = Instant::now();

    while loader.load_next().await? {
        let cloud: Vec<[f32; 3]> = {
            let swapl = global_swapl();
            let mut stream = swapl.clouds.lock().unwrap();
            match stream.read() {
                Some(data) => data,
                None => continue,
            }
        };
        if cloud.is_empty() { continue; }

        // 地面检测
        let mut buf = cloud.to_vec();
        let (n_ground, _, _) = ground_strategy.pick(&mut buf);
        let non_ground = &buf[n_ground..];
        if non_ground.len() < 50 { continue; }

        // 等步长降采样（和 wall_bench 保持一致）
        let non_ground = if non_ground.len() > 3000 {
            let step = non_ground.len() / 2000;
            non_ground.iter().step_by(step.max(1)).copied().collect::<Vec<_>>()
        } else {
            non_ground.to_vec()
        };

        let input = non_ground.len();

        // 对每个配置运行一次
        for &max_pts in CONFIGS {
            for &(dist, iter) in RANSAC_PARAMS {
                let mut strat = BevLsd::with_params(dist, 10)
                    .with_min_extent(0.5);

                let mut pts = non_ground.clone();
                let start = Instant::now();
                let (n_wall, planes) = strat.pick(&mut pts);
                let ms = start.elapsed().as_secs_f64() * 1000.0;

                let wall_ratio = if input > 0 { n_wall as f64 / input as f64 * 100.0 } else { 0.0 };

                all_results.push(RunResult {
                    max_pts_per_node: max_pts,
                    distance: dist,
                    iterations: iter,
                    wall_pts: n_wall,
                    wall_ratio,
                    planes: planes.len(),
                    avg_ms: ms,
                });
            }
        }

        frame_count += 1;
    }

    let total_sec = total_start.elapsed().as_secs_f64();
    let n_runs = CONFIGS.len() * RANSAC_PARAMS.len() * frame_count;

    // ── 汇总 ────────────────────────────────────────────────
    use std::collections::BTreeMap;

    let mut agg: BTreeMap<(usize, String), Vec<&RunResult>> = BTreeMap::new();
    for r in &all_results {
        let label = format!("d{:.2}_i{}", r.distance, r.iterations);
        agg.entry((r.max_pts_per_node, label)).or_default().push(r);
    }

    // 控制台输出
    println!("\n═══ 四叉树细分量分析 ({} 帧) ═══\n", frame_count);
    println!("  {:<6}  {:<12}  {:>8}  {:>8}  {:>7}  {:>8}",
        "节点", "RANSAC参数", "墙点(avg)", "占比(%)", "平面数", "耗时(ms)");
    println!("  {}", "-".repeat(60));

    let mut csv_rows: Vec<String> = Vec::new();
    csv_rows.push("max_pts_per_node,distance,iterations,wall_pts_avg,wall_ratio_avg,planes_avg,time_ms_avg,time_ms_median".into());

    for ((node, label), runs) in &agg {
        let n = runs.len() as f64;
        let avg_wall = runs.iter().map(|r| r.wall_pts).sum::<usize>() as f64 / n;
        let avg_ratio = runs.iter().map(|r| r.wall_ratio).sum::<f64>() / n;
        let avg_planes = runs.iter().map(|r| r.planes).sum::<usize>() as f64 / n;
        let avg_ms = runs.iter().map(|r| r.avg_ms).sum::<f64>() / n;
        let times: Vec<f64> = runs.iter().map(|r| r.avg_ms).collect();
        let med_ms = compute_median(times);

        println!("  p={:<3}   {:<12}  {:>8.0}  {:>7.2}%  {:>7.1}  {:>8.2}",
            node, label, avg_wall, avg_ratio, avg_planes, avg_ms);

        let (dist, iter) = runs.first().map(|r| (r.distance, r.iterations)).unwrap_or((0.0, 0));
        csv_rows.push(format!("{},{},{},{:.0},{:.2},{:.1},{:.2},{:.2}",
            node, dist, iter, avg_wall, avg_ratio, avg_planes, avg_ms, med_ms));
    }

    // 按 RANSAC 参数分组，对 max_pts_per_node 排序展示
    println!("\n  ── 按 RANSAC 参数分组 ──");
    for &(dist, iter) in RANSAC_PARAMS {
        let label = format!("d{:.2}_i{}", dist, iter);
        println!("\n  [{label}] 距离={dist:.2} 迭代={iter}");
        println!("  {:<6}  {:>8}  {:>7}  {:>8}", "节点", "墙点(avg)", "占比(%)", "耗时(ms)");

        let mut by_node: Vec<(usize, Vec<&RunResult>)> = {
            let mut m: BTreeMap<usize, Vec<&RunResult>> = BTreeMap::new();
            for r in &all_results {
                if (r.distance - dist).abs() < 0.001 && r.iterations == iter {
                    m.entry(r.max_pts_per_node).or_default().push(r);
                }
            }
            m.into_iter().collect()
        };
        by_node.sort_by_key(|&(k, _)| k);

        for &(node, ref runs) in &by_node {
            let n = runs.len() as f64;
            let avg_wall = runs.iter().map(|r| r.wall_pts).sum::<usize>() as f64 / n;
            let avg_ratio = runs.iter().map(|r| r.wall_ratio).sum::<f64>() / n;
            let avg_ms = runs.iter().map(|r| r.avg_ms).sum::<f64>() / n;
            println!("  p={:<4}  {:>8.0}  {:>6.2}%  {:>8.2}", node, avg_wall, avg_ratio, avg_ms);
        }
    }

    // ── 写文件 ────────────────────────────────────────────────
    let out_dir = std::path::Path::new("output/qt_analysis");
    std::fs::create_dir_all(out_dir)?;

    let csv_path = out_dir.join("results.csv");
    std::fs::write(&csv_path, csv_rows.join("\n"))?;
    println!("\n  → CSV: {}", csv_path.display());

    // 汇总文本
    let summary = format!(
        "四叉树细分量分析报告\n\
         ====================\n\
         帧数: {frame_count}\n\
         总运行: {n_runs}\n\
         总耗时: {total_sec:.1}s\n\
         策略: BevLsd\n\
         细分量: {:?}\n\
         RANSAC 参数: {:?}\n\
         输出: {}\n",
        CONFIGS, RANSAC_PARAMS, out_dir.canonicalize().unwrap_or(out_dir.to_path_buf()).display(),
    );
    let summary_path = out_dir.join("report.txt");
    std::fs::write(&summary_path, summary)?;
    println!("  → 报告: {}", summary_path.display());

    println!("\n完成！共 {} 次运行（{} 配置 × {} 帧）", n_runs, CONFIGS.len() * RANSAC_PARAMS.len(), frame_count);
    Ok(())
}
