//! 管线评估脚本 — 按文件名编号统计
//!
//! 用法：
//!   cargo run --example eval_pipeline
//!   cargo run --example eval_pipeline -- --start 200 --end 906
//!   cargo run --example eval_pipeline -- --start 200 --end 906 --fold-size 100

use std::time::Instant;

use perple::cloud::core::Lidar;
use perple::color::core::Camera;
use perple::fuse::Fuse;
use perple::optional::data_loader::DataLoader;
use perple::swapl::global_swapl;
use perple::tracker::core::Tracker;
use perple::tracker::output::Target;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Warn)
        .init();

    let args: Vec<String> = std::env::args().collect();
    let start_file: u32 = args.iter()
        .position(|a| a == "--start").and_then(|i| args.get(i+1))
        .and_then(|s| s.parse().ok()).unwrap_or(101);
    let end_file: u32 = args.iter()
        .position(|a| a == "--end").and_then(|i| args.get(i+1))
        .and_then(|s| s.parse().ok()).unwrap_or(953);
    let fold_size: usize = args.iter()
        .position(|a| a == "--fold-size").and_then(|i| args.get(i+1))
        .and_then(|s| s.parse().ok()).unwrap_or(0);

    // ─── 读取实际文件列表构建帧编号映射 ──────────────────────────────
    let lidar_dir = std::fs::read_dir("./data/cloud/lidar")
        .map_err(|e| format!("读取 lidar 目录失败: {}", e))?;
    let mut file_nums: Vec<u32> = lidar_dir
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            let name = e.file_name();
            let name = name.to_str()?;
            name.trim_end_matches(".pcd").parse::<u32>().ok()
        })
        .collect();
    file_nums.sort();

    // ─── 加载并确定帧范围 ────────────────────────────────────────────────
    let mut data_loader = DataLoader::new("./data/cloud".to_string());
    data_loader.load().await?;
    let n_total = data_loader.frame_count().min(file_nums.len());
    eprintln!("总帧数: {}, 文件范围: {}-{}", n_total, file_nums[0], file_nums[n_total-1]);

    // ─── 初始化管线 ──────────────────────────────────────────────────────
    let mut lidar = Lidar::new();
    let mut camera = Camera::new();
    let mut fuse = Fuse::new();
    let mut tracker = Tracker::new();

    if !data_loader.load_next().await? { return Ok(()); }
    if n_total > 1 { data_loader.load_next().await?; }

    let mut l_handle = Some(tokio::spawn(async move { let _ = lidar.act().await; lidar }));
    let mut c_handle = Some(tokio::spawn(async move { let _ = camera.act().await; camera }));

    let mut stats: Vec<(u32, usize, usize, usize, usize)> = Vec::new();
    let total_start = Instant::now();

    for i in 0..n_total {
        let (l_res, c_res) = tokio::join!(l_handle.take().unwrap(), c_handle.take().unwrap());
        lidar = l_res.unwrap();
        camera = c_res.unwrap();

        // DualBuf swap: 检测阶段 → 后融合阶段
        let swapl = global_swapl();
        swapl.swap_pipeline();

        // 提前启动下一帧检测（与后融合并行）
        if i + 1 < n_total {
            l_handle = Some(tokio::spawn(async move { let _ = lidar.act().await; lidar }));
            c_handle = Some(tokio::spawn(async move { let _ = camera.act().await; camera }));
        }

        // 后融合（与下一帧检测并行）
        fuse.act().await;
        let _ = tracker.run().await;

        let file_num = file_nums.get(i).copied().unwrap_or(0);
        let clusters = swapl.cld_buds_raw.consumer().lock().unwrap().len();
        let targets: Vec<Target> = swapl.targets.lock().unwrap().read().unwrap_or_default();
        let n_moving = targets.iter().filter(|t| t.classification == "moving").count();
        let n_person = targets.iter().filter(|t| t.class_type == "person").count();
        let n_obstacle = targets.iter().filter(|t| t.class_type != "person").count();

        if file_num >= start_file && file_num <= end_file {
            stats.push((file_num, clusters, n_moving, n_person, n_obstacle));
        }

        if i % 100 == 0 {
            println!("  进度: {}/{} 帧", i, n_total);
        }

        if i + 2 < n_total { data_loader.load_next().await?; }
    }

    let elapsed = total_start.elapsed().as_secs_f64();

    // ─── 输出分段统计 ──────────────────────────────────────────────────
    println!("\n=== 评估结果: {} 帧 ({:.1}s) ===", stats.len(), elapsed);

    if fold_size > 0 {
        let n_folds = stats.len() / fold_size;
        let total_used = n_folds * fold_size;
        println!("  拆分为 {} 折 × {} 帧（丢弃 {} 帧）\n", n_folds, fold_size, stats.len() - total_used);
        println!("  {:>6}  {:>6}  {:>5}  {:>6}  {:>6}  {:>6}  {:>5}", "Fold", "文件范围", "帧数", "簇均值", "行人均", "障碍均", "有人帧");

        for f in 0..n_folds {
            let beg = f * fold_size;
            let end = beg + fold_size;
            let seg = &stats[beg..end];

            let first_file = seg.first().map(|s| s.0).unwrap_or(0);
            let last_file = seg.last().map(|s| s.0).unwrap_or(0);
            let avg_c = seg.iter().map(|s| s.1).sum::<usize>() as f64 / fold_size as f64;
            let avg_p = seg.iter().map(|s| s.3).sum::<usize>() as f64 / fold_size as f64;
            let avg_o = seg.iter().map(|s| s.4).sum::<usize>() as f64 / fold_size as f64;
            let with_p = seg.iter().filter(|s| s.3 > 0).count();

            println!("  {:>3}     {:>3}-{:>3}  {:>5}  {:>6.1}  {:>6.1}  {:>6.1}  {:>3}/{} ({:>2}%)",
                f + 1, first_file, last_file, fold_size, avg_c, avg_p, avg_o,
                with_p, fold_size, with_p * 100 / fold_size);
        }

        // 总体（按折平均）
        let avg_recall = (0..n_folds).map(|f| {
            let beg = f * fold_size;
            let seg = &stats[beg..beg + fold_size];
            seg.iter().filter(|s| s.3 > 0).count() as f64 / fold_size as f64
        }).sum::<f64>() / n_folds as f64;

        println!("\n  >>> 平均帧级召回率: {:.1}%", avg_recall * 100.0);
    } else {
        let with_person = stats.iter().filter(|s| s.3 > 0).count();
        let total_p: usize = stats.iter().map(|s| s.3).sum();
        let total_o: usize = stats.iter().map(|s| s.4).sum();
        let max_concurrent_p = stats.iter().map(|s| s.3).max().unwrap_or(0);
        println!("  有行人帧 {}/{} ({:.1}%) | 累计行人 {} | 累计障碍 {} | 帧均 {:.2}/{:.2} | 最多 {} 人",
            with_person, stats.len(),
            with_person as f64 / stats.len() as f64 * 100.0,
            total_p, total_o,
            total_p as f64 / stats.len() as f64,
            total_o as f64 / stats.len() as f64,
            max_concurrent_p);
    }

    Ok(())
}
