//! 管线演化对比 Benchmark — 同一聚类算法在不同预处理管线下的表现。
//!
//! 展示技术路线变迁（4 个时代）：
//!   Era 1a (RawFull)   : 原始全量点云 → DBSCAN — 全量无降采样，标注"不可行"
//!   Era 1b (RawDown)   : 原始点云 + 体素降采样 → DBSCAN — 朴素但噪声大
//!   Era 2  (GroundDown): 去地面 + 体素降采样 → DBSCAN — 提速有限，残留墙体干扰
//!   Era 3  (WallClean) : 去地面 + 去墙体 → DBSCAN — 三层管线，又快又稳
//!
//! 用法：
//!   cargo run --example pipeline_evolution_bench
//!   cargo run --example pipeline_evolution_bench -- --frames 20

use std::time::{Duration, Instant};

use perple::bench::{
    BenchStrategy, BenchStats, BenchHarness, BenchRecorder, FrameData, WallPreprocessor,
};
use perple::cloud::classify::strategy::DbscanStrategy;

/// 输入数据源 — 决定聚类拿到的点集。
#[derive(Clone, Copy, PartialEq)]
enum InputSource {
    /// frame.cloud — 原始点云，无任何预处理
    Raw,
    /// frame.non_ground() — 仅去地面
    NonGround,
    /// frame.non_wall() — 去地面+墙体
    NonWall,
}

struct EvolutionCase {
    name: &'static str,
    era_label: &'static str,
    input_source: InputSource,
    /// DbscanStrategy voxel_size（内部降采样）。全量用 0.1 降采样，三层管线用 0.0 跳过
    voxel_size: f32,
    // 累计统计
    total_ms: f64,
    frame_count: usize,
    acc_input: usize,
    acc_clusters: usize,
    acc_noise: usize,
    frame_times: Vec<f64>,
}

impl EvolutionCase {
    fn new(name: &'static str, era_label: &'static str, input_source: InputSource, voxel_size: f32) -> Self {
        Self {
            name, era_label, input_source, voxel_size,
            total_ms: 0.0, frame_count: 0,
            acc_input: 0, acc_clusters: 0, acc_noise: 0,
            frame_times: Vec::new(),
        }
    }
}

impl BenchStrategy for EvolutionCase {
    fn name(&self) -> &str { self.name }

    fn run(&mut self, frame: &FrameData) -> Duration {
        let input = match self.input_source {
            InputSource::Raw => frame.cloud.to_vec(),
            InputSource::NonGround => frame.non_ground().to_vec(),
            InputSource::NonWall => frame.non_wall().to_vec(),
        };

        // 统一 DBSCAN：固定 eps=0.20, min_pts=5, Sloc=0 (禁用自适应 eps)
        // voxel_size 按 Era 区分：全量管线用降采样控制点数，三层管线无需额外降采样
        let mut dbscan = DbscanStrategy::with_params(0.20, 0.0, 5, 50, 10, self.voxel_size);

        let start = Instant::now();
        let (sampled, objects) = dbscan.run(&input);
        let elapsed = start.elapsed();

        let total_cluster_pts: usize = objects.iter().map(|c| c.len()).sum();
        let noise = sampled.len().saturating_sub(total_cluster_pts);

        let ms = elapsed.as_secs_f64() * 1000.0;
        self.total_ms += ms;
        self.frame_count += 1;
        self.frame_times.push(ms);
        self.acc_input += input.len();
        self.acc_clusters += objects.len();
        self.acc_noise += noise;

        // 每帧进度（仅 Era 最慢那个全量打日志，减少输出）
        if self.name == "Era1a_RawFull" || (self.name == "Era3_WallClean" && self.frame_count % 10 == 0) {
            println!("  [{:15}] 帧{:2} 入{} 簇{} 噪{} | {:>6.1}ms",
                self.name, self.frame_count, input.len(), objects.len(), noise, ms);
        }

        elapsed
    }

    fn write_frame(&mut self, recorder: &mut BenchRecorder, frame: &FrameData) {
        recorder.begin_frame(frame.frame_idx);
        recorder.end_frame();
    }

    fn summarize(&self) {
        let n = self.frame_count.max(1) as f64;
        let avg_ms = self.total_ms / n;
        let tag = if avg_ms > 5000.0 { " [不可行]" } else if avg_ms > 1000.0 { " [偏慢]" } else { "" };
        println!(
            "  {:16} {:6} | {:>6.0} pts | {:>4.1} cls | {:>5.0} noise | {:>7.1}ms{}",
            self.name, self.era_label,
            self.acc_input as f64 / n,
            self.acc_clusters as f64 / n,
            self.acc_noise as f64 / n,
            avg_ms, tag,
        );
    }

    fn stats(&self) -> BenchStats {
        BenchStats {
            name: self.name.to_string(),
            frame_count: self.frame_count,
            total_ms: self.total_ms,
            frame_times: self.frame_times.clone(),
        }
    }

    fn extra_metrics(&self) -> Vec<(String, f64)> {
        let n = self.frame_count.max(1) as f64;
        vec![
            ("avg_input".into(), self.acc_input as f64 / n),
            ("avg_clusters".into(), self.acc_clusters as f64 / n),
            ("avg_noise".into(), self.acc_noise as f64 / n),
        ]
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Warn)
        .init();

    let args: Vec<String> = std::env::args().collect();
    let frame_limit: usize = args
        .iter()
        .position(|a| a == "--frames")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);

    println!("═══ 管线演化对比 ({} 帧, 统一 DBSCAN eps=0.20 min=5) ═══\n", frame_limit);
    println!("时代分区:");
    println!("  Era1a [RawFull]   : 原始点云 → DBSCAN (无降采样, 无预处理)");
    println!("  Era1b [RawDown]   : 原始点云+体素0.10 → DBSCAN (朴素降采样)");
    println!("  Era2  [GroundDown]: 去地面+体素0.10 → DBSCAN (旧管线)");
    println!("  Era3  [WallClean] : 去地面+去墙体 → DBSCAN (三层管线, 无额外降采样)");
    println!();

    let mut strategies: Vec<Box<dyn BenchStrategy>> = vec![
        Box::new(EvolutionCase::new("Era1a_RawFull",   "[不可行]", InputSource::Raw,       0.0)),
        Box::new(EvolutionCase::new("Era1b_RawDown",   "[朴素]",   InputSource::Raw,       0.10)),
        Box::new(EvolutionCase::new("Era2_GroundDown",  "[旧管线]",  InputSource::NonGround, 0.10)),
        Box::new(EvolutionCase::new("Era3_WallClean",   "[三层]",   InputSource::NonWall,   0.0)),
    ];

    let tmp = std::env::temp_dir().join("evolution_bench");
    let mut recorders: Vec<BenchRecorder> = (0..strategies.len())
        .map(|i| BenchRecorder::new(tmp.join(format!("{}.db", i))).expect("创建 recorder 失败"))
        .collect();
    let harness = BenchHarness::new("./data/cloud", frame_limit);
    let mut preprocessor = WallPreprocessor::default();
    harness.run(&mut preprocessor, &mut strategies, &mut recorders).await?;

    // 按速度排序输出
    strategies.sort_by(|a, b| {
        let at = a.stats().total_ms / a.stats().frame_count.max(1) as f64;
        let bt = b.stats().total_ms / b.stats().frame_count.max(1) as f64;
        at.partial_cmp(&bt).unwrap_or(std::cmp::Ordering::Equal)
    });

    println!("\n═══ 按速度升序 ═══");
    println!("{:-<85}", "");
    println!("  {:16} {:6} | {:>6} | {:>4} | {:>5} | {:>7}",
        "管线", "时代", "pts/帧", "簇", "噪声", "ms/帧");
    println!("{:-<85}", "");
    for s in &strategies {
        s.summarize();
    }
    println!("{:-<85}", "");

    // 计算各 Era 的加速贡献
    let get_ms = |name: &str| -> f64 {
        strategies.iter().find(|s| s.name() == name)
            .map(|s| { let st = s.stats(); if st.frame_count > 0 { st.total_ms / st.frame_count as f64 } else { 0.0 } })
            .unwrap_or(0.0)
    };
    let era1a = get_ms("Era1a_RawFull");
    let era1b = get_ms("Era1b_RawDown");
    let era2  = get_ms("Era2_GroundDown");
    let era3  = get_ms("Era3_WallClean");
    if era1a > 0.0 && era1b > 0.0 {
        println!("  Era1b vs Era1a: {:.0}ms vs {:.0}ms → {:.1}x (降采样的作用)", era1b, era1a, era1a / era1b);
    }
    if era2 > 0.0 {
        println!("  Era2  vs Era1b: {:.0}ms vs {:.0}ms → {:.1}x (去地面的作用)", era2, era1b, era1b / era2);
    }
    if era3 > 0.0 {
        println!("  Era3  vs Era2 : {:.0}ms vs {:.0}ms → {:.1}x (去墙体的作用)", era3, era2, era2 / era3);
        println!("  Era3  vs Era1a: {:.0}ms vs {:.0}ms → {:.1}x (三层管线总加速比)", era3, era1a, era1a / era3);
    }

    Ok(())
}
