//! 全流程策略对比 Benchmark
//!
//! 测试不同的墙体提取 + 聚类策略组合，输出性能和质量对比。
//! 排除过往 bench 中确认耗时的策略（voxel=0.05, range_image 0.5° 等）。
//!
//! 用法：
//!   cargo run --example pipeline_bench
//!   cargo run --example pipeline_bench -- --frames 50

use std::time::{Duration, Instant};

use perple::bench::{BenchStrategy, BenchStats, BenchHarness, BenchRecorder, FrameData, GroundPreprocessor};
use perple::cloud::wall::{
    WallPickStrategy, BevLsd, XYGrid,
};
use perple::cloud::classify::strategy::{ClusteringStrategy, DbscanStrategy};
use perple::config::fixif;
use perple::utils::boxes::Box3D;

/// 单帧统计快照
#[derive(Default, Clone)]
#[allow(dead_code)]
struct FrameStats {
    n_wall: usize,
    n_after_voxel: usize,
    n_after_range: usize,
    n_sampled: usize,
    n_clusters: usize,
}

/// 组合策略测试用例
struct PipelineBenchCase {
    name: String,
    wall: Option<usize>,          // 索引到墙体策略工厂，None = 无墙体提取
    cluster_idx: usize,            // 索引到聚类策略工厂
    voxel_min_occ: usize,
    // 累计统计
    total_ms: f64,
    frame_count: usize,
    acc_wall: usize,
    acc_total_input: usize,
    acc_after_voxel: usize,
    acc_after_range: usize,
    acc_clusters: usize,
    frame_times: Vec<f64>,
    last: FrameStats,
    last_boxes: Vec<Box3D>,        // 障碍物包围盒（供 write_frame 输出）
}

impl PipelineBenchCase {
    fn new(name: &str, wall: Option<usize>, cluster_idx: usize) -> Self {
        Self {
            name: name.to_string(),
            wall,
            cluster_idx,
            voxel_min_occ: 3,
            total_ms: 0.0,
            frame_count: 0,
            acc_wall: 0,
            acc_total_input: 0,
            acc_after_voxel: 0,
            acc_after_range: 0,
            acc_clusters: 0,
            frame_times: Vec::new(),
            last: FrameStats::default(),
            last_boxes: Vec::new(),
        }
    }

    /// 创建对应索引的墙体策略（每次 new 一个，避免 clone 问题）
    fn create_wall(&self) -> Option<Box<dyn WallPickStrategy>> {
        match self.wall {
            Some(0) => Some(Box::new(BevLsd::with_params(0.05, 20))),
            Some(1) => Some(Box::new(BevLsd::with_params(0.08, 20))),
            _ => None,
        }
    }

    /// 创建对应索引的聚类策略
    fn create_cluster(&self) -> Box<dyn ClusteringStrategy> {
        match self.cluster_idx {
            0 => Box::new(DbscanStrategy::with_params(0.10, 0.20, 10, 50, 10, 0.10)),
            1 => Box::new(DbscanStrategy::with_params(0.15, 0.0, 5, 50, 10, 0.10)),
            2 => Box::new(DbscanStrategy::with_params(0.30, 0.0, 5, 50, 10, 0.20)),
            _ => Box::new(DbscanStrategy::with_params(0.10, 0.20, 10, 50, 10, 0.10)),
        }
    }
}

impl BenchStrategy for PipelineBenchCase {
    fn name(&self) -> &str { &self.name }

    fn run(&mut self, frame: &FrameData) -> Duration {
        let start = Instant::now();
        let mut buf = frame.non_ground().to_vec();

        // 1. 墙体提取
        let n_wall_before = buf.len();
        let n_wall = if let Some(ref mut w) = self.create_wall() {
            let (n, _) = w.pick(&mut buf);
            n
        } else {
            0
        };
        self.acc_total_input += n_wall_before;

        // 2. LV-DOT 体素占用过滤
        let (after_voxel, _) = XYGrid::voxel_occupancy_filter(&buf[n_wall..], 0.10, self.voxel_min_occ);

        let n_after_voxel = after_voxel.len();

        // 3. 天花板 + 范围过滤
        let config = fixif();
        let mut cluster_input = after_voxel;
        if config.cluster.ceiling_filter && config.cluster.ceiling_height > 0.0 {
            cluster_input.retain(|p| p[2] <= config.cluster.ceiling_height);
        }
        if config.cluster.max_range > 0.0 {
            let max_d2 = config.cluster.max_range * config.cluster.max_range;
            cluster_input.retain(|p| p[0] * p[0] + p[1] * p[1] + p[2] * p[2] <= max_d2);
        }

        // 4. 聚类
        let mut cluster = self.create_cluster();
        let (sampled, objects) = cluster.run(&cluster_input);
        let n_clusters = objects.len();

        // 构建输出可视化包围盒（仅障碍物，不分类）
        let mut last_boxes = Vec::with_capacity(objects.len());
        for obj in &objects {
            if obj.len() < 3 { continue; }
            let pts: Vec<[f32; 3]> = obj.iter().map(|&i| sampled[i]).collect();
            let b = Box3D::from_cloud_aabb(&pts, 0.05);
            let w = b.length.max(b.width);
            let h = b.height;
            if w < 0.25 || h < 0.5 { continue; }
            last_boxes.push(b);
        }
        self.last_boxes = last_boxes;

        self.last = FrameStats {
            n_wall,
            n_after_voxel,
            n_after_range: cluster_input.len(),
            n_sampled: sampled.len(),
            n_clusters,
        };

        let elapsed = start.elapsed();
        let ms = elapsed.as_secs_f64() * 1000.0;
        self.total_ms += ms;
        self.frame_count += 1;
        self.frame_times.push(ms);
        self.acc_wall += n_wall;
        self.acc_after_voxel += n_after_voxel;
        self.acc_after_range += cluster_input.len();
        self.acc_clusters += n_clusters;

        elapsed
    }

    fn write_frame(&mut self, recorder: &mut BenchRecorder, frame: &FrameData) {
        recorder.begin_frame(frame.frame_idx);

        // 写入非地面点云背景（受 write_raw 开关控制，默认关闭）
        let ng = frame.non_ground();
        let bg_step = (ng.len() / 3000).max(1);
        for i in (0..ng.len()).step_by(bg_step) {
            recorder.write_raw_cloud(&[ng[i]], "point_cloud", 1);
        }

        // 写入聚类包围盒（全部以统一材质输出，不区分行人）
        let boxes: Vec<(Box3D, String)> = self.last_boxes.iter()
            .map(|b| (b.clone(), "obstacle".into()))
            .collect();
        if !boxes.is_empty() {
            recorder.write_boxes(&boxes, "obstacle");
        }

        recorder.end_frame();

        // 每帧进度显示
        if self.frame_count % 20 == 0 {
            let avg_ms = self.total_ms / self.frame_count as f64;
            println!("  [{}] 帧 {} | 墙={} 剩余={} 簇={} | {:.0}ms/帧",
                self.name, self.frame_count, self.last.n_wall,
                self.last.n_after_range, self.last.n_clusters, avg_ms);
        }
    }

    fn summarize(&self) {
        let n = self.frame_count.max(1) as f64;
        let avg_ms = self.total_ms / n;
        let avg_remain = self.acc_after_range as f64 / n;
        let avg_clusters = self.acc_clusters as f64 / n;
        let total_in = self.acc_total_input as f64;
        let wall_pct = if total_in > 0.0 {
            self.acc_wall as f64 / total_in * 100.0
        } else {
            0.0
        };
        let status = if avg_ms > 100.0 { " [OVER]" } else { "" };
        println!("  {:<36} | {:>5.0}% | {:>6.0} | {:>4.1} | {:>6.1}ms | {}{}",
            self.name, wall_pct, avg_remain, avg_clusters, avg_ms, n as usize, status);
    }

    fn stats(&self) -> BenchStats {
        BenchStats {
            name: self.name.clone(),
            frame_count: self.frame_count,
            total_ms: self.total_ms,
            frame_times: self.frame_times.clone(),
        }
    }
}

fn build_strategies() -> Vec<Box<dyn BenchStrategy>> {
    let wall_indices: Vec<Option<usize>> = vec![
        None,           // 0: no_wall（基准）
        Some(0),        // 1: TopDownCluster p50_m5_w0.3
        Some(1),        // 2: TopDownCluster p100_m3_w0.3
        Some(2),        // 3: XYRansacWall 0.05/50/30（当前默认）
        Some(3),        // 4: XYRansacWall 0.08/50/30
        Some(4),        // 5: GridWall 0.10/3/1.5
    ];
    let cluster_indices: Vec<usize> = vec![
        0,  // dbscan_adaptive eps=0.10 slope=0.20 min=10 voxel=0.10
        1,  // dbscan_fixed eps=0.15 min=5 voxel=0.10
        2,  // dbscan_fixed eps=0.30 min=5 voxel=0.20
    ];

    let wall_names: [&str; 5] = [
        "td_c0.05_d5_m2", "td_c0.10_d3_m2",
        "ransac_d0.05", "ransac_d0.08",
        "qt_c0.10_p3_z1.5",
    ];
    let cluster_names: [&str; 3] = [
        "dbscan_adpt_e0.10_s0.20_m10",
        "dbscan_fixed_e0.15_m5_v0.10",
        "dbscan_fixed_e0.30_m5_v0.20",
    ];

    let mut strategies: Vec<Box<dyn BenchStrategy>> = Vec::new();

    for &w_idx in &wall_indices {
        for &c_idx in &cluster_indices {
            let w_name = w_idx.map(|i| wall_names[i]).unwrap_or("no_wall");
            let full_name = format!("{} + {}", w_name, cluster_names[c_idx]);
            strategies.push(Box::new(PipelineBenchCase::new(&full_name, w_idx, c_idx)));
        }
    }

    println!("策略组合：{}（墙体 {} × 聚类 {}）", strategies.len(), wall_indices.len(), cluster_indices.len());
    println!("排除已知慢速策略：voxel=0.05 DBSCAN、range_image 0.5°\n");

    if std::env::var("QUICK_TEST").is_ok() {
        let keep: Vec<&str> = vec![
            "no_wall + dbscan_adpt_e0.10_s0.20_m10",
            "td_c0.05_d5_m2 + dbscan_adpt_e0.10_s0.20_m10",
            "ransac_d0.05 + dbscan_adpt_e0.10_s0.20_m10",
        ];
        strategies.retain(|s| keep.contains(&s.name()));
        println!("QUICK_TEST 模式：仅 {} 个策略\n", strategies.len());
    }

    strategies
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Warn)
        .init();

    let args: Vec<String> = std::env::args().collect();
    let frame_limit: usize = args.iter()
        .position(|a| a == "--frames")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(100);

    println!("═══ 全流程策略对比 ({} 帧上限) ═══\n", frame_limit);

    let mut strategies = build_strategies();
    let tmp_dir = std::env::temp_dir().join("pipeline_bench");
    let mut recorders: Vec<BenchRecorder> = (0..strategies.len())
        .map(|i| BenchRecorder::new(tmp_dir.join(format!("{}.db", i))).expect("创建 recorder 失败"))
        .collect();
    let harness = BenchHarness::new("./data/cloud", frame_limit);
    let mut preprocessor = GroundPreprocessor::default();
    harness.run(&mut preprocessor, &mut strategies, &mut recorders).await?;

    // ─── 按速度升序排序输出 ─────────────────────────────────────────────
    strategies.sort_by(|a, b| {
        let at = a.stats().total_ms / a.stats().frame_count.max(1) as f64;
        let bt = b.stats().total_ms / b.stats().frame_count.max(1) as f64;
        at.partial_cmp(&bt).unwrap_or(std::cmp::Ordering::Equal)
    });

    println!("\n═══ 按速度升序 ═══");
    println!("{:-<80}", "");
    println!("  {:<36} | {:>5} | {:>6} | {:>4} | {:>7} | {}",
        "策略", "墙%", "剩余", "簇", "ms/帧", "帧数");
    println!("{:-<80}", "");
    for s in &strategies {
        s.summarize();
    }
    println!("{:-<80}", "");
    println!("标记 [OVER] = 帧均超过 100ms，建议排除");

    Ok(())
}
