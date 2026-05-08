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
    WallPickStrategy, TopDownCluster, XYRansacWall, QuadtreeWall, XYGrid,
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
    n_humans: usize,
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
    acc_after_voxel: usize,
    acc_after_range: usize,
    acc_clusters: usize,
    acc_humans: usize,
    frame_times: Vec<f64>,
    last: FrameStats,
    last_boxes: Vec<(Box3D, bool)>, // (包围盒, 是否行人) 供 write_frame 输出
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
            acc_after_voxel: 0,
            acc_after_range: 0,
            acc_clusters: 0,
            acc_humans: 0,
            frame_times: Vec::new(),
            last: FrameStats::default(),
            last_boxes: Vec::new(),
        }
    }

    /// 创建对应索引的墙体策略（每次 new 一个，避免 clone 问题）
    fn create_wall(&self) -> Option<Box<dyn WallPickStrategy>> {
        match self.wall {
            Some(0) => Some(Box::new(TopDownCluster::with_params(0.05, 5, 2))),
            Some(1) => Some(Box::new(TopDownCluster::with_params(0.10, 3, 2))),
            Some(2) => Some(Box::new(XYRansacWall::with_params(0.05, 50, 30))),
            Some(3) => Some(Box::new(XYRansacWall::with_params(0.08, 50, 30))),
            Some(4) => Some(Box::new(QuadtreeWall::with_params(0.10, 3, 1.5).with_merge_dist(2))),
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
        let n_wall = if let Some(ref mut w) = self.create_wall() {
            let (n, _) = w.pick(&mut buf);
            n
        } else {
            0
        };

        // 2. LV-DOT 体素占用过滤
        let (after_voxel, _) = XYGrid::voxel_occupancy_filter(&buf[n_wall..], 0.10, self.voxel_min_occ);

        let n_after_voxel = after_voxel.len();

        // 3. 天花板 + 范围过滤
        let config = fixif();
        let mut cluster_input = after_voxel;
        if config.claster.ceiling_filter && config.claster.ceiling_height > 0.0 {
            cluster_input.retain(|p| p[2] <= config.claster.ceiling_height);
        }
        if config.claster.max_range > 0.0 {
            let max_d2 = config.claster.max_range * config.claster.max_range;
            cluster_input.retain(|p| p[0] * p[0] + p[1] * p[1] + p[2] * p[2] <= max_d2);
        }

        // 4. 聚类
        let mut cluster = self.create_cluster();
        let (sampled, objects) = cluster.run(&cluster_input);
        let n_clusters = objects.len();
        let n_humans = count_human_like(&objects, &sampled);

        // 构建输出可视化包围盒
        let mut last_boxes = Vec::with_capacity(objects.len());
        for obj in &objects {
            if obj.len() < 3 { continue; }
            let pts: Vec<[f32; 3]> = obj.iter().map(|&i| sampled[i]).collect();
            let b = Box3D::from_cloud_aabb(&pts, 0.05);
            let w = b.length.max(b.width);
            let h = b.height;
            if w < 0.25 || h < 0.5 { continue; }
            let is_human = h > w * 0.5 && w < 1.2 && h > 1.0 && h < 2.5;
            last_boxes.push((b, is_human));
        }
        self.last_boxes = last_boxes;

        self.last = FrameStats {
            n_wall,
            n_after_voxel,
            n_after_range: cluster_input.len(),
            n_sampled: sampled.len(),
            n_clusters,
            n_humans,
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
        self.acc_humans += n_humans;

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

        // 写入聚类包围盒（行人为绿色 person，障碍物为灰色 disabled）
        let mut human_boxes = Vec::new();
        let mut obst_boxes = Vec::new();
        for (box3d, is_human) in &self.last_boxes {
            let tag = if *is_human { "person".into() } else { "obstacle".into() };
            if *is_human {
                human_boxes.push((box3d.clone(), tag));
            } else {
                obst_boxes.push((box3d.clone(), tag));
            }
        }
        if !human_boxes.is_empty() {
            recorder.write_boxes(&human_boxes, "person");
        }
        if !obst_boxes.is_empty() {
            recorder.write_boxes(&obst_boxes, "disabled");
        }

        recorder.end_frame();

        // 每帧进度显示
        if self.frame_count % 20 == 0 {
            let avg_ms = self.total_ms / self.frame_count as f64;
            println!("  [{}] 帧 {} | 墙={} 剩余={} 簇={} 人={} | {:.0}ms/帧",
                self.name, self.frame_count, self.last.n_wall,
                self.last.n_after_range, self.last.n_clusters, self.last.n_humans, avg_ms);
        }
    }

    fn summarize(&self) {
        let n = self.frame_count.max(1) as f64;
        let avg_ms = self.total_ms / n;
        let avg_remain = self.acc_after_range as f64 / n;
        let avg_clusters = self.acc_clusters as f64 / n;
        let avg_humans = self.acc_humans as f64 / n;
        let total_in = (self.acc_wall + self.acc_after_voxel) as f64;
        let wall_pct = if total_in > 0.0 {
            self.acc_wall as f64 / total_in * 100.0
        } else {
            0.0
        };
        let status = if avg_ms > 100.0 { " [OVER]" } else { "" };
        println!("  {:<36} | {:>5.0}% | {:>6.0} | {:>4.1} | {:>4.1} | {:>6.1}ms | {}{}",
            self.name, wall_pct, avg_remain, avg_clusters, avg_humans, avg_ms, n as usize, status);
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
        Some(0),        // 1: TopDownCluster 0.05/5/2
        Some(1),        // 2: TopDownCluster 0.10/3/2
        Some(2),        // 3: XYRansacWall 0.05/50/30（当前默认）
        Some(3),        // 4: XYRansacWall 0.08/50/30
        Some(4),        // 5: QuadtreeWall 0.10/3/1.5
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
    let harness = BenchHarness::new("./data/test", frame_limit, "output/pipeline_bench");
    let mut preprocessor = GroundPreprocessor::default();
    harness.run(&mut preprocessor, &mut strategies).await?;

    // ─── 按速度升序排序输出 ─────────────────────────────────────────────
    // 按速度排序
    strategies.sort_by(|a, b| {
        let at = a.stats().total_ms / a.stats().frame_count.max(1) as f64;
        let bt = b.stats().total_ms / b.stats().frame_count.max(1) as f64;
        at.partial_cmp(&bt).unwrap_or(std::cmp::Ordering::Equal)
    });

    println!("\n═══ 按速度升序 ═══");
    println!("{:-<88}", "");
    println!("  {:<36} | {:>5} | {:>6} | {:>4} | {:>4} | {:>7} | {}",
        "策略", "墙%", "剩余", "簇", "人", "ms/帧", "帧数");
    println!("{:-<88}", "");
    for s in &strategies {
        s.summarize();
    }
    println!("{:-<88}", "");
    println!("标记 [OVER] = 帧均超过 100ms，建议排除");

    Ok(())
}

/// 统计类人物体数量（宽<1.2m, 高1.0-2.5m, 高>宽×0.5）
fn count_human_like(objects: &[Vec<usize>], points: &[[f32; 3]]) -> usize {
    let mut count = 0;
    for obj in objects {
        if obj.len() < 3 { continue; }
        let pts: Vec<[f32; 3]> = obj.iter().map(|&i| points[i]).collect();
        let b = Box3D::from_cloud_aabb(&pts, 0.05);
        let w = b.length.max(b.width);
        let h = b.height;
        if w < 0.25 || h < 0.5 { continue; }
        if h > w * 0.5 && w < 1.2 && h > 1.0 && h < 2.5 {
            count += 1;
        }
    }
    count
}
