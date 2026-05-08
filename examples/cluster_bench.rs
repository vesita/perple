use std::time::{Duration, Instant};

use perple::bench::{BenchStrategy, BenchStats, BenchHarness, BenchRecorder, FrameData, GroundPreprocessor};
use perple::cloud::classify::strategy::{ClusteringStrategy, DbscanStrategy, RangeImageStrategy, WallClusterStrategy, LvdotClusterStrategy};
use perple::cloud::wall::{WallPickStrategy, XYRansacWall, XYGrid};
use perple::utils::boxes::Box3D;

// redra 语义材质短名
const MAT_BG: &str = "point_cloud";
const MAT_BOX: &str = "disabled";
const CLUSTER_COLORS: &[&str] = &[
    "cluster_01", "cluster_02", "cluster_03", "cluster_04",
    "cluster_05", "cluster_06", "cluster_07", "cluster_08",
    "cluster_09", "cluster_10", "cluster_11", "cluster_12",
];

struct ClusterBenchCase {
    name: String,
    strategy: Box<dyn ClusteringStrategy>,
    /// true = 输入先经过墙体提取+LV-DOT过滤（模拟真实管线）
    pipeline_mode: bool,
    total_ms: f64,
    frame_count: usize,
    total_clusters: usize,
    total_noise: usize,
    total_humans: usize,
    frame_times: Vec<f64>,
    last_clusters: Vec<Vec<[f32; 3]>>,
    last_noise: usize,
    last_input_n: usize,
    total_input_n: usize,
}

impl ClusterBenchCase {
    fn new(name: &str, strategy: Box<dyn ClusteringStrategy>) -> Self {
        Self {
            name: name.to_string(),
            strategy,
            pipeline_mode: false,
            total_ms: 0.0, frame_count: 0,
            total_clusters: 0, total_noise: 0, total_humans: 0,
            frame_times: Vec::new(),
            last_clusters: Vec::new(), last_noise: 0,
            last_input_n: 0, total_input_n: 0,
        }
    }

    fn with_pipeline(mut self) -> Self {
        self.pipeline_mode = true;
        self
    }
}

impl BenchStrategy for ClusterBenchCase {
    fn name(&self) -> &str { &self.name }

    fn run(&mut self, frame: &FrameData) -> Duration {
        let ng = frame.non_ground().to_vec();
        let start = Instant::now();

        let (sampled, objects) = if self.pipeline_mode {
            // 模拟真实管线：墙体提取 → LV-DOT 体素过滤 → 聚类
            let mut buf = ng.clone();
            let mut wall = XYRansacWall::with_params(0.05, 50, 30).with_seed(42);
            let (n_wall, _) = wall.pick(&mut buf);
            let remaining = &buf[n_wall..];
            let (filtered, _) = XYGrid::voxel_occupancy_filter(remaining, 0.10, 3);
            self.last_input_n = filtered.len();
            self.strategy.run(&filtered)
        } else {
            self.last_input_n = ng.len();
            self.strategy.run(&ng)
        };

        let elapsed = start.elapsed();
        let (clusters, noise) = to_cluster_result(&sampled, &objects);
        let n_humans = count_human_like(&clusters);

        let ms = elapsed.as_secs_f64() * 1000.0;
        self.total_ms += ms;
        self.frame_count += 1;
        self.frame_times.push(ms);
        self.total_clusters += clusters.len();
        self.total_noise += noise;
        self.total_humans += n_humans;
        self.total_input_n += self.last_input_n;
        self.last_clusters = clusters;
        self.last_noise = noise;
        elapsed
    }

    fn write_frame(&mut self, recorder: &mut BenchRecorder, frame: &FrameData) {
        recorder.begin_frame(frame.frame_idx);
        let ng = frame.non_ground();
        let bg_step = (ng.len() / 3000).max(1);
        for i in (0..ng.len()).step_by(bg_step) {
            recorder.write_raw_cloud(&[ng[i]], MAT_BG, 1);
        }
        for (ci, cluster) in self.last_clusters.iter().enumerate() {
            if cluster.is_empty() { continue; }
            let color = CLUSTER_COLORS[ci % CLUSTER_COLORS.len()];
            let step = (cluster.len() / 500).max(1);
            for i in (0..cluster.len()).step_by(step) {
                recorder.write_point_cloud(&[cluster[i]], color, 1);
            }
            let box3d = Box3D::from_cloud_aabb(cluster, 0.05);
            let tag = format!("c{} {}p h{:.1}", ci, cluster.len(), box3d.height);
            recorder.write_boxes(&[(box3d, tag)], MAT_BOX);
        }
        let n = self.frame_count.max(1) as f64;
        println!("[{}] 入{} 簇{} 噪{} 人{} | {:.0}ms",
            self.name, self.last_input_n, self.last_clusters.len(),
            self.last_noise, count_human_like(&self.last_clusters),
            self.total_ms / n);
        recorder.end_frame();
    }

    fn summarize(&self) {
        let n = self.frame_count.max(1) as f64;
        let avg_ms = self.total_ms / n;
        let avg_clusters = self.total_clusters as f64 / n;
        let avg_noise = self.total_noise as f64 / n;
        let avg_humans = self.total_humans as f64 / n;
        let avg_in = self.total_input_n as f64 / n;
        let status = if avg_ms > 100.0 { " [OVER 100ms]" } else { "" };
        println!("  {:<42} | 入{:>5.0} | 簇{:>4.1} | 人{:>4.1} | {:>6.1}ms | {}{}",
            self.name, avg_in, avg_clusters, avg_humans, avg_ms, n as usize, status);
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

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Warn)
        .init();

    println!("=== 室内聚类策略对比（含 LV-DOT 管线） ===\n");

    let mut strategies: Vec<Box<dyn BenchStrategy>> = Vec::new();

    // ═══════════════════════════════════════════════════════════════
    // 第1组: 管线真实路径 — 输入先经墙体提取+LV-DOT过滤，再进聚类
    // ═══════════════════════════════════════════════════════════════
    {
        // LV-DOT direct 变体（墙体→LV-DOT体素下采样→XY DBSCAN）
        for &(voxel, min_occ) in &[(0.10f32, 3usize), (0.15, 2), (0.20, 2), (0.10, 2), (0.15, 3)] {
            for &eps in &[0.20f32, 0.30, 0.40, 0.50] {
                for &min_pts in &[3usize, 5, 8] {
                    let name = format!("pipe_lv_d{:.2}_o{}_e{:.2}_m{}", voxel, min_occ, eps, min_pts);
                    strategies.push(Box::new(ClusterBenchCase::new(
                        &name,
                        Box::new(LvdotClusterStrategy::direct(voxel, min_occ, eps, min_pts)),
                    ).with_pipeline()));
                }
            }
        }

        // LV-DOT box 模式（墙体→box预聚类→LV-DOT下采样→DBSCAN）
        for &(bc, bm) in &[(0.20f32, 3usize), (0.30, 3), (0.30, 5)] {
            for &(voxel, min_occ) in &[(0.10f32, 3usize), (0.10, 2), (0.15, 2)] {
                for &eps in &[0.20f32, 0.30, 0.50] {
                    for &min_pts in &[3usize, 5] {
                        let name = format!("pipe_box_c{:.2}_d{:.2}_o{}_e{:.2}_m{}", bc, voxel, min_occ, eps, min_pts);
                        strategies.push(Box::new(ClusterBenchCase::new(
                            &name,
                            Box::new(LvdotClusterStrategy::direct(voxel, min_occ, eps, min_pts)
                                .with_box_filter(bc, bm, 12.0)),
                        ).with_pipeline()));
                    }
                }
            }
        }

        // dbscan_light（无内部下采样）— 管线已做墙体+LV-DOT，直接进四叉树DBSCAN
        for &eps_0 in &[0.10f32, 0.15, 0.20] {
            for &slope in &[0.0f32, 0.05, 0.10] {
                for &min_pts in &[3usize, 5, 8] {
                    let mode = if slope > 0.0 { "ad" } else { "fx" };
                    let name = format!("pipe_dbl_e{:.2}_s{:.2}_m{}", eps_0, slope, min_pts);
                    strategies.push(Box::new(ClusterBenchCase::new(
                        &name,
                        Box::new(DbscanStrategy::with_params(eps_0, slope, min_pts, 50, 10, 0.0)),
                    ).with_pipeline()));
                    // 每个组合只保留最有代表性的名称
                    if slope == 0.0 && eps_0 != 0.15 { continue; }
                    if slope > 0.0 && eps_0 != 0.15 { continue; }
                }
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // 第2组: 原始 DBSCAN 策略（直接非地面全量，内部自己下采样）
    // ═══════════════════════════════════════════════════════════════
    {
        // 基线
        strategies.push(Box::new(ClusterBenchCase::new(
            "raw_dbscan_ad_e0.10_s0.20_m10_v0.10",
            Box::new(DbscanStrategy::with_params(0.10, 0.20, 10, 50, 10, 0.10)),
        )));

        // 精选固定 eps
        for &eps in &[0.15f32, 0.25, 0.35, 0.50] {
            for &min_pts in &[3usize, 5, 8] {
                for &voxel in &[0.10f32, 0.20] {
                    let name = format!("raw_dbscan_e{:.2}_m{}_v{:.2}", eps, min_pts, voxel);
                    strategies.push(Box::new(ClusterBenchCase::new(
                        &name,
                        Box::new(DbscanStrategy::with_params(eps, 0.0, min_pts, 50, 10, voxel)),
                    )));
                }
            }
        }

        // 精选自适应 eps
        for &eps_0 in &[0.10f32, 0.15] {
            for &slope in &[0.05f32, 0.10] {
                for &min_pts in &[3usize, 5, 8] {
                    for &voxel in &[0.10f32, 0.20] {
                        let name = format!("raw_dbscan_ad_e{:.2}_s{:.2}_m{}_v{:.2}", eps_0, slope, min_pts, voxel);
                        strategies.push(Box::new(ClusterBenchCase::new(
                            &name,
                            Box::new(DbscanStrategy::with_params(eps_0, slope, min_pts, 50, 10, voxel)),
                        )));
                    }
                }
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // 第3组: WallCluster + RangeImage（现有参考）
    // ═══════════════════════════════════════════════════════════════
    {
        for &cell_size in &[0.20f32, 0.30] {
            for &dbscan_eps in &[0.20f32, 0.30, 0.50] {
                for &dbscan_min in &[3usize, 5] {
                    let name = format!("wall_c{:.2}_e{:.2}_m{}", cell_size, dbscan_eps, dbscan_min);
                    strategies.push(Box::new(ClusterBenchCase::new(
                        &name,
                        Box::new(WallClusterStrategy::with_params(
                            Box::new(XYRansacWall::with_params(0.05, 50, 30).with_seed(42)),
                            cell_size, 3, 12.0, dbscan_eps, dbscan_min,
                        )),
                    )));
                }
            }
        }

        for &(az, el, thresh, min_pts, label) in &[
            (1.0, 1.0, 0.5, 3, "range_1.0_t0.5_m3"),
            (1.0, 1.0, 1.0, 3, "range_1.0_t1.0_m3"),
            (2.0, 2.0, 1.0, 3, "range_2.0_t1.0_m3"),
        ] {
            strategies.push(Box::new(ClusterBenchCase::new(
                label,
                Box::new(RangeImageStrategy::with_params(az, el, thresh, min_pts)),
            )));
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // 快速模式过滤
    // ═══════════════════════════════════════════════════════════════
    if std::env::var("FULL").is_err() {
        strategies.retain(|s| {
            let n = s.name();
            n.starts_with("pipe_lv_") || n.starts_with("pipe_box_") || n.starts_with("pipe_dbl_")
                || n == "raw_dbscan_ad_e0.10_s0.20_m10_v0.10"
                || n == "raw_dbscan_e0.15_m5_v0.10" || n == "raw_dbscan_e0.25_m5_v0.10"
                || n == "raw_dbscan_e0.35_m5_v0.10" || n == "raw_dbscan_ad_e0.15_s0.10_m5_v0.10"
                || n.starts_with("wall_c0") || n.starts_with("range_1")
        });
    }
    println!("共 {} 个策略（{} 模式）\n", strategies.len(),
        if std::env::var("FULL").is_ok() { "FULL" } else { "QUICK" });

    let harness = BenchHarness::new("./data/test", 32, "output/cluster_bench");
    let mut preprocessor = GroundPreprocessor::default();
    harness.run(&mut preprocessor, &mut strategies).await?;

    // 按平均耗时排序输出
    strategies.sort_by(|a, b| {
        let at = a.stats().total_ms / a.stats().frame_count.max(1) as f64;
        let bt = b.stats().total_ms / b.stats().frame_count.max(1) as f64;
        at.partial_cmp(&bt).unwrap_or(std::cmp::Ordering::Equal)
    });

    println!("\n=== 按速度升序 ===");
    println!("{:-<100}", "");
    println!("| {:<42} | {:>5} | {:>4} | {:>4} | {:>7} | {:>4} |",
        "策略", "输入", "簇", "人", "ms/帧", "帧");
    println!("{:-<100}", "");
    for s in &strategies { s.summarize(); }
    println!("{:-<100}", "");

    Ok(())
}

fn to_cluster_result(points: &[[f32; 3]], objects: &[Vec<usize>]) -> (Vec<Vec<[f32; 3]>>, usize) {
    let total: usize = objects.iter().map(|c| c.len()).sum();
    let clusters: Vec<Vec<[f32; 3]>> = objects.iter()
        .map(|c| c.iter().map(|&i| points[i]).collect())
        .collect();
    (clusters, points.len() - total)
}

fn count_human_like(clusters: &[Vec<[f32; 3]>]) -> usize {
    let mut count = 0;
    for cluster in clusters {
        if cluster.len() < 3 { continue; }
        let box3d = Box3D::from_cloud_aabb(cluster, 0.05);
        let w = box3d.length.max(box3d.width);
        let h = box3d.height;
        if w < 0.25 || h < 0.5 { continue; }
        if h > w * 0.5 && w < 1.2 && h > 1.0 && h < 2.5 { count += 1; }
    }
    count
}
