use std::time::{Duration, Instant};

use perple::bench::{BenchStrategy, BenchHarness, BenchRecorder, FrameData, GroundPreprocessor};
use perple::cloud::classify::strategy::{ClusteringStrategy, DbscanStrategy, RangeImageStrategy};
use perple::utils::boxes::Box3D;

// redra 语义材质短名
const MAT_BG: &str = "point_cloud";        // 暖白背景
const MAT_BOX: &str = "glass";             // 半透明包围盒，可透视内部点

// 聚类色板（12 色最大感知区分，30° 色相间距）
const CLUSTER_COLORS: &[&str] = &[
    "cluster_01", "cluster_02", "cluster_03", "cluster_04",
    "cluster_05", "cluster_06", "cluster_07", "cluster_08",
    "cluster_09", "cluster_10", "cluster_11", "cluster_12",
];

struct ClusterBenchCase {
    name: String,
    strategy: Box<dyn ClusteringStrategy>,
    total_ms: f64,
    frame_count: usize,
    total_clusters: usize,
    total_noise: usize,
    total_humans: usize,
    last_clusters: Option<Vec<Vec<[f32; 3]>>>,
}

impl ClusterBenchCase {
    fn new(name: &str, strategy: Box<dyn ClusteringStrategy>) -> Self {
        Self {
            name: name.to_string(),
            strategy,
            total_ms: 0.0,
            frame_count: 0,
            total_clusters: 0,
            total_noise: 0,
            total_humans: 0,
            last_clusters: None,
        }
    }
}

impl BenchStrategy for ClusterBenchCase {
    fn name(&self) -> &str { &self.name }

    fn run(&mut self, frame: &FrameData) -> Duration {
        let points = frame.non_ground().to_vec();
        let start = Instant::now();
        let (processed, objects) = self.strategy.run(&points);
        let elapsed = start.elapsed();

        let (clusters, noise) = to_cluster_result(&processed, &objects);
        let n_humans = count_human_like(&clusters);

        self.total_ms += elapsed.as_secs_f64() * 1000.0;
        self.frame_count += 1;
        self.total_clusters += clusters.len();
        self.total_noise += noise;
        self.total_humans += n_humans;
        self.last_clusters = Some(clusters);

        elapsed
    }

    fn write_frame(&mut self, recorder: &mut BenchRecorder, frame: &FrameData) {
        if let Some(ref clusters) = self.last_clusters {
            if clusters.is_empty() { return; }

            recorder.begin_frame(frame.frame_idx);

            // 非地面输入点背景（受 write_raw 开关控制，默认关闭避免与簇点云重复）
            let ng = frame.non_ground();
            let bg_step = (ng.len() / 3000).max(1);
            for i in (0..ng.len()).step_by(bg_step) {
                recorder.write_raw_cloud(&[ng[i]], MAT_BG, 1);
            }

            // 各簇点云 + 包围盒（12 色聚类色板循环）
            for (ci, cluster) in clusters.iter().enumerate() {
                if cluster.is_empty() { continue; }
                let color = CLUSTER_COLORS[ci % CLUSTER_COLORS.len()];
                let step = (cluster.len() / 1000).max(1);
                for i in (0..cluster.len()).step_by(step) {
                    recorder.write_point_cloud(&[cluster[i]], color, 1);
                }
                let mut box3d = Box3D::empty_box();
                box3d.cloud2box(cluster);
                let tag = format!("{}pts h={:.1}m", cluster.len(), box3d.height);
                recorder.write_boxes(&[(box3d, tag)], MAT_BOX);
            }

            // 汇总信息（半透明框，与簇包围盒一致）
            let n = self.frame_count.max(1) as f64;
            let summary_tag = format!("{} | {:.1}簇 {:.1}人 {:.0}ms/帧",
                self.name,
                self.total_clusters as f64 / n,
                self.total_humans as f64 / n,
                self.total_ms / n);
            let dummy = Box3D::empty_box();
            recorder.write_boxes(&[(dummy, summary_tag)], MAT_BOX);

            recorder.end_frame();
        }
    }

    fn summarize(&self) {
        let n = self.frame_count.max(1) as f64;
        let avg_clusters = self.total_clusters as f64 / n;
        let avg_noise = self.total_noise as f64 / n;
        let avg_humans = self.total_humans as f64 / n;
        let avg_ms = self.total_ms / n;
        let human_ratio = if self.total_clusters > 0 {
            self.total_humans as f64 / self.total_clusters as f64 * 100.0
        } else {
            0.0
        };
        let status = if avg_ms > 100.0 { " [OVER 100ms]" } else { "" };
        println!("  {:<32} | 簇 {:>5.1} | 噪 {:>5.0} | 人 {:>5.1} | {:>6.1}ms | 人占比 {:>4.0}% | {} 帧{}",
            self.name, avg_clusters, avg_noise, avg_humans, avg_ms, human_ratio, self.frame_count, status);
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();

    println!("=== 室内聚类策略对比测试（串行） ===\n");

    let mut strategies: Vec<Box<dyn BenchStrategy>> = Vec::new();

    // 策略 1: 当前默认参数
    strategies.push(Box::new(ClusterBenchCase::new(
        "默认 eps0.20",
        Box::new(DbscanStrategy::with_params(0.20, 0.0, 10, 50, 10, 0.10)),
    )));

    // 策略 2: 固定 eps DBSCAN
    // 剔除 v0.05：体素过小导致邻域查询慢，实测 118-275ms/帧（v0.10/v0.20 均 <100ms）
    for &voxel in &[0.10f32, 0.20] {
        for &eps in &[0.15f32, 0.25, 0.35, 0.50, 0.80] {
            for &min_pts in &[3usize, 5, 8, 15] {
                let name = format!("eps{:.2}_m{}_v{:.2}", eps, min_pts, voxel);
                strategies.push(Box::new(ClusterBenchCase::new(
                    &name,
                    Box::new(DbscanStrategy::with_params(eps, 0.0, min_pts, 50, 10, voxel)),
                )));
            }
        }
    }

    // 策略 3: 自适应 eps DBSCAN
    // 剔除 v0.05：同上，155-330ms/帧
    for &voxel in &[0.10f32, 0.20] {
        for &eps_0 in &[0.05f32, 0.10, 0.15] {
            for &slope in &[0.02f32, 0.05, 0.10] {
                for &min_pts in &[3usize, 5, 8, 15] {
                    let name = format!("adapt_e{:.2}_s{:.2}_m{}_v{:.2}", eps_0, slope, min_pts, voxel);
                    strategies.push(Box::new(ClusterBenchCase::new(
                        &name,
                        Box::new(DbscanStrategy::with_params(eps_0, slope, min_pts, 50, 10, voxel)),
                    )));
                }
            }
        }
    }

    // 策略 4: 无下采样 — 剔除
    // 原因：跳过体素化直接对全量点云做 DBSCAN，点数多必然超时

    // 策略 5: Range Image
    // 剔除 0.5° 分辨率：网格过密（720×360 像素），实测超时
    for &(az, el, thresh, min_pts, label) in &[
        (1.0, 1.0, 0.5, 3, "ri_1.0deg_t0.5_m3"),
        (1.0, 1.0, 1.0, 3, "ri_1.0deg_t1.0_m3"),
        (2.0, 2.0, 1.0, 3, "ri_2.0deg_t1.0_m3"),
    ] {
        strategies.push(Box::new(ClusterBenchCase::new(
            label,
            Box::new(RangeImageStrategy::with_params(az, el, thresh, min_pts)),
        )));
    }

    println!("共 {} 个策略\n", strategies.len());

    // 全量帧执行，输出到 output/cluster_bench
    let harness = BenchHarness::new("./data/test", 64, "output/cluster_bench");
    let mut preprocessor = GroundPreprocessor::default();
    harness.run(&mut preprocessor, &mut strategies).await?;

    // 按平均耗时排序，标注超 100ms 的策略
    println!("\n=== 按平均耗时排序 ===");
    println!("{:-<90}", "");
    println!("| {:<32} | {:>8} | {:>6} | {:>5} | {:>7} |",
        "策略", "平均ms", "帧均簇", "帧均人", "帧数");
    println!("{:-<90}", "");

    // Collect summary data for sorting (we can't sort in-place through trait objects easily,
    // so we just print the summary from each strategy which already handles the format)

    println!("{:-<90}", "");
    println!("提示：标记 [OVER 100ms] 的策略建议排除出对比");

    Ok(())
}

fn to_cluster_result(points: &[[f32; 3]], objects: &[Vec<usize>]) -> (Vec<Vec<[f32; 3]>>, usize) {
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
        box3d.cloud2box(cluster);
        let w = box3d.length.max(box3d.width);
        let h = box3d.height;
        if w < 0.25 || h < 0.5 { continue; }
        if h > w * 0.5 && w < 1.2 && h > 1.0 && h < 2.5 {
            count += 1;
        }
    }
    count
}
