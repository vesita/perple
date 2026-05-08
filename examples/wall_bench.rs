use std::collections::HashSet;
use std::time::{Duration, Instant};

use perple::bench::{BenchStrategy, BenchStats, BenchHarness, BenchRecorder, FrameData, WallPreprocessor};
use perple::cloud::wall::{
    WallPickStrategy, TopDownCluster, XYRansacWall,
    NormalWall, cluster_obstacles_with_indices,
};
use perple::utils::boxes::Box3D;

// redra 语义材质短名（高对比度配色）
const MAT_WALL: &str = "red";              // 红色墙面点
const MAT_REMAIN: &str = "yellow";         // 黄色待选点（交下一步重聚类）
const MAT_DISCARD: &str = "blue";          // 蓝色弃置点（box 过滤掉的墙体残片/小簇）
const MAT_BOX: &str = "disabled";          // 暗灰半透明障碍物包围盒（近距）
const MAT_FAR_BOX: &str = "cluster_01";    // 远距丢弃包围盒（青色）

struct WallBenchCase {
    name: String,
    strategy: Box<dyn WallPickStrategy>,
    min_box_pts: usize,
    total_ms: f64,
    frame_count: usize,
    total_wall_points: usize,
    total_obstacles: usize,
    total_far_obstacles: usize,
    total_discarded_pts: usize,
    frame_times: Vec<f64>,
    last_n_wall: usize,
    last_cloud: Vec<[f32; 3]>,
    last_obstacles: Vec<Box3D>,
    last_far_obstacles: Vec<(Box3D, f32)>,  // (box, 质心距离m)
    last_discarded_abs_indices: Vec<usize>,  // 被过滤 box 内点的绝对索引
}

impl WallBenchCase {
    fn new(name: &str, strategy: Box<dyn WallPickStrategy>, min_box_pts: usize) -> Self {
        Self {
            name: name.to_string(),
            strategy,
            min_box_pts,
            total_ms: 0.0,
            frame_count: 0,
            total_wall_points: 0,
            total_obstacles: 0,
            total_far_obstacles: 0,
            total_discarded_pts: 0,
            frame_times: Vec::new(),
            last_n_wall: 0,
            last_cloud: Vec::new(),
            last_obstacles: Vec::new(),
            last_far_obstacles: Vec::new(),
            last_discarded_abs_indices: Vec::new(),
        }
    }
}

impl BenchStrategy for WallBenchCase {
    fn name(&self) -> &str { &self.name }

    fn run(&mut self, frame: &FrameData) -> Duration {
        let mut cloud = frame.non_ground().to_vec();
        let start = Instant::now();
        let (n_wall, _planes) = self.strategy.pick(&mut cloud);
        // 对非墙面点做障碍物聚类（带回索引，不限距离）
        let (all_boxes, all_indices) =
            cluster_obstacles_with_indices(&cloud[n_wall..], 0.30, 3, 0.05, 0.0);
        let elapsed = start.elapsed();

        // box 过滤：法线检测 + 最小点数 + 远近分类
        let wall_nz_threshold: f32 = 0.15;
        let remaining = &cloud[n_wall..];
        let max_d2 = 12.0f32 * 12.0;
        let mut obstacles = Vec::new();
        let mut far_obstacles = Vec::new();
        let mut discarded_abs_indices: Vec<usize> = Vec::new();

        for (b, indices) in all_boxes.into_iter().zip(all_indices.into_iter()) {
            let mut discard = false;

            // 最小点数检查（无下采样=20，有下采样=10）
            if indices.len() < self.min_box_pts {
                discard = true;
            }

            // 3D PCA 法线检查：|n_z| < 0.15 → 墙体残片
            if !discard {
                if let Some((normal, _)) = fit_plane_3d_wallbench(&indices, remaining, b.height) {
                    if normal[2].abs() < wall_nz_threshold {
                        discard = true;
                    }
                }
            }

            if discard {
                for &rel_idx in &indices {
                    discarded_abs_indices.push(n_wall + rel_idx);
                }
                continue;
            }

            let c = b.center();
            let d2 = c[0] * c[0] + c[1] * c[1];
            if d2 <= max_d2 {
                obstacles.push(b);
            } else {
                far_obstacles.push((b, d2.sqrt()));
            }
        }

        let ms = elapsed.as_secs_f64() * 1000.0;
        self.total_ms += ms;
        self.frame_count += 1;
        self.total_wall_points += n_wall;
        self.total_obstacles += obstacles.len();
        self.total_far_obstacles += far_obstacles.len();
        self.total_discarded_pts += discarded_abs_indices.len();
        self.frame_times.push(ms);
        self.last_n_wall = n_wall;
        self.last_obstacles = obstacles;
        self.last_far_obstacles = far_obstacles;
        self.last_discarded_abs_indices = discarded_abs_indices;
        self.last_cloud = cloud;

        elapsed
    }

    fn write_frame(&mut self, recorder: &mut BenchRecorder, frame: &FrameData) {
        recorder.begin_frame(frame.frame_idx);

        let cloud = &self.last_cloud;
        let n_wall = self.last_n_wall;

        // 构建弃置点集合（绝对索引 → O(1) 查找）
        let discarded_set: HashSet<usize> =
            self.last_discarded_abs_indices.iter().copied().collect();

        // 1. 墙面点（红色）
        let wall_step = (n_wall / 2000).max(1);
        for i in (0..n_wall).step_by(wall_step) {
            recorder.write_point_cloud(&[cloud[i]], MAT_WALL, 1);
        }

        // 2. 弃置点（蓝色）— 被 box 过滤掉的墙体残片/小簇
        let disc_n = self.last_discarded_abs_indices.len();
        let disc_step = (disc_n / 1000).max(1);
        for k in (0..disc_n).step_by(disc_step) {
            let idx = self.last_discarded_abs_indices[k];
            recorder.write_point_cloud(&[cloud[idx]], MAT_DISCARD, 1);
        }

        // 3. 待选点（黄色）— 非墙面非弃置，交给下一步重聚类
        let remaining = cloud.len() - n_wall;
        let remain_step = (remaining / 3000).max(1);
        for i in (n_wall..cloud.len()).step_by(remain_step) {
            if !discarded_set.contains(&i) {
                recorder.write_point_cloud(&[cloud[i]], MAT_REMAIN, 1);
            }
        }

        // 近距障碍物包围盒（≤12m，暗灰色）
        let obstacle_tags: Vec<(Box3D, String)> = self.last_obstacles.iter().enumerate()
            .map(|(i, b)| {
                let tag = format!("obj{} {:.1}x{:.1}x{:.1}", i, b.length, b.width, b.height);
                (b.clone(), tag)
            })
            .collect();
        recorder.write_boxes(&obstacle_tags, MAT_BOX);

        // 远距障碍物包围盒（>12m，青色，含距离便于脚本分析）
        let far_tags: Vec<(Box3D, String)> = self.last_far_obstacles.iter().enumerate()
            .map(|(i, (b, dist))| {
                let tag = format!("far{} d{:.0}m {:.1}x{:.1}x{:.1}", i, dist, b.length, b.width, b.height);
                (b.clone(), tag)
            })
            .collect();
        recorder.write_boxes(&far_tags, MAT_FAR_BOX);

        // 进度打印（含弃置点数）
        let n = self.frame_count.max(1) as f64;
        let avg_ms = self.total_ms / n;
        let avg_obs = self.total_obstacles as f64 / n;
        let avg_far = self.total_far_obstacles as f64 / n;
        let avg_wall_pts = self.total_wall_points as f64 / n;
        let avg_disc = self.total_discarded_pts as f64 / n;
        println!("[{}] 墙={} 弃={} 近={} 远={} 待选≈{} | 累计 {:.0}墙 {:.0}弃 {:.1}近 {:.1}远 {:.0}ms",
            self.name, n_wall, disc_n, self.last_obstacles.len(), self.last_far_obstacles.len(),
            remaining.saturating_sub(disc_n), avg_wall_pts, avg_disc, avg_obs, avg_far, avg_ms);

        recorder.end_frame();
    }

    fn summarize(&self) {
        let n = self.frame_count.max(1) as f64;
        let avg_ms = self.total_ms / n;
        let avg_obs = self.total_obstacles as f64 / n;
        let avg_far = self.total_far_obstacles as f64 / n;
        let avg_wall_pts = self.total_wall_points as f64 / n;
        let status = if avg_ms > 100.0 { " [OVER 100ms]" } else { "" };
        println!("  {:<32} | 墙点 {:>6.0} | 近 {:>4.1} 远 {:>4.1} | {:>6.1}ms | {} 帧{}",
            self.name, avg_wall_pts, avg_obs, avg_far, avg_ms, self.frame_count, status);
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
        .filter_level(log::LevelFilter::Info)
        .init();

    println!("=== 墙体提取策略对比测试（输出障碍物 box）===\n");

    let mut strategies: Vec<Box<dyn BenchStrategy>> = Vec::new();

    // ─── 基线参考（非 NormalWall 中最优，无下采样 → min_box_pts=20） ───
    strategies.push(Box::new(WallBenchCase::new(
        "td_c0.05_d5_m2_w0.15",
        Box::new(TopDownCluster::with_params(0.05, 5, 2).with_width_ratio(0.15)),
        20,
    )));
    strategies.push(Box::new(WallBenchCase::new(
        "xy_ransac_d0.05_i100_rng",
        Box::new(XYRansacWall::with_params(0.05, 100, 30)),
        20,
    )));
    strategies.push(Box::new(WallBenchCase::new(
        "xy_ransac_d0.05_i50_s42",
        Box::new(XYRansacWall::with_params(0.05, 50, 30).with_seed(42)),
        20,
    )));
    strategies.push(Box::new(WallBenchCase::new(
        "xy_ransac_d0.05_i50_rng",
        Box::new(XYRansacWall::with_params(0.05, 50, 30)),
        20,
    )));
    strategies.push(Box::new(WallBenchCase::new(
        "xy_ransac_d0.08_i50_s42",
        Box::new(XYRansacWall::with_params(0.08, 50, 30).with_seed(42)),
        20,
    )));
    strategies.push(Box::new(WallBenchCase::new(
        "xy_ransac_d0.08_i50_rng",
        Box::new(XYRansacWall::with_params(0.08, 50, 30)),
        20,
    )));

    // ─── NormalWall 交叉网格搜索 ───
    // 参数解释: c=cell_size p=min_pts f=far_distance z=normal_z_threshold
    //           zg=z_gap mzs=min_z_span ds=downsample_target

    // 核心交叉: cell_size × normal_z_threshold（无下采样 → min_box_pts=20）
    for &cs in &[0.05f32, 0.07, 0.10, 0.15, 0.20, 0.25, 0.30] {
        for &nz in &[0.08f32, 0.12, 0.17, 0.22] {
            let name = format!("nw_c{:.2}_z{:.2}", cs, nz);
            strategies.push(Box::new(WallBenchCase::new(
                &name,
                Box::new(NormalWall::with_params(cs, 10, 30.0).with_normal_threshold(nz)),
                20,
            )));
        }
    }

    // Z 分层切分（无下采样 → min_box_pts=20）
    for &cs in &[0.25f32, 0.30] {
        for &nz in &[0.12f32, 0.17] {
            for &(zg, mzs) in &[(0.20f32, 0.50), (0.30, 0.80), (0.40, 1.00)] {
                let name = format!("nw_c{:.2}_z{:.2}_zg{:.2}_mzs{:.2}", cs, nz, zg, mzs);
                strategies.push(Box::new(WallBenchCase::new(
                    &name,
                    Box::new(NormalWall::with_params(cs, 10, 30.0)
                        .with_normal_threshold(nz)
                        .with_z_split(zg, mzs)),
                    20,
                )));
            }
        }
    }

    // 下采样 + 大 cell（有下采样 → min_box_pts=10）
    for &ds in &[2000usize, 5000] {
        for &cs in &[0.25f32, 0.30, 0.40] {
            for &nz in &[0.12f32, 0.17] {
                let name = format!("nw_c{:.2}_z{:.2}_ds{}", cs, nz, ds);
                strategies.push(Box::new(WallBenchCase::new(
                    &name,
                    Box::new(NormalWall::with_params(cs, 10, 30.0)
                        .with_normal_threshold(nz)
                        .with_downsample_target(ds)),
                    10,
                )));
            }
        }
    }

    let mode = if std::env::var("FULL").is_ok() { "FULL" } else { "QUICK" };
    if std::env::var("FULL").is_err() {
        // 快速模式：只保留最佳候选
        strategies.retain(|s| {
            let n = s.name();
            n == "td_c0.05_d5_m2_w0.15"
                || n == "xy_ransac_d0.05_i100_rng"
                || n == "xy_ransac_d0.05_i50_s42"
                || n == "xy_ransac_d0.05_i50_rng"
                || n == "xy_ransac_d0.08_i50_s42"
                || n == "xy_ransac_d0.08_i50_rng"
                || n == "nw_c0.05_z0.17" || n == "nw_c0.05_z0.22"
                || n == "nw_c0.07_z0.17" || n == "nw_c0.07_z0.22"
                || n == "nw_c0.10_z0.12" || n == "nw_c0.10_z0.17"
                || n == "nw_c0.15_z0.17"
                || n == "nw_c0.20_z0.12" || n == "nw_c0.20_z0.17"
                || n == "nw_c0.25_z0.17"
                || n == "nw_c0.30_z0.17"
        });
    }

    println!("=== NormalWall 参数优化 ({}) ===\n", mode);
    println!("共 {} 个策略\n", strategies.len());

    let harness = BenchHarness::new("./data/test", 10, "output/wall_bench");
    let mut preprocessor = WallPreprocessor::default();
    harness.run(&mut preprocessor, &mut strategies).await?;

    println!("\n提示：标记 [OVER 100ms] 的策略建议排除出对比");

    Ok(())
}

/// 3D PCA 法线检测：对给定点索引计算最小特征值方向。
///
/// 仅用于 box 级墙体过滤。返回 (normal, d)。|n_z| < threshold → 墙体残片，应跳过。
fn fit_plane_3d_wallbench(
    indices: &[usize],
    points: &[[f32; 3]],
    box_h: f32,
) -> Option<([f32; 3], f32)> {
    let n = indices.len();
    if n < 10 { return None; }  // 点太少不可靠
    if box_h < 0.8 { return None; }  // 低矮 box 不会是墙
    let nf = n as f32;
    let mut cx = 0.0f32; let mut cy = 0.0f32; let mut cz = 0.0f32;
    for &i in indices {
        let p = &points[i];
        cx += p[0]; cy += p[1]; cz += p[2];
    }
    cx /= nf; cy /= nf; cz /= nf;

    let mut cov = nalgebra::Matrix3::zeros();
    for &i in indices {
        let p = &points[i];
        let dx = p[0] - cx; let dy = p[1] - cy; let dz = p[2] - cz;
        cov[(0, 0)] += dx * dx; cov[(0, 1)] += dx * dy; cov[(0, 2)] += dx * dz;
        cov[(1, 1)] += dy * dy; cov[(1, 2)] += dy * dz;
        cov[(2, 2)] += dz * dz;
    }
    cov /= nf;
    cov[(1, 0)] = cov[(0, 1)]; cov[(2, 0)] = cov[(0, 2)]; cov[(2, 1)] = cov[(1, 2)];

    let eig = cov.symmetric_eigen();
    let mut min_idx = 0;
    let mut min_val = eig.eigenvalues[0];
    for i in 1..3 {
        if eig.eigenvalues[i] < min_val { min_val = eig.eigenvalues[i]; min_idx = i; }
    }
    let nv = eig.eigenvectors.column(min_idx);
    let normal = [nv[0], nv[1], nv[2]];
    let d = -(normal[0] * cx + normal[1] * cy + normal[2] * cz);
    Some((normal, d))
}
