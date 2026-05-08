use std::time::{Duration, Instant};

use perple::bench::{BenchStrategy, BenchStats, BenchHarness, BenchRecorder, FrameData, WallPreprocessor};
use perple::cloud::wall::{
    WallPickStrategy, TopDownCluster, QuadtreeWall, XYRansacWall,
};
use perple::utils::boxes::Box3D;

// redra 语义材质短名（高对比度配色）
const MAT_WALL: &str = "red";              // 红色墙面
const MAT_REMAIN: &str = "yellow";         // 黄色非墙面剩余点
const MAT_BOX: &str = "disabled";          // 暗灰半透明包围盒

struct WallBenchCase {
    name: String,
    strategy: Box<dyn WallPickStrategy>,
    total_ms: f64,
    frame_count: usize,
    total_wall_points: usize,
    total_walls: usize,
    frame_times: Vec<f64>,
    last_n_wall: usize,
    last_wall_planes: Vec<[f32; 4]>,
    last_wall_counts: Vec<usize>,
    last_cloud: Vec<[f32; 3]>,
}

impl WallBenchCase {
    fn new(name: &str, strategy: Box<dyn WallPickStrategy>) -> Self {
        Self {
            name: name.to_string(),
            strategy,
            total_ms: 0.0,
            frame_count: 0,
            total_wall_points: 0,
            total_walls: 0,
            frame_times: Vec::new(),
            last_n_wall: 0,
            last_wall_planes: Vec::new(),
            last_wall_counts: Vec::new(),
            last_cloud: Vec::new(),
        }
    }
}

impl BenchStrategy for WallBenchCase {
    fn name(&self) -> &str { &self.name }

    fn run(&mut self, frame: &FrameData) -> Duration {
        let mut cloud = frame.non_ground().to_vec();
        let start = Instant::now();
        let (n_wall, wall_planes) = self.strategy.pick(&mut cloud);
        let elapsed = start.elapsed();

        let wall_counts = assign_wall_counts(&cloud[..n_wall], &wall_planes);

        let ms = elapsed.as_secs_f64() * 1000.0;
        self.total_ms += ms;
        self.frame_count += 1;
        self.total_wall_points += n_wall;
        self.total_walls += wall_planes.len();
        self.frame_times.push(ms);
        self.last_n_wall = n_wall;
        self.last_wall_planes = wall_planes;
        self.last_wall_counts = wall_counts;
        self.last_cloud = cloud;

        elapsed
    }

    fn write_frame(&mut self, recorder: &mut BenchRecorder, frame: &FrameData) {
        recorder.begin_frame(frame.frame_idx);

        let cloud = &self.last_cloud;
        let n_wall = self.last_n_wall;

        let wall_step = (n_wall / 2000).max(1);
        for i in (0..n_wall).step_by(wall_step) {
            recorder.write_point_cloud(&[cloud[i]], MAT_WALL, 1);
        }

        let remaining = cloud.len() - n_wall;
        let remain_step = (remaining / 3000).max(1);
        for i in (n_wall..cloud.len()).step_by(remain_step) {
            recorder.write_point_cloud(&[cloud[i]], MAT_REMAIN, 1);
        }

        let mut offset = 0usize;
        for (wi, (plane, &count)) in self.last_wall_planes.iter().zip(self.last_wall_counts.iter()).enumerate() {
            if count == 0 { continue; }
            let wall_pts: Vec<[f32; 3]> = cloud[offset..offset + count].to_vec();
            let box3d = Box3D::from_cloud_aabb(&wall_pts, 0.05);
            let tag = format!("wall{} n=({:.2},{:.2},{:.2}) d={:.2} {:.1}x{:.1}x{:.1} {}pts",
                wi, plane[0], plane[1], plane[2], plane[3],
                box3d.length, box3d.width, box3d.height, count);
            recorder.write_boxes(&[(box3d, tag)], MAT_BOX);
            offset += count;
        }

        let n = self.frame_count.max(1) as f64;
        let avg_ms = self.total_ms / n;
        let avg_walls = self.total_walls as f64 / n;
        let avg_wall_pts = self.total_wall_points as f64 / n;
        println!("[{}] 墙面={} 剩余={} | 累计 {:.1}面 {:.0}pts {:.0}ms/帧",
            self.name, n_wall, remaining, avg_walls, avg_wall_pts, avg_ms);

        recorder.end_frame();
    }

    fn summarize(&self) {
        let n = self.frame_count.max(1) as f64;
        let avg_ms = self.total_ms / n;
        let avg_walls = self.total_walls as f64 / n;
        let avg_wall_pts = self.total_wall_points as f64 / n;
        let status = if avg_ms > 100.0 { " [OVER 100ms]" } else { "" };
        println!("  {:<32} | 面 {:>5.1} | 墙点 {:>6.0} | {:>6.1}ms | {} 帧{}",
            self.name, avg_walls, avg_wall_pts, avg_ms, self.frame_count, status);
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

fn assign_wall_counts(wall_points: &[[f32; 3]], planes: &[[f32; 4]]) -> Vec<usize> {
    if planes.is_empty() { return Vec::new(); }
    let mut counts = vec![0usize; planes.len()];
    for p in wall_points {
        let mut best = 0;
        let mut best_dist = f32::MAX;
        for (i, plane) in planes.iter().enumerate() {
            let dist = (plane[0]*p[0] + plane[1]*p[1] + plane[2]*p[2] + plane[3]).abs();
            if dist < best_dist {
                best_dist = dist;
                best = i;
            }
        }
        counts[best] += 1;
    }
    counts
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();

    println!("=== 墙体提取策略对比测试 ===\n");

    let mut strategies: Vec<Box<dyn BenchStrategy>> = Vec::new();

    // ─── Top-Down Cluster ───
    for &cell_size in &[0.05, 0.10, 0.15] {
        for &min_density in &[3, 5, 8] {
            for &merge_dist in &[1, 2, 3] {
                let name = format!("td_c{:.2}_d{}_m{}",
                    cell_size, min_density, merge_dist);
                strategies.push(Box::new(WallBenchCase::new(
                    &name,
                    Box::new(TopDownCluster::with_params(cell_size, min_density, merge_dist)),
                )));
            }
        }
    }

    // ─── Top-Down + 2D 法线校验变体 ───
    for &ratio in &[0.15, 0.20, 0.30, 0.50] {
        let name = format!("td_c0.05_d5_m2_w{:.2}", ratio);
        strategies.push(Box::new(WallBenchCase::new(
            &name,
            Box::new(TopDownCluster::with_params(0.05, 5, 2).with_width_ratio(ratio)),
        )));
    }

    // ─── Quadtree 连通域 + 2D PCA ───
    for &cell_size in &[0.05, 0.10, 0.15] {
        for &min_pts in &[3, 5] {
            for &z_span in &[1.0, 1.5, 2.0] {
                let name = format!("qt_c{:.2}_p{}_z{:.1}", cell_size, min_pts, z_span);
                strategies.push(Box::new(WallBenchCase::new(
                    &name,
                    Box::new(QuadtreeWall::with_params(cell_size, min_pts, z_span)),
                )));
            }
        }
    }

    // ─── Quadtree + width_ratio 变体 ───
    for &ratio in &[0.15, 0.20, 0.30, 0.50] {
        let name = format!("qt_c0.10_p3_z1.5_w{:.2}", ratio);
        strategies.push(Box::new(WallBenchCase::new(
            &name,
            Box::new(QuadtreeWall::with_params(0.10, 3, 1.5).with_width_ratio(ratio)),
        )));
    }

    // ─── XY RANSAC 线检测 ───
    for &distance in &[0.05, 0.08, 0.10, 0.15] {
        for &iterations in &[50, 100] {
            let name = format!("xy_ransac_d{:.2}_i{}", distance, iterations);
            strategies.push(Box::new(WallBenchCase::new(
                &name,
                Box::new(XYRansacWall::with_params(distance, iterations, 30)),
            )));
        }
    }

    println!("共 {} 个策略\n", strategies.len());

    let harness = BenchHarness::new("./data/test", 10, "output/wall_bench");
    let mut preprocessor = WallPreprocessor::default();
    harness.run(&mut preprocessor, &mut strategies).await?;

    println!("\n提示：标记 [OVER 100ms] 的策略建议排除出对比");

    Ok(())
}
