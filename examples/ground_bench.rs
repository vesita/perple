use std::time::{Duration, Instant};

use perple::bench::{BenchStrategy, BenchHarness, BenchRecorder, FrameData};
use perple::cloud::ground::*;
use perple::utils::boxes::Box3D;

struct GroundBenchCase {
    name: String,
    strategy: Box<dyn GroundPickStrategy>,
    total_us: u128,
    frame_count: usize,
    last_n_ground: usize,
}

impl GroundBenchCase {
    fn new(name: &str, strategy: Box<dyn GroundPickStrategy>) -> Self {
        Self {
            name: name.to_string(),
            strategy,
            total_us: 0,
            frame_count: 0,
            last_n_ground: 0,
        }
    }
}

impl BenchStrategy for GroundBenchCase {
    fn name(&self) -> &str { &self.name }

    fn run(&mut self, frame: &FrameData) -> Duration {
        let mut cloud = frame.cloud.to_vec();
        let start = Instant::now();
        let (n_ground, _, _) = self.strategy.pick(&mut cloud);
        let elapsed = start.elapsed();
        self.total_us += elapsed.as_micros();
        self.frame_count += 1;
        self.last_n_ground = n_ground;
        elapsed
    }

    fn write_frame(&mut self, recorder: &mut BenchRecorder, frame: &FrameData) {
        let mut cloud = frame.cloud.to_vec();
        let (n_ground, _, _) = self.strategy.pick(&mut cloud);

        recorder.begin_frame(frame.frame_idx);

        let cloud_step = (cloud.len() / 5000).max(1);
        for i in (0..cloud.len()).step_by(cloud_step) {
            let p = cloud[i];
            let mat = if i < n_ground { "green" } else { "white" };
            recorder.write_point_cloud(&[p], mat, 1);
        }

        if n_ground > 0 {
            let mut ground_box = Box3D::empty_box();
            ground_box.cloud2box(&cloud[..n_ground].to_vec());
            let avg_us = self.total_us / self.frame_count.max(1) as u128;
            let tag = format!("{} | {}pts | {}μs/帧", self.name, n_ground, avg_us);
            recorder.write_boxes(&[(ground_box, tag)], "glass");
        }

        recorder.end_frame();
    }

    fn summarize(&self) {
        let n = self.frame_count.max(1) as f64;
        let avg_us = self.total_us as f64 / n;
        let avg_ms = avg_us / 1000.0;
        let status = if avg_ms > 100.0 { " [OVER 100ms]" } else { "" };
        println!("  {:<28} | 平均 {:>7.0}μs ({:>5.1}ms) | {} 帧{}",
            self.name, avg_us, avg_ms, self.frame_count, status);
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();

    println!("=== 地面检测策略对比测试（串行） ===\n");

    let mut strategies: Vec<Box<dyn BenchStrategy>> = Vec::new();

    // 策略 1：Z-直方图 + expand
    for &expand in &[0.05, 0.10, 0.15, 0.20, 0.30] {
        let name = format!("histogram_ex={:.2}", expand);
        strategies.push(Box::new(GroundBenchCase::new(
            &name,
            Box::new(HistogramExpand::with_expand(expand)),
        )));
    }

    // 策略 2：峰下扫 + 上扩
    for &threshold in &[0.05, 0.10, 0.15, 0.20] {
        for &expand in &[0.05, 0.10, 0.20] {
            let name = format!("peak_sd={:.2}_ex={:.2}", threshold, expand);
            strategies.push(Box::new(GroundBenchCase::new(
                &name,
                Box::new(PeakDownExpandUp::with_params(threshold, expand)),
            )));
        }
    }

    // 策略 3：RANSAC
    for &distance in &[0.3, 0.5] {
        for &iterations in &[100, 200] {
            let name = format!("ransac_d={:.1}_i={}", distance, iterations);
            strategies.push(Box::new(GroundBenchCase::new(
                &name,
                Box::new(RansacGround::with_params(distance, iterations)),
            )));
        }
    }

    // 策略 4：种子+生长
    for &expand in &[0.10, 0.20] {
        for &distance in &[0.3, 0.5] {
            for &iterations in &[50, 100] {
                let name = format!("histoseed_ex={:.2}_d={:.1}_i={}", expand, distance, iterations);
                strategies.push(Box::new(GroundBenchCase::new(
                    &name,
                    Box::new(HistoseedPlane::with_params(expand, distance, iterations)),
                )));
            }
        }
    }

    // 策略 5：GPF
    for &n_lpr in &[100, 200] {
        for &th_dist in &[0.2, 0.3, 0.5] {
            let name = format!("gpf_nlpr={}_d={:.1}", n_lpr, th_dist);
            strategies.push(Box::new(GroundBenchCase::new(
                &name,
                Box::new(GpfGround::with_params(n_lpr, 0.5, th_dist)),
            )));
        }
    }

    println!("共 {} 个策略\n", strategies.len());

    let harness = BenchHarness::new("./data/test", 5, "output/ground_bench");
    harness.run(&mut strategies).await?;

    Ok(())
}
