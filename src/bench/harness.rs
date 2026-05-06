use std::time::Instant;

use crate::optional::data_loader::DataLoader;
use crate::swapl::global_swapl;
use crate::cloud::ground::create_ground_strategy;
use super::recorder::BenchRecorder;
use super::strategy::{BenchStrategy, FrameData};

const WARN_THRESHOLD_MS: f64 = 100.0;

/// 策略测试执行器。
///
/// 自动完成数据加载、逐帧迭代、预处理（每帧一次）、策略串行执行和结果输出。
/// 每个策略的结果写入独立的 .rdra 文件。
pub struct BenchHarness {
    data_path: String,
    frame_limit: usize,
    output_dir: String,
}

impl BenchHarness {
    pub fn new(data_path: &str, frame_limit: usize, output_dir: &str) -> Self {
        BenchHarness {
            data_path: data_path.to_string(),
            frame_limit,
            output_dir: output_dir.to_string(),
        }
    }

    /// 运行所有策略的 benchmark。
    ///
    /// 每帧预处理（create_ground_strategy）只执行一次，所有策略串行执行。
    pub async fn run(&self, strategies: &mut Vec<Box<dyn BenchStrategy>>) -> Result<(), Box<dyn std::error::Error>> {
        let n_strategies = strategies.len();
        println!("=== 策略测试 ({} 策略, {} 帧) ===", n_strategies, self.frame_limit);

        let mut data_loader = DataLoader::new(self.data_path.clone());
        data_loader.set_frame_limit(self.frame_limit);
        data_loader.load().await?;

        std::fs::create_dir_all(&self.output_dir)?;

        let mut recorders: Vec<BenchRecorder> = (0..n_strategies)
            .map(|_| BenchRecorder::new())
            .collect();

        let mut frame_idx = 0usize;
        let total_start = Instant::now();

        while data_loader.load_next().await? {
            let cloud: Vec<[f32; 3]> = {
                let swapl = global_swapl();
                let mut stream = swapl.clouds.lock().await;
                match stream.read() {
                    Some(data) => data,
                    None => continue,
                }
            };

            if cloud.is_empty() {
                frame_idx += 1;
                continue;
            }

            // 预处理：每帧执行一次默认地面提取
            let mut preprocess_cloud = cloud.clone();
            let mut ground_strategy = create_ground_strategy();
            let (n_ground, _grounds, _plane_eq) = ground_strategy.pick(&mut preprocess_cloud);
            let non_ground = &preprocess_cloud[n_ground..];

            let frame = FrameData {
                cloud: &cloud,
                preprocessed: &preprocess_cloud,
                non_ground,
                frame_idx,
            };

            // 串行执行所有策略
            for (i, strategy) in strategies.iter_mut().enumerate() {
                let elapsed = strategy.run(&frame);
                let ms = elapsed.as_secs_f64() * 1000.0;
                if ms > WARN_THRESHOLD_MS {
                    println!("  [WARN] {} 第 {} 帧耗时 {:.1}ms (> {:.0}ms)",
                        strategy.name(), frame_idx, ms, WARN_THRESHOLD_MS);
                }
                strategy.write_frame(&mut recorders[i], &frame);
            }

            frame_idx += 1;
            if frame_idx % 10 == 0 {
                println!("已处理 {} 帧...", frame_idx);
            }
        }

        let total_elapsed = total_start.elapsed();
        println!("\n共 {} 帧，总耗时: {:.1}s\n", frame_idx, total_elapsed.as_secs_f64());

        for (i, strategy) in strategies.iter().enumerate() {
            strategy.summarize();

            let safe_label = strategy.name().replace(['=', '.', ' '], "_");
            let path = format!("{}/{}.rdra", self.output_dir, safe_label);
            recorders[i].save(&path)?;
            println!("  [{}] {} → {}", i + 1, strategy.name(), path);
        }

        println!("\n共 {} 个 .rdra 文件保存到 {}", n_strategies, self.output_dir);
        Ok(())
    }
}
