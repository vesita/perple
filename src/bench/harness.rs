use std::time::{Instant, SystemTime, UNIX_EPOCH};

use crate::optional::data_loader::DataLoader;
use crate::swapl::global_swapl;
use super::recorder::BenchRecorder;
use super::strategy::{BenchStrategy, BenchStats, FrameData, Preprocessor};
use super::config::{self, StrategyFamily, load_task_strategies, param_dirname, compute_median, StatsInfo};
use super::cli::BenchMode;

const WARN_THRESHOLD_MS: f64 = 100.0;

/// 策略测试执行器。
pub struct BenchHarness {
    data_path: String,
    frame_limit: usize,
}

impl BenchHarness {
    pub fn new(data_path: &str, frame_limit: usize) -> Self {
        BenchHarness { data_path: data_path.to_string(), frame_limit }
    }

    /// 运行所有策略，返回统计信息。
    ///
    /// `recorders` 必须与 `strategies` 长度相同，每个 recorder 对应一个策略的输出目标。
    /// 数据在每帧处理时通过 `end_frame()` 直接写入 SQLite，无需事后保存。
    pub async fn run(
        &self,
        preprocessor: &mut dyn Preprocessor,
        strategies: &mut Vec<Box<dyn BenchStrategy>>,
        recorders: &mut [BenchRecorder],
    ) -> Result<Vec<BenchStats>, Box<dyn std::error::Error>> {
        let n_strategies = strategies.len();
        assert_eq!(recorders.len(), n_strategies,
            "recorders.len() ({}) 必须等于 strategies.len() ({})", recorders.len(), n_strategies);

        println!("=== 策略测试 ({} 策略, {} 帧, 预处理: {}) ===",
            n_strategies, self.frame_limit, preprocessor.name());

        let mut data_loader = DataLoader::new(self.data_path.clone());
        data_loader.set_frame_limit(self.frame_limit);
        data_loader.load().await?;

        let mut frame_idx = 0usize;
        let total_start = Instant::now();
        let mut skipped: Vec<bool> = vec![false; n_strategies];
        let mut skip_count = 0;

        while data_loader.load_next().await? {
            let cloud: Vec<[f32; 3]> = {
                let swapl = global_swapl();
                let mut stream = swapl.clouds.lock().unwrap();
                match stream.read() {
                    Some(data) => data,
                    None => continue,
                }
            };

            if cloud.is_empty() { frame_idx += 1; continue; }

            let preprocessed = preprocessor.preprocess(&cloud);
            let frame = FrameData { cloud: &cloud, preprocessed: &preprocessed, frame_idx };

            for (i, strategy) in strategies.iter_mut().enumerate() {
                if skipped[i] { continue; }

                let elapsed = strategy.run(&frame);
                let ms = elapsed.as_secs_f64() * 1000.0;
                if ms > WARN_THRESHOLD_MS {
                    if frame_idx == 0 {
                        println!("  >>> 跳过 {} ({:.1}ms > {:.0}ms)", strategy.name(), ms, WARN_THRESHOLD_MS);
                        skipped[i] = true;
                        skip_count += 1;
                    } else {
                        println!("  [WARN] {} 第 {} 帧耗时 {:.1}ms (> {:.0}ms)",
                            strategy.name(), frame_idx, ms, WARN_THRESHOLD_MS);
                    }
                }
                strategy.write_frame(&mut recorders[i], &frame);
            }

            frame_idx += 1;
            if skip_count > 0 && frame_idx == 1 {
                println!("  >>> 跳过 {} 个慢策略，剩余 {} 个\n", skip_count, n_strategies - skip_count);
            }
            if frame_idx % 10 == 0 { println!("已处理 {} 帧...", frame_idx); }
        }

        let total_elapsed = total_start.elapsed();
        println!("\n共 {} 帧，总耗时: {:.1}s\n", frame_idx, total_elapsed.as_secs_f64());

        for s in strategies.iter() { s.summarize(); }

        let stats: Vec<_> = strategies.iter().map(|s| s.stats()).collect();
        Ok(stats)
    }
}

// ── TOML 驱动的全流程编排 ─────────────────────────────────

/// 策略构建器：给定策略类型和参数表，返回 BenchStrategy trait object。
pub trait StrategyBuilder {
    fn build(&self, strategy_type: &str, params: &toml::Table) -> Box<dyn BenchStrategy>;
}

impl<F> StrategyBuilder for F
where
    F: Fn(&str, &toml::Table) -> Box<dyn BenchStrategy>,
{
    fn build(&self, strategy_type: &str, params: &toml::Table) -> Box<dyn BenchStrategy> {
        (*self)(strategy_type, params)
    }
}

/// TOML 驱动的完整 bench 运行：加载配置 → 构建策略 → 运行 → 输出 → 更新统计。
///
/// `task` 对应 `config/bench/{task}/` 目录下的 TOML 文件集合。
/// `builder` 负责将 (strategy_type, params) 转换为具体的 BenchStrategy。
///
/// 快速测试与全量测试共享速度过滤器：上次最慢 >100ms 的策略会被跳过。
/// 帧数取所有策略族中的最大值，确保每个 TOML 的 frames 设置生效。
pub async fn run_toml_bench(
    task: &str,
    data_path: &str,
    mode: BenchMode,
    preprocessor: &mut dyn Preprocessor,
    builder: &dyn StrategyBuilder,
) -> Result<(), Box<dyn std::error::Error>> {
    let families = load_task_strategies(task);
    if families.is_empty() {
        eprintln!("WARN: no strategy TOML files for task '{}'", task);
        return Ok(());
    }

    // 清理该任务下所有旧数据，避免已移除策略的残留
    let task_out_dir = format!("output/bench/{}", task);
    let _ = std::fs::remove_dir_all(&task_out_dir);
    std::fs::create_dir_all(&task_out_dir)?;

    let (mode_label, expected_frames) = match mode {
        BenchMode::Quick => ("quick", families[0].quick.frames),
        BenchMode::Full => ("full", families.iter().map(|f| f.full.frames).max().unwrap_or(1)),
        _ => unreachable!(),
    };

    // 跳过上次最慢 >100ms 的策略（快速测试也维护此标签，供全量测试参考）
    let filtered: Vec<&StrategyFamily> = families.iter().filter(|f| {
        if let Some(ref st) = f.stats {
            if st.slowest_ms > 100.0 && st.last_run.is_some() {
                println!("  跳过 {} (上次最慢 {:.0}ms > 100ms)", f.name, st.slowest_ms);
                return false;
            }
        }
        true
    }).collect();

    if filtered.is_empty() { println!("无可用策略（全部被跳过）"); return Ok(()); }

    // 展开为平坦的策略列表
    fn family_params<'a>(mode: BenchMode, f: &'a StrategyFamily) -> &'a [toml::Table] {
        match mode { BenchMode::Quick => &f.quick.params, BenchMode::Full => &f.full.params, _ => unreachable!() }
    }

    let mut strategies: Vec<Box<dyn BenchStrategy>> = Vec::new();
    let mut output_dirs: Vec<String> = Vec::new();
    let mut filenames: Vec<String> = Vec::new();
    let mut entry_family: Vec<usize> = Vec::new();
    let mut all_params: Vec<toml::Table> = Vec::new();

    for (fi, family) in filtered.iter().enumerate() {
        for params in family_params(mode, family) {
            let s = builder.build(&family.strategy_type, params);
            let dn = param_dirname(&family.strategy_type, params);
            strategies.push(s);
            output_dirs.push(format!("output/bench/{}/{}", task, family.strategy_type));
            filenames.push(dn);
            entry_family.push(fi);
            all_params.push(params.clone());
        }
    }

    if strategies.is_empty() { println!("无可用策略组合"); return Ok(()); }

    println!("\n=== {}.{} ({} 组合, {} 帧, 预处理: {}) ===\n", task, mode_label, strategies.len(), expected_frames, preprocessor.name());

    // 创建 recorders（每个策略一个 SQLite 数据库文件）
    // 删除旧 DB 文件确保 bench 从干净状态开始，避免跨版本数据累积
    let mut recorders: Vec<BenchRecorder> = Vec::with_capacity(strategies.len());
    for i in 0..strategies.len() {
        let dir = &output_dirs[i];
        let path = format!("{}/{}.db", dir, filenames[i]);
        std::fs::create_dir_all(dir)?;
        let _ = std::fs::remove_file(&path);
        recorders.push(BenchRecorder::new(&path)?);
    }

    // 执行（直接写入 SQLite，无需片段模式）
    let harness = BenchHarness::new(data_path, expected_frames);
    let all_stats = harness.run(preprocessor, &mut strategies, &mut recorders).await?;

    // VACUUM 压缩各数据库文件
    for rec in &recorders {
        rec.save()?;
    }

    // 聚合：按策略目录分组写 info.json
    let extra_all: Vec<Vec<(String, f64)>> = strategies.iter().map(|s| s.extra_metrics()).collect();

    let mut group_map: std::collections::BTreeMap<String, Vec<usize>> = std::collections::BTreeMap::new();
    for (i, dir) in output_dirs.iter().enumerate() {
        group_map.entry(dir.clone()).or_default().push(i);
    }

    for (dir, indices) in &group_map {
        let all_times: Vec<f64> = indices.iter().flat_map(|&i| all_stats[i].frame_times.iter().copied()).collect();
        let fastest = all_times.iter().cloned().fold(f64::MAX, f64::min);
        let slowest = all_times.iter().cloned().fold(f64::MIN, f64::max);
        let avg_all = if all_times.is_empty() { 0.0 } else { all_times.iter().sum::<f64>() / all_times.len() as f64 };
        let median = compute_median(all_times);

        let strategy_name = dir.rsplit('/').next().unwrap_or("");
        let results: Vec<serde_json::Value> = indices.iter().map(|&i| {
            let st = &all_stats[i];
            let extra: serde_json::Map<String, serde_json::Value> = extra_all[i].iter()
                .map(|(k, v)| (k.clone(), serde_json::Value::Number(serde_json::Number::from_f64(*v).unwrap_or(serde_json::Number::from_f64(0.0).unwrap()))))
                .collect();
            let pv = params_to_json(&all_params[i]);
            serde_json::json!({
                "params": pv,
                "frame_count": st.frame_count,
                "total_ms": st.total_ms,
                "avg_ms": if st.frame_count > 0 { st.total_ms / st.frame_count as f64 } else { 0.0 },
                "extra": serde_json::Value::Object(extra),
            })
        }).collect();

        let info = serde_json::json!({
            "strategy": strategy_name,
            "mode": mode_label,
            "stats": {
                "fastest_ms": fastest,
                "slowest_ms": slowest,
                "avg_ms": avg_all,
                "median_ms": median,
            },
            "results": results,
        });

        std::fs::write(format!("{}/info.json", dir), serde_json::to_string_pretty(&info)?)?;
    }

    // 更新 TOML [stats]
    let timestamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();
    for (fi, family) in filtered.iter().enumerate() {
        let mut family_times: Vec<f64> = Vec::new();
        for (i, st) in all_stats.iter().enumerate() {
            if entry_family[i] == fi { family_times.extend(&st.frame_times); }
        }
        if family_times.is_empty() { continue; }

        let fastest = family_times.iter().cloned().fold(f64::MAX, f64::min);
        let slowest = family_times.iter().cloned().fold(f64::MIN, f64::max);
        let avg = family_times.iter().sum::<f64>() / family_times.len() as f64;
        let median = compute_median(family_times);

        let stats_info = StatsInfo {
            fastest_ms: fastest,
            slowest_ms: slowest,
            avg_ms: avg,
            median_ms: median,
            last_run: Some(timestamp.to_string()),
        };

        let toml_path = format!("config/bench/{}/{}.toml", task, family.strategy_type);
        if let Err(e) = config::update_strategy_stats(&toml_path, &stats_info) {
            eprintln!("WARN: 更新 TOML 统计失败 {}: {}", toml_path, e);
        }
    }

    // 打印跨策略汇总
    println!("\n=== {} {} 汇总 ===", task, mode_label);
    for (fi, family) in filtered.iter().enumerate() {
        let times: Vec<f64> = all_stats.iter().enumerate()
            .filter(|(i, _)| entry_family[*i] == fi)
            .flat_map(|(_, s)| s.frame_times.iter().copied())
            .collect();
        let avg = if times.is_empty() { 0.0 } else { times.iter().sum::<f64>() / times.len() as f64 };
        println!("  {:<20} {:>6.1}ms ({} 参数组合)", family.strategy_type, avg, family_params(mode, family).len());
    }

    Ok(())
}

/// 将 toml::Table 转为 serde_json::Value，确保数值类型正确。
fn params_to_json(table: &toml::Table) -> serde_json::Value {
    let mut map = serde_json::Map::new();
    for (k, v) in table {
        let jv = match v {
            toml::Value::Float(f) => serde_json::Value::Number(serde_json::Number::from_f64(*f).unwrap_or(serde_json::Number::from_f64(0.0).unwrap())),
            toml::Value::Integer(i) => serde_json::Value::Number(serde_json::Number::from(*i)),
            toml::Value::String(s) => serde_json::Value::String(s.clone()),
            toml::Value::Boolean(b) => serde_json::Value::Bool(*b),
            _ => serde_json::Value::String(format!("{:?}", v)),
        };
        map.insert(k.clone(), jv);
    }
    serde_json::Value::Object(map)
}
