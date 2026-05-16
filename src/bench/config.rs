use serde::{Deserialize, Serialize};
use std::path::Path;

/// 单个策略族定义（对应一个 TOML 文件）。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyFamily {
    pub name: String,
    #[serde(rename = "type")]
    pub strategy_type: String,
    pub preprocessor: String,
    pub quick: ModeConfig,
    pub full: ModeConfig,
    pub stats: Option<StatsInfo>,
}

/// 一个测试模式下的参数网格（快速/全量）。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModeConfig {
    pub frames: usize,
    pub params: Vec<toml::Table>,
}

/// 持久化速度统计。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatsInfo {
    pub fastest_ms: f64,
    pub slowest_ms: f64,
    pub avg_ms: f64,
    pub median_ms: f64,
    pub last_run: Option<String>,
}

/// 加载某任务下所有策略 TOML 文件。
pub fn load_task_strategies(task: &str) -> Vec<StrategyFamily> {
    let dir = format!("config/bench/{}", task);
    let mut families = Vec::new();
    let dir_path = Path::new(&dir);
    if !dir_path.is_dir() {
        eprintln!("WARN: bench config dir not found: {}", dir);
        return families;
    }
    let mut entries: Vec<_> = std::fs::read_dir(dir_path)
        .unwrap()
        .filter_map(|e| e.ok())
        .collect();
    entries.sort_by_key(|e| e.path());
    for entry in entries {
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) == Some("toml") {
            let toml_str = match std::fs::read_to_string(&path) {
                Ok(s) => s,
                Err(e) => { eprintln!("WARN: cannot read {}: {}", path.display(), e); continue; }
            };
            match toml::from_str::<StrategyFamily>(&toml_str) {
                Ok(family) => families.push(family),
                Err(e) => eprintln!("WARN: failed to parse {}: {}", path.display(), e),
            }
        }
    }
    families
}

/// 更新策略 TOML 文件中的 [stats] 段（文本级替换，保留原格式）。
pub fn update_strategy_stats(path: &str, stats: &StatsInfo) -> Result<(), String> {
    let content = std::fs::read_to_string(path).map_err(|e| e.to_string())?;

    let stats_text = format!(
        "[stats]\nfastest_ms = {}\nslowest_ms = {}\navg_ms = {}\nmedian_ms = {}\nlast_run = \"{}\"\n",
        stats.fastest_ms, stats.slowest_ms, stats.avg_ms, stats.median_ms,
        stats.last_run.as_deref().unwrap_or(""),
    );

    let new_content = if let Some(pos) = content.find("[stats]") {
        format!("{}{}", &content[..pos], stats_text)
    } else {
        format!("{}\n{}", content.trim_end(), stats_text)
    };

    std::fs::write(path, new_content).map_err(|e| e.to_string())
}

/// 为参数组合生成短文件名（不含扩展名）。
pub fn param_dirname(strategy_type: &str, params: &toml::Table) -> String {
    match strategy_type {
        "histogram" => format!("ex{:.2}", float(params, "expand")),
        "peak_scan" => format!("t{:.2}_e{:.2}", float(params, "threshold"), float(params, "expand")),
        "histoseed" => format!("e{:.2}_d{:.1}_i{}", float(params, "expand"), float(params, "distance"), int(params, "iterations")),
        "gpf" => format!("l{}_s{:.1}_d{:.2}", int(params, "n_lpr"), float(params, "th_seed"), float(params, "th_dist")),
        "xy_grid_dbscan" | "xy_grid_dbscan_grid" | "lvdot_grid" | "lvdot" => format!("e{:.2}_m{}", float(params, "eps"), int(params, "min_pts")),
        "xy_dbscan" => format!("e{:.2}_m{}", float(params, "eps"), int(params, "min_pts")),
        "prune_qt" | "lvdot_qt" => format!("o{}_e{:.2}_m{}", int(params, "min_occ"), float(params, "eps"), int(params, "min_pts")),
        "range_image" => format!("a{:.1}_e{:.1}_t{:.1}_m{}",
            float(params, "azimuth"), float(params, "elevation"),
            float(params, "threshold"), int(params, "min_pts")),
        "dbscan_qt" | "dbscan_adaptive" => format!("p{:.2}_s{:.2}_m{}_v{:.2}",
            float(params, "patience"), float(params, "slope"),
            int(params, "min_pts"), float(params, "voxel_size")),
        "dbscan_grid" => format!("e{:.2}_m{}", float(params, "eps"), int(params, "min_pts")),
        "cc_pca_qt" => format!("p{}_m{}_w{:.2}",
            int(params, "max_pts_per_node"), int(params, "min_points"), float(params, "width_ratio")),
        "cc_pca_grid" => format!("c{:.2}_m{}_w{:.2}",
            float(params, "cell_size"), int(params, "min_pts"), float(params, "width_ratio")),
        "cc_l2_grid" => format!("c{:.2}_m{}_r{:.2}",
            float(params, "cell_size"), int(params, "min_pts"), float(params, "max_rms")),
        "cc_l2_qt" => format!("p{}_m{}_r{:.2}",
            int(params, "max_pts_per_node"), int(params, "min_points"), float(params, "max_rms")),
        "ransac_l2_grid" => {
            let base = format!("d{:.2}_i{}", float(params, "distance"), int(params, "iterations"));
            if params.contains_key("min_extent") {
                format!("{}_e{:.1}", base, float(params, "min_extent"))
            } else { base }
        },
        "ransac_l2_qt" => format!("d{:.2}_i{}", float(params, "distance"), int(params, "iterations")),
        "bev_hough" => {
            let base = format!("d{:.2}_m{}", float(params, "distance"), int(params, "min_wall_pts"));
            if params.contains_key("hough_threshold") {
                format!("{}_ht{:.2}", base, float(params, "hough_threshold"))
            } else { base }
        },
        "bev_edlines" => format!("d{:.2}_m{}", float(params, "distance"), int(params, "min_wall_pts")),
        "nrm_pca_grid" => format!("c{:.2}_z{:.2}", float(params, "cell_size"), float(params, "normal_threshold")),
        "nrm_l2_grid" => format!("c{:.2}_r{:.2}", float(params, "cell_size"), float(params, "max_rms")),
        "nrm_pca_qt" => format!("p{}_z{:.2}", int(params, "max_pts_per_node"), float(params, "normal_threshold")),
        "nrm_l2_qt" => format!("p{}_r{:.2}", int(params, "max_pts_per_node"), float(params, "max_rms")),
        "seq_pca_grid" => format!("d{:.2}_t{:.1}_w{}",
            float(params, "distance"), float(params, "normal_threshold"), int(params, "max_walls")),
        "seq_l2_grid" => format!("d{:.2}_m{}",
            float(params, "distance"), int(params, "min_wall_pts")),
        "adapt_pca_grid" => format!("be{:.3}_s{:.3}_m{}",
            float(params, "base_eps"), float(params, "scale_factor"), int(params, "min_pts")),
        "dbscan_pca_qt" => format!("e{:.2}_m{}_z{:.1}",
            float(params, "eps"), int(params, "min_pts"), float(params, "min_z_span")),
        "dbscan_l2_qt" => format!("e{:.2}_m{}_z{:.1}",
            float(params, "eps"), int(params, "min_pts"), float(params, "min_z_span")),
        "dbscan_l2_qt_dif" => format!("p{}_m{}_z{:.1}",
            int(params, "max_pts_per_node"), int(params, "min_points"), float(params, "min_z_span")),
        "adapt_l2_grid" => format!("be{:.3}_s{:.3}_m{}",
            float(params, "base_eps"), float(params, "scale_factor"), int(params, "min_pts")),
        "radius_outlier" => format!("r{:.2}_m{}",
            float(params, "radius"), int(params, "min_pts")),
        "cc_grid" | "cc" => format!("c{:.2}_m{}_mr{}",
            float(params, "cell_size"), int(params, "min_pts"), int(params, "merge_dist")),
        "ransac" => format!("d{:.2}_i{}_m{}",
            float(params, "distance"), int(params, "iterations"), int(params, "min_pts")),
        "seq" => format!("d{:.2}_m{}_w{}",
            float(params, "distance"), int(params, "min_pts"), int(params, "max_walls")),
        _ => {
            let parts: Vec<String> = params.iter()
                .map(|(k, v)| format!("{}_{}", k, fmt_val(v)))
                .collect();
            parts.join("_")
        }
    }
}

/// 计算有序序列的中位数（输入会被排序）。
pub fn compute_median(mut times: Vec<f64>) -> f64 {
    times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = times.len();
    if n == 0 { return 0.0; }
    if n % 2 == 0 { (times[n / 2 - 1] + times[n / 2]) / 2.0 } else { times[n / 2] }
}

// ── helpers ──────────────────────────────────────────────

fn float(t: &toml::Table, key: &str) -> f64 {
    t.get(key).and_then(|v| v.as_float()).unwrap_or(0.0)
}

fn int(t: &toml::Table, key: &str) -> i64 {
    t.get(key).and_then(|v| v.as_integer()).unwrap_or(0)
}

/// 从 TOML 表中提取 f32 参数（bench 示例通用）。
pub fn get_f32(t: &toml::Table, key: &str) -> f32 { float(t, key) as f32 }

/// 从 TOML 表中提取 i64 参数（bench 示例通用）。
pub fn get_i64(t: &toml::Table, key: &str) -> i64 { int(t, key) }

fn fmt_val(v: &toml::Value) -> String {
    match v {
        toml::Value::Float(f) => format!("{:.2}", f),
        toml::Value::Integer(i) => i.to_string(),
        toml::Value::String(s) => s.clone(),
        toml::Value::Boolean(b) => b.to_string(),
        _ => format!("{:?}", v),
    }
}
