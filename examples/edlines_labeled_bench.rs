//! 基于标注数据的 EDLines 对比评估
//!
//! 固定管线：地面检测(PeakScan) → 墙体检测(两种策略) → 后聚类(连通域)
//! 只换墙体策略，对比最终检测精度。
//!
//! 匹配方式：中心距离匹配
//!
//! 用法:
//!   cargo run --release --example edlines_labeled_bench
//!   cargo run --release --example edlines_labeled_bench -- --frames 50
//!   cargo run --release --example edlines_labeled_bench -- --center-dist 0.5

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::time::Instant;

use perple::cloud::ground::{GroundPickStrategy, PeakScan};
use perple::cloud::wall::{BevEdLines, EdLinesRef, WallPickStrategy, cluster_obstacles_with_indices};
use perple::swapl::global_swapl;
use perple::tracker::hungarian::hungarian;
use perple::utils::boxes::Box3D;

use serde::{Deserialize, Serialize};

// ═══════════════════════════════════════════════════════════════════════════════
// Label types
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Deserialize)]
struct LabelItem {
    obj_type: String,
    psr: LabelPsr,
}

#[derive(Debug, Clone, Deserialize)]
struct LabelPsr {
    position: LabelVec3,
    scale: LabelVec3,
    rotation: LabelVec3,
}

#[derive(Debug, Clone, Deserialize)]
struct LabelVec3 {
    x: f32, y: f32, z: f32,
}

impl LabelItem {
    fn to_box3d(&self) -> Box3D {
        let p = &self.psr;
        Box3D::from_position_and_angles(
            p.position.x, p.position.y, p.position.z,
            0.0, 0.0, p.rotation.z,
            p.scale.x, p.scale.y, p.scale.z,
        )
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Statistics
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Default, Serialize)]
struct ClassStats {
    tp: usize,
    fp: usize,
    #[serde(rename = "fn")]
    fn_: usize,
}

impl ClassStats {
    fn precision(&self) -> f64 {
        if self.tp + self.fp == 0 { 0.0 } else { self.tp as f64 / (self.tp + self.fp) as f64 }
    }
    fn recall(&self) -> f64 {
        if self.tp + self.fn_ == 0 { 0.0 } else { self.tp as f64 / (self.tp + self.fn_) as f64 }
    }
    fn f1(&self) -> f64 {
        let p = self.precision();
        let r = self.recall();
        if p + r == 0.0 { 0.0 } else { 2.0 * p * r / (p + r) }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Matching
// ═══════════════════════════════════════════════════════════════════════════════

/// 用匈牙利最优指派进行中心距离匹配，返回 (tp, fp, fn)
fn match_boxes(
    det_boxes: &[Box3D],
    gt_boxes: &[Box3D],
    center_dist_threshold: f32,
    hungarian_buf: &mut Vec<Vec<f64>>,
) -> (usize, usize, usize) {
    let n_det = det_boxes.len();
    let n_gt = gt_boxes.len();

    if n_det == 0 { return (0, 0, n_gt); }
    if n_gt == 0 { return (0, n_det, 0); }

    // 代价矩阵：距离在阈值内为实际距离，否则 f64::MAX
    let mut cost = vec![vec![f64::MAX; n_gt]; n_det];
    for (i, det) in det_boxes.iter().enumerate() {
        let dc = det.center();
        for (j, gt) in gt_boxes.iter().enumerate() {
            let gc = gt.center();
            let dist = ((dc.x - gc.x).powi(2) + (dc.y - gc.y).powi(2)).sqrt();
            if dist <= center_dist_threshold {
                cost[i][j] = dist as f64;
            }
        }
    }

    let assignment = hungarian(&cost, hungarian_buf);
    let mut matched_gt = HashSet::new();
    let mut tp = 0usize;

    for (i, &j) in assignment.iter().enumerate() {
        if j < n_gt && cost[i][j] < f64::MAX / 2.0 {
            matched_gt.insert(j);
            tp += 1;
        }
    }

    (tp, n_det - tp, n_gt - matched_gt.len())
}

// ═══════════════════════════════════════════════════════════════════════════════
// 加载标注 JSON
// ═══════════════════════════════════════════════════════════════════════════════

fn load_labels(label_dir: &str) -> Vec<Vec<LabelItem>> {
    let mut entries: Vec<_> = std::fs::read_dir(label_dir)
        .expect("无法读取 label 目录")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|s| s == "json"))
        .collect();
    entries.sort_by_key(|e| e.file_name());

    let mut all = Vec::with_capacity(entries.len());
    for entry in &entries {
        let content = std::fs::read_to_string(entry.path())
            .unwrap_or_else(|_| panic!("无法读取: {:?}", entry.path()));
        let items: Vec<LabelItem> = serde_json::from_str(&content)
            .unwrap_or_else(|_| panic!("解析失败: {:?}", entry.path()));
        all.push(items);
    }
    all
}

fn filter_gt(items: &[LabelItem], max_range: f32) -> Vec<Box3D> {
    items.iter()
        .filter(|it| it.obj_type == "Pedestrian")
        .filter(|it| {
            let d = (it.psr.position.x.powi(2) + it.psr.position.y.powi(2)).sqrt();
            d <= max_range
        })
        .map(|it| it.to_box3d())
        .collect()
}

// ═══════════════════════════════════════════════════════════════════════════════
// Output types
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Serialize)]
struct FrameDetail {
    frame_idx: usize,
    n_gt: usize,
    bev_boxes: usize, ref_boxes: usize,
    bev_tp: usize, bev_fp: usize, bev_fn: usize,
    ref_tp: usize, ref_fp: usize, ref_fn: usize,
    bev_wall: usize, ref_wall: usize,
    bev_time_ms: f64, ref_time_ms: f64,
}

#[derive(Serialize)]
struct EvalOutput {
    config: HashMap<String, f64>,
    frames: Vec<FrameDetail>,
    bev_total: ClassStats,
    ref_total: ClassStats,
}

// ═══════════════════════════════════════════════════════════════════════════════
// 主流程
// ═══════════════════════════════════════════════════════════════════════════════

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Warn)
        .init();

    let args: Vec<String> = std::env::args().collect();
    let n_frames: Option<usize> = args.iter()
        .position(|a| a == "--frames")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok());
    let center_dist: f32 = args.iter()
        .position(|a| a == "--center-dist")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.5);

    // ─── 加载标注 ─────────────────────────────────────────────────────────
    let label_dir = "data/labeled/label";
    let all_labels = load_labels(label_dir);
    let n_label_frames = all_labels.len();
    let max_range = 10.0;

    // ─── 加载 LiDAR 数据 ─────────────────────────────────────────────────
    use perple::optional::data_loader::DataLoader;
    let mut data_loader = DataLoader::new_independent(
        "data/labeled/camera/image".to_string(),
        "data/labeled/lidar".to_string(),
    );
    data_loader.load().await?;
    let n_avail = data_loader.frame_count().min(n_label_frames);
    let n_total = n_frames.map(|n| n.min(n_avail)).unwrap_or(n_avail);

    println!("\n═══ EDLines 标注对比 (中心距 ≤ {:.1}m, 范围 ≤ {}m, {} 帧) ═══\n",
        center_dist, max_range, n_total);

    // ─── 策略 ─────────────────────────────────────────────────────────────
    let mut ground = PeakScan::new();
    let mut bev = BevEdLines::new()
        .with_min_extent(0.5)
        .with_gaussian_blur(0.8)
        .with_anchor_threshold(0.04)
        .with_fit_error_threshold(0.5);
    let mut ref_ed = EdLinesRef::new()
        .with_min_extent(0.5)
        .with_gaussian_blur(0.8)
        .with_anchor_threshold(0.04)
        .with_nfa(true)
        .with_nfa_epsilon(1.0);

    // ─── 统计 ─────────────────────────────────────────────────────────────
    let mut bev_total = ClassStats::default();
    let mut ref_total = ClassStats::default();
    let mut frames_detail = Vec::new();
    let mut hungarian_buf = Vec::new();
    let total_start = Instant::now();

    println!("{:-<125}", "");
    println!("| {:>4} | {:>20} | {:>5} {:>5} {:>5} | {:>5} {:>5} {:>5} | {:>7} {:>7} |",
        "帧", "", "TP", "FP", "FN", "TP", "FP", "FN", "BEVms", "REFms");
    println!("{:-<125}", "");

    let mut frame_idx = 0usize;
    while data_loader.load_next().await? {
        if frame_idx >= n_total { break; }

        let cloud: Vec<[f32; 3]> = {
            let mut stream = global_swapl().clouds.lock().unwrap();
            match stream.read() {
                Some(data) => data,
                None => continue,
            }
        };
        if cloud.is_empty() { frame_idx += 1; continue; }

        // 地面检测（共享）
        let mut ground_buf = cloud;
        let (n_ground, _, _) = ground.pick(&mut ground_buf);
        let non_ground = &ground_buf[n_ground..];

        // GT
        let gt = filter_gt(&all_labels.get(frame_idx).cloned().unwrap_or_default(), max_range);
        let n_gt = gt.len();

        // ── BevEdLines ──
        let mut bev_buf = non_ground.to_vec();
        let t0 = Instant::now();
        let (n_wall_bev, _) = bev.pick(&mut bev_buf);
        let bev_time = t0.elapsed().as_secs_f64() * 1000.0;
        let (bev_boxes, _) = cluster_obstacles_with_indices(&bev_buf[n_wall_bev..], 0.30, 3, 0.05, 0.0);
        let (bev_tp, bev_fp, bev_fn) = match_boxes(&bev_boxes, &gt, center_dist, &mut hungarian_buf);

        // ── EdLinesRef ──
        let mut ref_buf = non_ground.to_vec();
        let t1 = Instant::now();
        let (n_wall_ref, _) = ref_ed.pick(&mut ref_buf);
        let ref_time = t1.elapsed().as_secs_f64() * 1000.0;
        let (ref_boxes, _) = cluster_obstacles_with_indices(&ref_buf[n_wall_ref..], 0.30, 3, 0.05, 0.0);
        let (ref_tp, ref_fp, ref_fn) = match_boxes(&ref_boxes, &gt, center_dist, &mut hungarian_buf);

        bev_total.tp += bev_tp; bev_total.fp += bev_fp; bev_total.fn_ += bev_fn;
        ref_total.tp += ref_tp; ref_total.fp += ref_fp; ref_total.fn_ += ref_fn;

        let diff = if bev_tp != ref_tp || bev_fp != ref_fp { " D" } else { "" };
        println!("| {:>4} | {:>20} | {:>5} {:>5} {:>5} | {:>5} {:>5} {:>5} | {:>7.2} {:>7.2} |{}",
            frame_idx, "BevEdLines / EdLinesRef",
            bev_tp, bev_fp, bev_fn, ref_tp, ref_fp, ref_fn,
            bev_time, ref_time, diff);

        frames_detail.push(FrameDetail {
            frame_idx, n_gt,
            bev_boxes: bev_boxes.len(), ref_boxes: ref_boxes.len(),
            bev_tp, bev_fp, bev_fn, ref_tp, ref_fp, ref_fn,
            bev_wall: n_wall_bev, ref_wall: n_wall_ref,
            bev_time_ms: bev_time, ref_time_ms: ref_time,
        });

        frame_idx += 1;
    }

    let total_elapsed = total_start.elapsed();

    // ─── 汇总 ─────────────────────────────────────────────────────────────
    println!("\n═══ 汇总 ({} 帧, {:.1}s) ═══\n", frame_idx, total_elapsed.as_secs_f64());

    println!("{:<40} {:>20} {:>20}", "", "BevEdLines", "EdLinesRef");
    println!("{:-<80}", "");
    println!("{:<40} {:>8}/{:>4}/{:>4} {:>8}/{:>4}/{:>4}",
        "TP / FP / FN",
        bev_total.tp, bev_total.fp, bev_total.fn_,
        ref_total.tp, ref_total.fp, ref_total.fn_);
    println!("{:<40} {:>19.1}% {:>19.1}%",
        "Precision", bev_total.precision() * 100.0, ref_total.precision() * 100.0);
    println!("{:<40} {:>19.1}% {:>19.1}%",
        "Recall", bev_total.recall() * 100.0, ref_total.recall() * 100.0);
    println!("{:<40} {:>19.4} {:>19.4}",
        "F1 Score", bev_total.f1(), ref_total.f1());
    println!();
    println!("{:<40}", "差值 (BevEdLines - EdLinesRef):");
    let dp = (bev_total.precision() - ref_total.precision()) * 100.0;
    let dr = (bev_total.recall() - ref_total.recall()) * 100.0;
    let df = (bev_total.f1() - ref_total.f1()) * 100.0;
    println!("{:<40} {:>+.2}pp", "  Precision", dp);
    println!("{:<40} {:>+.2}pp", "  Recall", dr);
    println!("{:<40} {:>+.2}pp", "  F1", df);

    // ─── 输出 JSON ────────────────────────────────────────────────────────
    let out_dir = PathBuf::from("output/edlines_bench");
    std::fs::create_dir_all(&out_dir)?;

    let mut config = HashMap::new();
    config.insert("n_frames".into(), frame_idx as f64);
    config.insert("center_dist".into(), center_dist as f64);
    config.insert("max_range".into(), max_range as f64);

    let output = EvalOutput {
        config, frames: frames_detail,
        bev_total: bev_total.clone(), ref_total: ref_total.clone(),
    };

    let json_path = out_dir.join("labeled_results.json");
    std::fs::write(&json_path, serde_json::to_string_pretty(&output)?)?;
    println!("\n结果已保存到: {}", json_path.display());

    Ok(())
}
