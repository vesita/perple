//! PR 曲线 + F1 随阈值变化
//!
//! 管线运行一次，在多个中心距阈值下计算 Precision/Recall/F1，
//! 输出 JSON 供 Python 绘制 PR 曲线和 F1 曲线。
//!
//! 用法：
//!   cargo run --example eval_pr_curve
//!   cargo run --example eval_pr_curve -- --frames 408 --output ./output/pr_curve

use std::collections::HashSet;
use std::path::PathBuf;
use std::time::Instant;

use perple::cloud::core::Lidar;
use perple::color::core::Camera;
use perple::fuse::Fuse;
use perple::optional::data_loader::DataLoader;
use perple::swapl::global_swapl;
use perple::tracker::core::Tracker;
use perple::tracker::output::Target;
use perple::utils::boxes::Box3D;

use log::info;
use serde::{Deserialize, Serialize};

// ═══════════════════════════════════════════════════════════════════════════════
//  Label 类型
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Deserialize)]
struct LabelItem {
    #[allow(dead_code)]
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
    x: f32,
    y: f32,
    z: f32,
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
//  加载标注
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

// ═══════════════════════════════════════════════════════════════════════════════
//  单帧匹配（中心距模式）
// ═══════════════════════════════════════════════════════════════════════════════

fn match_frame_center(
    detections: &[Target],
    gt_boxes: &[Box3D],
    threshold: f32,
    hungarian_buf: &mut Vec<Vec<f64>>,
) -> (usize, usize, usize) {
    let n_det = detections.len();
    let n_gt = gt_boxes.len();

    if n_det == 0 {
        return (0, 0, n_gt);
    }
    if n_gt == 0 {
        return (0, n_det, 0);
    }

    let mut cost = vec![vec![f64::MAX; n_gt]; n_det];
    for (i, det) in detections.iter().enumerate() {
        let dc = det.the_box.center();
        for (j, gt_box) in gt_boxes.iter().enumerate() {
            let gc = gt_box.center();
            let dist = ((dc.x - gc.x).powi(2) + (dc.y - gc.y).powi(2)).sqrt();
            if dist <= threshold {
                cost[i][j] = dist as f64;
            }
        }
    }

    let assignment = perple::tracker::hungarian::hungarian(&cost, hungarian_buf);

    let mut matched_gt = HashSet::new();
    let mut tp = 0usize;
    for (i, &gt_idx) in assignment.iter().enumerate() {
        if gt_idx < n_gt && cost[i][gt_idx] < f64::MAX / 2.0 {
            tp += 1;
            matched_gt.insert(gt_idx);
        }
    }
    let fp = n_det - tp;
    let fn_count = n_gt - matched_gt.len();

    (tp, fp, fn_count)
}

// ═══════════════════════════════════════════════════════════════════════════════
//  阈值扫描
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Serialize)]
struct Point {
    threshold: f32,
    precision: f64,
    recall: f64,
    f1: f64,
    tp: usize,
    fp: usize,
    fn_: usize,
}

fn evaluate_all_thresholds(
    frame_data: &[(Vec<Target>, Vec<Box3D>)],
    thresholds: &[f32],
) -> Vec<Point> {
    let mut results = Vec::with_capacity(thresholds.len());
    let mut hungarian_buf = Vec::new();

    for &thresh in thresholds {
        let mut total_tp = 0usize;
        let mut total_fp = 0usize;
        let mut total_fn = 0usize;

        for (targets, gt_boxes) in frame_data {
            let (tp, fp, fn_) = match_frame_center(targets, gt_boxes, thresh, &mut hungarian_buf);
            total_tp += tp;
            total_fp += fp;
            total_fn += fn_;
        }

        let precision = if total_tp + total_fp > 0 {
            total_tp as f64 / (total_tp + total_fp) as f64
        } else {
            0.0
        };
        let recall = if total_tp + total_fn > 0 {
            total_tp as f64 / (total_tp + total_fn) as f64
        } else {
            0.0
        };
        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };

        results.push(Point {
            threshold: thresh,
            precision,
            recall,
            f1,
            tp: total_tp,
            fp: total_fp,
            fn_: total_fn,
        });

        println!("  阈值 {:>4.2}: P={:>5.1}%  R={:>5.1}%  F1={:.4}  (TP={:>4} FP={:>4} FN={:>4})",
            thresh, precision * 100.0, recall * 100.0, f1,
            total_tp, total_fp, total_fn);
    }

    results
}

// ═══════════════════════════════════════════════════════════════════════════════
//  主流程
// ═══════════════════════════════════════════════════════════════════════════════

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Warn)
        .init();

    // ─── CLI ───────────────────────────────────────────────
    let args: Vec<String> = std::env::args().collect();
    let n_frames_limit: Option<usize> = args.iter()
        .position(|a| a == "--frames")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok());
    let out_prefix: String = args.iter()
        .position(|a| a == "--output")
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_default();

    // ─── 阈值列表 ──────────────────────────────────────────
    let thresholds: Vec<f32> = (1..=20).map(|i| i as f32 * 0.05).collect();

    // ─── 加载标注 ──────────────────────────────────────────
    let all_labels = load_labels("data/labeled/label");
    let n_label_frames = all_labels.len();
    info!("加载 {} 帧标注", n_label_frames);

    // ─── 检查 YOLO 模型 ────────────────────────────────────
    let config = perple::config::fixif();
    if !std::path::Path::new(&config.model_path).exists() {
        eprintln!("YOLO 模型不存在（{}）", config.model_path);
        std::process::exit(1);
    }

    // ─── 数据加载器 ────────────────────────────────────────
    let mut data_loader = DataLoader::new_independent(
        "data/labeled/camera/image".to_string(),
        "data/labeled/lidar".to_string(),
    );
    data_loader.load().await?;

    let n_frames = n_frames_limit
        .map(|n| n.min(data_loader.frame_count()).min(n_label_frames))
        .unwrap_or(data_loader.frame_count().min(n_label_frames));
    println!("将评估 {} 帧，{} 个阈值", n_frames, thresholds.len());
    println!();

    if n_frames == 0 {
        info!("没有帧需要处理");
        return Ok(());
    }

    // ─── 初始化管线 ────────────────────────────────────────
    let mut lidar = Lidar::new();
    let mut camera = Camera::new();
    let mut fuse = Fuse::new();
    let mut tracker = Tracker::new();

    // ─── 输出目录 ──────────────────────────────────────────
    let out_dir = if out_prefix.is_empty() {
        let secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();
        PathBuf::from(format!("output/pr_curve_{}", secs))
    } else {
        PathBuf::from(&out_prefix)
    };
    std::fs::create_dir_all(&out_dir)?;

    // ─── 存储每帧数据（两组：person 过滤 + 全部检测） ───────
    let mut frame_data_all: Vec<(Vec<Target>, Vec<Box3D>)> = Vec::with_capacity(n_frames);
    let mut frame_data_person: Vec<(Vec<Target>, Vec<Box3D>)> = Vec::with_capacity(n_frames);

    // ─── 预加载 ────────────────────────────────────────────
    if !data_loader.load_next().await? { info!("数据为空"); return Ok(()); }
    if n_frames > 1 { data_loader.load_next().await?; }

    let total_start = Instant::now();

    let mut l_handle = Some(tokio::spawn(async move { let _ = lidar.act().await; lidar }));
    let mut c_handle = Some(tokio::spawn(async move { let _ = camera.act().await; camera }));

    // ══════════════════════════════════════════════════════
    //  管线主循环
    // ══════════════════════════════════════════════════════
    for i in 0..n_frames {
        let (l_res, c_res) = tokio::join!(l_handle.take().unwrap(), c_handle.take().unwrap());
        lidar = l_res.unwrap();
        camera = c_res.unwrap();

        let swapl = global_swapl();
        swapl.cld_buds_raw.swap();
        swapl.clr_objs.swap();
        swapl.clouds_filtered.swap();
        swapl.ground_buds.swap();
        swapl.wall_buds.swap();

        if i + 1 < n_frames {
            l_handle = Some(tokio::spawn(async move { let _ = lidar.act().await; lidar }));
            c_handle = Some(tokio::spawn(async move { let _ = camera.act().await; camera }));
        }

        fuse.act().await;
        if i + 2 < n_frames { data_loader.load_next().await?; }
        let _ = tracker.run().await;

        let targets_all: Vec<Target> = swapl.targets.lock().unwrap().read().unwrap_or_default();
        let targets_person: Vec<Target> = targets_all.iter()
            .filter(|t| t.class_type == "person")
            .cloned()
            .collect();
        let gt_items = &all_labels[i];

        let gt_boxes: Vec<Box3D> = gt_items.iter().map(|item| item.to_box3d()).collect();
        frame_data_all.push((targets_all, gt_boxes.clone()));
        frame_data_person.push((targets_person, gt_boxes));

        if i % 50 == 0 || i == n_frames - 1 {
            println!("  管线进度: {:>4}/{} 帧", i + 1, n_frames);
        }
    }

    let pipeline_elapsed = total_start.elapsed().as_secs_f64();
    println!("\n管线完成: {:.1}s，开始多阈值评估...\n", pipeline_elapsed);

    // ══════════════════════════════════════════════════════
    //  多阈值计算（两组：person 过滤 + 全部检测）
    // ══════════════════════════════════════════════════════
    println!("  [Person 过滤]");
    let points_person = evaluate_all_thresholds(&frame_data_person, &thresholds);
    println!("\n  [全部检测]");
    let points_all = evaluate_all_thresholds(&frame_data_all, &thresholds);

    // ══════════════════════════════════════════════════════
    //  保存 JSON
    // ══════════════════════════════════════════════════════
    #[derive(Serialize)]
    struct Output {
        n_frames: usize,
        pipeline_elapsed_s: f64,
        /// Person 过滤的 PR 曲线点（保持向后兼容）
        points: Vec<Point>,
        /// 全部检测的 PR 曲线点（新增）
        points_all: Vec<Point>,
    }

    let output = Output {
        n_frames,
        pipeline_elapsed_s: pipeline_elapsed,
        points: points_person,
        points_all,
    };

    let json_path = out_dir.join("pr_curve.json");
    std::fs::write(&json_path, serde_json::to_string_pretty(&output)?)?;
    println!("\nJSON → {}", json_path.display());
    println!("══════════════════════════════════════════");
    println!("  评估完成");
    println!("══════════════════════════════════════════");

    Ok(())
}
