//! 基于标注数据的精度评估（计算准确率）
//!
//! 用 data/labeled 中的标注数据评测管线精度，
//! 输出 Precision / Recall / F1。使用两种匹配策略：
//!   1. IoU ≥ threshold  — 默认 0.15（行人较小，修改用 --iou）
//!   2. 中心距离 ≤ 0.5m  — 用 --center-dist 开启
//!
//! 用法：
//!   cargo run --example eval_labeled
//!   cargo run --example eval_labeled -- --iou 0.25
//!   cargo run --example eval_labeled -- --center-dist 0.5
//!   cargo run --example eval_labeled -- --frames 100 --output ./eval_result

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::time::Instant;

use perple::cloud::core::Lidar;
use perple::color::core::Camera;
use perple::fuse::Fuse;
use perple::optional::data_loader::DataLoader;
use perple::swapl::global_swapl;
use perple::tracker::core::Tracker;
use perple::tracker::output::Target;
use perple::yolo_smooth::YoloSmoother;
use perple::utils::boxes::Box3D;

use log::info;
use serde::{Deserialize, Serialize};

// ═══════════════════════════════════════════════════════════════════════════════
//  Label 类型（STPoints JSON 格式）
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

    fn dist_to_origin(&self) -> f32 {
        (self.psr.position.x.powi(2) + self.psr.position.y.powi(2)).sqrt()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  评估统计
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

/// 单帧匹配结果
struct FrameMatch {
    tp: usize,
    fp: usize,
    fn_count: usize,
    /// 被匹配的 GT 索引集合
    matched_gt: HashSet<usize>,
    /// (gt_idx, detection_class_type) 匹配对
    matched_pairs: Vec<(usize, String)>,
}

// ═══════════════════════════════════════════════════════════════════════════════
//  IoU 匹配（匈牙利算法）
// ═══════════════════════════════════════════════════════════════════════════════

fn match_frame(
    detections: &[Target],
    n_gt: usize,
    gt_boxes: &[Box3D],
    iou_threshold: f32,
    center_dist_threshold: f32,
    hungarian_buf: &mut Vec<Vec<f64>>,
) -> FrameMatch {
    let n_det = detections.len();
    let use_center = center_dist_threshold > 0.0;

    if n_det == 0 {
        return FrameMatch {
            tp: 0, fp: 0, fn_count: n_gt, matched_gt: HashSet::new(), matched_pairs: Vec::new(),
        };
    }
    if n_gt == 0 {
        return FrameMatch {
            tp: 0, fp: n_det, fn_count: 0, matched_gt: HashSet::new(), matched_pairs: Vec::new(),
        };
    }

    // 代价矩阵：detections × ground_truth
    let mut cost = vec![vec![f64::MAX; n_gt]; n_det];
    for (i, det) in detections.iter().enumerate() {
        for (j, gt_box) in gt_boxes.iter().enumerate() {
            if use_center {
                let dc = det.the_box.center();
                let gc = gt_box.center();
                let dist = ((dc.x - gc.x).powi(2) + (dc.y - gc.y).powi(2)).sqrt();
                if dist <= center_dist_threshold {
                    cost[i][j] = dist as f64;
                }
            } else {
                let iou = det.the_box.iou(gt_box);
                if iou >= iou_threshold {
                    cost[i][j] = (1.0 - iou as f64).max(0.0);
                }
            }
        }
    }

    let assignment = perple::tracker::hungarian::hungarian(&cost, hungarian_buf);

    let mut matched_gt = HashSet::new();
    let mut matched_pairs = Vec::new();
    let mut fp = 0usize;
    let mut tp = 0usize;

    for (i, &gt_idx) in assignment.iter().enumerate() {
        if gt_idx < n_gt && cost[i][gt_idx] < f64::MAX / 2.0 {
            tp += 1;
            matched_gt.insert(gt_idx);
            matched_pairs.push((gt_idx, detections[i].class_type.clone()));
        } else {
            fp += 1;
        }
    }
    let fn_count = n_gt - matched_gt.len();

    FrameMatch { tp, fp, fn_count, matched_gt, matched_pairs }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  加载标注 JSON
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
//  打印辅助
// ═══════════════════════════════════════════════════════════════════════════════

fn print_metrics(title: &str, stats: &ClassStats, n_gt: usize, n_det: usize) {
    if !title.is_empty() {
        println!("  {:<25}", title);
    }
    println!("    GT: {:>4}  | 检测: {:>4}  | TP: {:>4}  FP: {:>4}  FN: {:>4}",
        n_gt, n_det, stats.tp, stats.fp, stats.fn_);
    println!("    Precision: {:.1}%  | Recall: {:.1}%  | F1: {:.4}",
        stats.precision() * 100.0, stats.recall() * 100.0, stats.f1());
}

/// 找到距离所在桶的索引
fn bucket_index(dist: f32) -> usize {
    if dist <= 10.0 { 0 }
    else if dist <= 20.0 { 1 }
    else if dist <= 30.0 { 2 }
    else { 3 }
}

const BUCKET_LABELS: &[&str] = &["0~10m", "10~20m", "20~30m", "30m+"];

// ═══════════════════════════════════════════════════════════════════════════════
//  主流程
// ═══════════════════════════════════════════════════════════════════════════════

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Warn)
        .init();

    // ─── CLI ───────────────────────────────────────────────────────────────
    let args: Vec<String> = std::env::args().collect();
    let iou_threshold: f32 = args.iter()
        .position(|a| a == "--iou")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.15);
    let center_dist: f32 = args.iter()
        .position(|a| a == "--center-dist")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.0); // 0 = 不使用中心距离
    let n_frames_limit: Option<usize> = args.iter()
        .position(|a| a == "--frames")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok());
    let out_prefix: String = args.iter()
        .position(|a| a == "--output")
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_default();
    let disable_yolo_smooth: bool = args.iter().any(|a| a == "--disable-yolo-smooth");

    // ─── 加载标注 ─────────────────────────────────────────────────────────
    let label_dir = "data/labeled/label";
    let all_labels = load_labels(label_dir);
    let n_label_frames = all_labels.len();
    info!("加载 {} 帧标注 ({})", n_label_frames, label_dir);

    // GT 统计
    let mut gt_by_class: HashMap<String, usize> = HashMap::new();
    for frame in &all_labels {
        for item in frame {
            *gt_by_class.entry(item.obj_type.clone()).or_insert(0) += 1;
        }
    }
    let total_gt: usize = gt_by_class.values().sum();
    info!("共 {} 个 GT 目标", total_gt);

    // ─── 检查 YOLO 模型 ──────────────────────────────────────────────────
    let config = perple::config::fixif();
    if !std::path::Path::new(&config.model_path).exists() {
        eprintln!("YOLO 模型不存在（{}）", config.model_path);
        std::process::exit(1);
    }

    // ─── 数据加载器（指向 labeled 目录） ───────────────────────────────────
    let mut data_loader = DataLoader::new_independent(
        "data/labeled/camera/image".to_string(),
        "data/labeled/lidar".to_string(),
    );
    data_loader.load().await?;

    let n_frames = n_frames_limit
        .map(|n| n.min(data_loader.frame_count()).min(n_label_frames))
        .unwrap_or(data_loader.frame_count().min(n_label_frames));
    info!("将评估 {} 帧", n_frames);

    if n_frames == 0 {
        info!("没有帧需要处理");
        return Ok(());
    }

    // ─── 初始化管线 ───────────────────────────────────────────────────────
    let mut lidar = Lidar::new();
    let mut camera = Camera::new();
    let mut fuse = Fuse::new();
    let mut tracker = Tracker::new();

    // ─── 输出目录 ─────────────────────────────────────────────────────────
    let out_dir = if out_prefix.is_empty() {
        let secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();
        PathBuf::from(format!("output/eval_labeled_{}", secs))
    } else {
        PathBuf::from(&out_prefix)
    };
    std::fs::create_dir_all(&out_dir)?;

    // ─── 预加载前两帧 ─────────────────────────────────────────────────────
    if !data_loader.load_next().await? { info!("数据为空"); return Ok(()); }
    if n_frames > 1 { data_loader.load_next().await?; }

    // ═════════════════════════════════════════════════════════════════════
    //  累计统计
    // ═════════════════════════════════════════════════════════════════════
    let mut overall = ClassStats::default();
    let mut overall_spatial = ClassStats::default();   // 空间匹配（忽略类名）
    let mut tp_person = 0usize;                         // 正确分类为 person 的匹配
    let mut tp_nonperson = 0usize;                      // 空间匹配但类名非 person
    let mut per_class: HashMap<String, ClassStats> = HashMap::new();
    let mut per_distance: Vec<ClassStats> = (0..4).map(|_| ClassStats::default()).collect();

    let mut hungarian_buf = Vec::new();
    let total_start = Instant::now();
    let mut total_gt_count = 0usize;
    let mut total_det_count = 0usize;
    let mut total_det_person = 0usize;

    // 启动第一帧
    let mut l_handle = Some(tokio::spawn(async move { let _ = lidar.act().await; lidar }));
    let mut c_handle = Some(tokio::spawn(async move { let _ = camera.act().await; camera }));
    let mut yolo_smoother = YoloSmoother::new();

    for i in 0..n_frames {
        // ── 等待检测完成 ─────────────────────────────────────────────────
        let (l_res, c_res) = tokio::join!(l_handle.take().unwrap(), c_handle.take().unwrap());
        lidar = l_res.unwrap();
        camera = c_res.unwrap();

        let swapl = global_swapl();
        swapl.cld_buds_raw.swap();
        swapl.clr_objs.swap();
        // YOLO 标签平滑（在 Camera→Fuse 之间）
        if !disable_yolo_smooth {
            yolo_smoother.smooth(&mut *swapl.clr_objs.consumer().lock().unwrap());
        }
        swapl.clouds_filtered.swap();
        swapl.ground_buds.swap();
        swapl.wall_buds.swap();

        if i + 1 < n_frames {
            l_handle = Some(tokio::spawn(async move { let _ = lidar.act().await; lidar }));
            c_handle = Some(tokio::spawn(async move { let _ = camera.act().await; camera }));
        }

        // ── 后融合 + 跟踪 ────────────────────────────────────────────────
        fuse.act().await;
        if i + 2 < n_frames { data_loader.load_next().await?; }
        let _ = tracker.run().await;

        let targets_all: Vec<Target> = swapl.targets.lock().unwrap().read().unwrap_or_default();
        let n_person = targets_all.iter().filter(|t| t.class_type == "person").count();
        // 不按 class_type 过滤，让所有检测参与空间匹配
        // 后续通过 matched_pairs 中的 class_type 区分 "正确分类" 和 "空间正确但类名不对"
        total_det_person += n_person;
        let gt_items = &all_labels[i];

        if gt_items.is_empty() && targets_all.is_empty() {
            continue;
        }

        // 预计算 GT Box3D
        let gt_boxes: Vec<Box3D> = gt_items.iter().map(|item| item.to_box3d()).collect();

        total_gt_count += gt_items.len();
        total_det_count += targets_all.len();

        // ── 匈牙利匹配（所有检测参与） ────────────────────────────────────
        let fm = match_frame(&targets_all, gt_items.len(), &gt_boxes, iou_threshold, center_dist, &mut hungarian_buf);

        // ── 按分类质量区分匹配 ────────────────────────────────────────────
        let mut tp_strict = 0usize;
        let mut tp_nonperson_here = 0usize;
        for (_, det_class) in &fm.matched_pairs {
            if det_class == "person" {
                tp_strict += 1;
            } else {
                tp_nonperson_here += 1;
            }
        }
        tp_person += tp_strict;
        tp_nonperson += tp_nonperson_here;

        // ── 累计 overall（strict: 仅 person 检测参与） ──────────────────
        // strict 的 TP = 匹配中 class_type=="person" 的数量
        // strict 的 FN = GT - tp_strict（未匹配或匹配了但检测非 person 都算 FN）
        // strict 的 FP = person 检测总数 - tp_strict
        overall.tp += tp_strict;
        overall.fp += n_person - tp_strict;
        overall.fn_ += gt_items.len() - tp_strict;

        // ── 累计 overall_spatial（所有检测参与） ─────────────────────────
        overall_spatial.tp += fm.tp;
        overall_spatial.fp += fm.fp;
        overall_spatial.fn_ += fm.fn_count;

        // ── 按类别 + 按距离累计 ───────────────────────────────────────────
        // TP: matched GT → 加 TP 到对应类别和距离桶
        for &gt_idx in &fm.matched_gt {
            let item = &gt_items[gt_idx];
            // per class
            let cls_stats = per_class.entry(item.obj_type.clone()).or_default();
            cls_stats.tp += 1;
            // per distance
            let d = item.dist_to_origin();
            let bi = bucket_index(d);
            per_distance[bi].tp += 1;
        }

        // FN: unmatched GT → 加 FN 到对应类别和距离桶
        for (gt_idx, item) in gt_items.iter().enumerate() {
            let d = item.dist_to_origin();
            let bi = bucket_index(d);

            if !fm.matched_gt.contains(&gt_idx) {
                let cls_stats = per_class.entry(item.obj_type.clone()).or_default();
                cls_stats.fn_ += 1;
                per_distance[bi].fn_ += 1;
            }
        }

        // FP: unmatched detections → 计入 overall，但不按类别/距离分
        // 因为 FP 没有对应 GT，无法确定其类别

        if i % 50 == 0 || i == n_frames - 1 || i < 5 {
            println!("  进度: {:>4}/{} | GT: {:>3} 检测: {:>3}(person:{}) | TP: {} FP: {} FN: {}",
                i + 1, n_frames, gt_items.len(), targets_all.len(), n_person,
                fm.tp, fm.fp, fm.fn_count);
        }
    }

    let total_elapsed = total_start.elapsed().as_secs_f64();

    // ═════════════════════════════════════════════════════════════════════
    //  输出结果
    // ═════════════════════════════════════════════════════════════════════
    println!();
    println!("╔════════════════════════════════════════════╗");
    println!("║       标注数据精度评估结果                  ║");
    println!("╚════════════════════════════════════════════╝");
    println!();
    let match_desc = if center_dist > 0.0 {
        format!("中心距离 ≤ {:.1}m", center_dist)
    } else {
        format!("IoU ≥ {:.2}", iou_threshold)
    };
    println!("  配置: {}  |  帧数: {}  |  耗时: {:.1}s",
        match_desc, n_frames, total_elapsed);
    println!();
    println!("  ── 严格评估 (仅 class_type == \"person\") ──");
    print_metrics("", &overall, total_gt_count, total_det_person);
    println!();
    println!("  ── 空间评估 (全部检测参与匹配) ──");
    print_metrics("", &overall_spatial, total_gt_count, total_det_count);
    println!();

    // ─── 行人类别识别分析 ──
    let spatial_recall = if total_gt_count > 0 {
        tp_person as f64 / total_gt_count as f64
    } else { 0.0 };
    let nonperson_recall = if total_gt_count > 0 {
        tp_nonperson as f64 / total_gt_count as f64
    } else { 0.0 };
    println!("  ── 行人类别识别分析 ──");
    println!("    空间匹配正确: {}/{} ({:.1}%)",
        tp_person + tp_nonperson, total_gt_count,
        (tp_person + tp_nonperson) as f64 / total_gt_count.max(1) as f64 * 100.0);
    println!("    ├─ 正确分类为 person: {} ({:.1}%)", tp_person, spatial_recall * 100.0);
    println!("    └─ 误分类为 obstacle/其他: {} ({:.1}%)", tp_nonperson, nonperson_recall * 100.0);
    println!();

    // ─── 按类别 ──
    println!("  ── 按类别（Recall 基于 GT 统计，FP 只计入总体）──");
    let mut class_vec: Vec<_> = per_class.iter().collect();
    class_vec.sort_by_key(|(_, s)| s.tp + s.fn_);
    class_vec.reverse();
    for (cls, stats) in &class_vec {
        let n = stats.tp + stats.fn_;
        let recall = if n > 0 { stats.tp as f64 / n as f64 } else { 0.0 };
        println!("  {:>12}: TP={:>3} FN={:>3}  Recall={:.1}% ({}/{})",
            cls, stats.tp, stats.fn_, recall * 100.0, stats.tp, n);
    }
    println!();

    // ─── 按距离 ──
    println!("  ── 按距离 ──");
    println!("  {:>8}  {:>5}  {:>5}  {:>5}  {:>5}  {:>8}  {:>8}  {:>6}",
        "范围", "GT", "TP", "FN", "检测", "Precision", "Recall", "F1");
    for (bi, label) in BUCKET_LABELS.iter().enumerate() {
        let s = &per_distance[bi];
        let n_gt = s.tp + s.fn_;
        // 该桶的 FP 无法单独统计，跳过 Precision
        let recall = if n_gt > 0 { s.tp as f64 / n_gt as f64 } else { 0.0 };
        println!("  {:>8}  {:>5}  {:>5}  {:>5}  {:>5}  {:>8}  {:>6.1}%  {:>.4}",
            label, n_gt, s.tp, s.fn_, "-",
            "-", recall * 100.0, recall);
    }
    println!();

    // ─── 保存 JSON ─────────────────────────────────────────────────────────
    #[derive(Serialize)]
    struct Output {
        iou_threshold: f32,
        n_frames: usize,
        n_gt: usize,
        // strict (person-only)
        n_detections: usize,
        tp: usize,
        fp: usize,
        fn_: usize,
        precision: f64,
        recall: f64,
        f1: f64,
        // spatial (all detections)
        n_detections_spatial: usize,
        tp_spatial: usize,
        fp_spatial: usize,
        fn_spatial: usize,
        precision_spatial: f64,
        recall_spatial: f64,
        f1_spatial: f64,
        // classification breakdown
        tp_person: usize,
        tp_nonperson: usize,
        per_class: Vec<ClassOutput>,
        per_distance: Vec<DistanceOutput>,
    }
    #[derive(Serialize)]
    struct ClassOutput {
        name: String,
        tp: usize,
        fn_: usize,
        recall: f64,
    }
    #[derive(Serialize)]
    struct DistanceOutput {
        range: String,
        tp: usize,
        fn_: usize,
        recall: f64,
    }

    let output = Output {
        iou_threshold,
        n_frames,
        n_gt: total_gt_count,
        n_detections: total_det_person,
        tp: overall.tp,
        fp: overall.fp,
        fn_: overall.fn_,
        precision: overall.precision(),
        recall: overall.recall(),
        f1: overall.f1(),
        n_detections_spatial: total_det_count,
        tp_spatial: overall_spatial.tp,
        fp_spatial: overall_spatial.fp,
        fn_spatial: overall_spatial.fn_,
        precision_spatial: overall_spatial.precision(),
        recall_spatial: overall_spatial.recall(),
        f1_spatial: overall_spatial.f1(),
        tp_person,
        tp_nonperson,
        per_class: {
            let mut v: Vec<_> = per_class.iter().map(|(name, s)| ClassOutput {
                name: name.clone(),
                tp: s.tp,
                fn_: s.fn_,
                recall: if s.tp + s.fn_ > 0 { s.tp as f64 / (s.tp + s.fn_) as f64 } else { 0.0 },
            }).collect();
            v.sort_by_key(|c| c.tp + c.fn_);
            v.reverse();
            v
        },
        per_distance: BUCKET_LABELS.iter().enumerate().map(|(bi, label)| {
            let s = &per_distance[bi];
            let n_gt = s.tp + s.fn_;
            DistanceOutput {
                range: label.to_string(),
                tp: s.tp,
                fn_: s.fn_,
                recall: if n_gt > 0 { s.tp as f64 / n_gt as f64 } else { 0.0 },
            }
        }).collect(),
    };

    let json_path = out_dir.join("eval_result.json");
    std::fs::write(&json_path, serde_json::to_string_pretty(&output)?)?;
    println!("  JSON → {}", json_path.display());

    // ─── CSV ───────────────────────────────────────────────────────────────
    {
        use std::io::Write;
        let csv_path = out_dir.join("eval_result.csv");
        let mut f = std::fs::File::create(&csv_path)?;
        writeln!(f, "metric,value")?;
        writeln!(f, "iou_threshold,{:.1}", iou_threshold)?;
        writeln!(f, "n_frames,{}", n_frames)?;
        writeln!(f, "n_gt,{}", total_gt_count)?;
        writeln!(f, "n_detections,{}", total_det_person)?;
        writeln!(f, "tp,{}", overall.tp)?;
        writeln!(f, "fp,{}", overall.fp)?;
        writeln!(f, "fn,{}", overall.fn_)?;
        writeln!(f, "precision,{:.4}", overall.precision())?;
        writeln!(f, "recall,{:.4}", overall.recall())?;
        writeln!(f, "f1,{:.4}", overall.f1())?;
        writeln!(f, "n_detections_spatial,{}", total_det_count)?;
        writeln!(f, "tp_spatial,{}", overall_spatial.tp)?;
        writeln!(f, "fp_spatial,{}", overall_spatial.fp)?;
        writeln!(f, "fn_spatial,{}", overall_spatial.fn_)?;
        writeln!(f, "precision_spatial,{:.4}", overall_spatial.precision())?;
        writeln!(f, "recall_spatial,{:.4}", overall_spatial.recall())?;
        writeln!(f, "f1_spatial,{:.4}", overall_spatial.f1())?;
        writeln!(f, "tp_person,{}", tp_person)?;
        writeln!(f, "tp_nonperson,{}", tp_nonperson)?;
        writeln!(f, "elapsed_s,{:.1}", total_elapsed)?;
    }
    println!("  CSV → {}/eval_result.csv", out_dir.display());

    println!();
    println!("══════════════════════════════════════════");
    println!("  评估完成");
    println!("══════════════════════════════════════════");

    Ok(())
}
