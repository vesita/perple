//! 综合评估工具 — 可调参数消融实验，同时输出 person 过滤与不过滤两种结果。
//!
//! 用法:
//!   cargo run --example eval_ablation
//!   cargo run --example eval_ablation -- --frames 408
//!   cargo run --example eval_ablation -- --ground-toml 'ground_strategy="peak_scan",ground_expand=0.15'
//!   cargo run --example eval_ablation -- --cluster-toml 'strategy="dbscan_qt",merge_patience=0.05,min_points_per_cluster=8'
//!   cargo run --example eval_ablation -- --denoise-toml 'denoise_radius=0.30,denoise_min_pts=5'
//!   cargo run --example eval_ablation -- --tracker-toml 'max_disappeared=24,min_confidence=0.3'
//!   cargo run --example eval_ablation -- --config ./experiment.toml
//!   cargo run --example eval_ablation -- --center-dist 0.5

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::time::Instant;

use perple::cloud::core::Lidar;
use perple::color::core::Camera;
use perple::config::{fixif, init_config, Config};
use perple::fuse::Fuse;
use perple::optional::data_loader::DataLoader;
use perple::swapl::global_swapl;
use perple::tracker::core::Tracker;
use perple::tracker::output::Target;
use perple::utils::boxes::Box3D;

use log::info;
use serde::{Deserialize, Serialize};

// ═══════════════════════════════════════════════════════════════════════════════
//  CLI 参数解析
// ═══════════════════════════════════════════════════════════════════════════════

struct Args {
    n_frames: Option<usize>,
    out_prefix: String,
    iou_threshold: f32,
    center_dist: f32,
    /// TOML 覆盖字符串（地面参数等顶层字段）
    ground_toml: Option<String>,
    /// TOML 覆盖字符串（聚类参数 → [cluster] 段）
    cluster_toml: Option<String>,
    /// TOML 覆盖字符串（降噪参数 → [cluster] 段）
    denoise_toml: Option<String>,
    /// TOML 覆盖字符串（跟踪参数 → [tracker] 段）
    tracker_toml: Option<String>,
    /// 完整配置文件的路径
    config_path: Option<String>,
    /// Person 类别标签（过滤 pipeline class_type）
    person_label: String,
    /// GT 中 person 类别标签（默认 "Pedestrian" 以匹配标注数据）
    gt_person_label: String,
}

fn parse_args() -> Args {
    let args: Vec<String> = std::env::args().collect();

    fn get(args: &[String], flag: &str) -> Option<String> {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1))
            .cloned()
    }

    Args {
        n_frames: get(&args, "--frames").and_then(|s| s.parse().ok()),
        out_prefix: get(&args, "--output").unwrap_or_default(),
        iou_threshold: get(&args, "--iou")
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.15),
        center_dist: get(&args, "--center-dist")
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.5),
        ground_toml: get(&args, "--ground-toml"),
        cluster_toml: get(&args, "--cluster-toml"),
        denoise_toml: get(&args, "--denoise-toml"),
        tracker_toml: get(&args, "--tracker-toml"),
        config_path: get(&args, "--config"),
        person_label: get(&args, "--person-label").unwrap_or_else(|| "person".to_string()),
        gt_person_label: get(&args, "--gt-person-label").unwrap_or_else(|| "Pedestrian".to_string()),
    }
}

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

// ═══════════════════════════════════════════════════════════════════════════════
//  匈牙利匹配（IoU / 中心距）
// ═══════════════════════════════════════════════════════════════════════════════

struct FrameMatch {
    tp: usize,
    fp: usize,
    fn_count: usize,
    matched_gt: HashSet<usize>,
    /// (gt_idx, detection_class_type) 匹配对，用于分类质量分析
    matched_pairs: Vec<(usize, String)>,
}

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
        return FrameMatch { tp: 0, fp: 0, fn_count: n_gt, matched_gt: HashSet::new(), matched_pairs: Vec::new() };
    }
    if n_gt == 0 {
        return FrameMatch { tp: 0, fp: n_det, fn_count: 0, matched_gt: HashSet::new(), matched_pairs: Vec::new() };
    }

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
    let mut tp = 0usize;
    for (i, &gt_idx) in assignment.iter().enumerate() {
        if gt_idx < n_gt && cost[i][gt_idx] < f64::MAX / 2.0 {
            tp += 1;
            matched_gt.insert(gt_idx);
            matched_pairs.push((gt_idx, detections[i].class_type.clone()));
        }
    }
    let fp = n_det - tp;
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
//  组装 TOML 覆盖
// ═══════════════════════════════════════════════════════════════════════════════

fn build_override_toml(args: &Args) -> String {
    let mut parts = Vec::new();

    // 地面参数 → 顶层字段
    if let Some(ref s) = args.ground_toml {
        for pair in s.split(',') {
            let pair = pair.trim();
            if !pair.is_empty() {
                parts.push(pair.to_string());
            }
        }
    }

    // 降噪参数 → [cluster] 段下字段
    if let Some(ref s) = args.denoise_toml {
        if !parts.iter().any(|p| p.starts_with("\n[cluster]")) {
            parts.push("\n[cluster]".to_string());
        }
        for pair in s.split(',') {
            let pair = pair.trim();
            if !pair.is_empty() {
                parts.push(pair.to_string());
            }
        }
    }

    // 聚类参数 → [cluster] 段下字段
    if let Some(ref s) = args.cluster_toml {
        if !parts.iter().any(|p| p.starts_with("\n[cluster]")) {
            parts.push("\n[cluster]".to_string());
        }
        for pair in s.split(',') {
            let pair = pair.trim();
            if !pair.is_empty() {
                parts.push(pair.to_string());
            }
        }
    }

    // 跟踪参数 → [tracker] 段
    if let Some(ref s) = args.tracker_toml {
        parts.push("\n[tracker]".to_string());
        for pair in s.split(',') {
            let pair = pair.trim();
            if !pair.is_empty() {
                parts.push(pair.to_string());
            }
        }
    }

    parts.join("\n")
}

// ═══════════════════════════════════════════════════════════════════════════════
//  打印辅助
// ═══════════════════════════════════════════════════════════════════════════════

fn print_metrics(title: &str, stats: &ClassStats, n_gt: usize, n_det: usize) {
    println!("  {:<30}", title);
    println!("    GT: {:>4}  | 检测: {:>4}  | TP: {:>4}  FP: {:>4}  FN: {:>4}",
        n_gt, n_det, stats.tp, stats.fp, stats.fn_);
    println!("    Precision: {:>5.1}%  | Recall: {:>5.1}%  | F1: {:.4}",
        stats.precision() * 100.0, stats.recall() * 100.0, stats.f1());
}

// ═══════════════════════════════════════════════════════════════════════════════
//  主流程
// ═══════════════════════════════════════════════════════════════════════════════

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Warn)
        .init();

    let args = parse_args();

    // ─── 配置初始化（支持覆盖）──────────────────────────────────────────────
    if let Some(ref config_path) = args.config_path {
        // 从指定文件完整加载
        let config = Config::from_file(config_path);
        init_config(config);
    } else {
        let toml_override = build_override_toml(&args);
        if toml_override.is_empty() {
            // 无覆盖：先 init_config 阻止 fixif() 的 get_or_init 读环境变量
            init_config(Config::new());
        } else {
            let mut config = Config::from_file("config/default.toml");
            config.update_from_toml(&toml_override)?;
            init_config(config);
        }
    }

    // 日志：打印当前配置摘要
    let cfg = fixif();
    info!("地面策略: {}", cfg.ground_strategy);
    info!("聚类策略: {}", cfg.cluster.strategy);
    info!("聚类参数: merge_patience={}, min_pts={:?}, voxel_size={}",
        cfg.cluster.merge_patience, cfg.cluster.min_points_per_cluster, cfg.cluster.voxel_size);
    info!("降噪参数: radius={}, min_pts={}", cfg.cluster.denoise_radius, cfg.cluster.denoise_min_pts);
    info!("跟踪参数: max_disappeared={}, min_confidence={}",
        cfg.tracker.max_disappeared, cfg.tracker.min_confidence);

    // ─── 加载标注 ─────────────────────────────────────────────────────────
    let label_dir = "data/labeled/label";
    let all_labels = load_labels(label_dir);
    let n_label_frames = all_labels.len();
    info!("加载 {} 帧标注 ({})", n_label_frames, label_dir);

    // GT 类别统计
    let mut gt_by_class: HashMap<String, usize> = HashMap::new();
    for frame in &all_labels {
        for item in frame {
            *gt_by_class.entry(item.obj_type.clone()).or_insert(0) += 1;
        }
    }
    let total_gt: usize = gt_by_class.values().sum();
    info!("共 {} 个 GT 目标: {:?}", total_gt, gt_by_class);

    // ─── 检查 YOLO 模型 ──────────────────────────────────────────────────
    if !std::path::Path::new(&cfg.model_path).exists() {
        eprintln!("YOLO 模型不存在（{}）", cfg.model_path);
        std::process::exit(1);
    }

    // ─── 数据加载器 ───────────────────────────────────────────────────────
    let mut data_loader = DataLoader::new_independent(
        "data/labeled/camera/image".to_string(),
        "data/labeled/lidar".to_string(),
    );
    data_loader.load().await?;

    let n_frames = args.n_frames
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
    let out_dir = if args.out_prefix.is_empty() {
        let secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();
        PathBuf::from(format!("output/eval_ablation_{}", secs))
    } else {
        PathBuf::from(&args.out_prefix)
    };
    std::fs::create_dir_all(&out_dir)?;

    // ═════════════════════════════════════════════════════════════════════
    //  累计统计（双模式：person 过滤 + 全部类别）
    // ═════════════════════════════════════════════════════════════════════
    let mut overall_person = ClassStats::default();
    let mut overall_all = ClassStats::default();
    let mut tp_person = 0usize;      // 全部检测模式中，匹配 GT 且 class_type=="person"
    let mut tp_nonperson = 0usize;   // 全部检测模式中，匹配 GT 但 class_type 非 person
    let mut per_class_all: HashMap<String, ClassStats> = HashMap::new();

    let mut hungarian_buf = Vec::new();
    let total_start = Instant::now();
    let mut total_gt_count = 0usize;
    let mut total_det_count_all = 0usize;
    let mut total_det_count_person = 0usize;
    let mut total_person_gt_count = 0usize;

    // 预加载前两帧
    if !data_loader.load_next().await? { info!("数据为空"); return Ok(()); }
    if n_frames > 1 { data_loader.load_next().await?; }

    let mut l_handle = Some(tokio::spawn(async move { let _ = lidar.act().await; lidar }));
    let mut c_handle = Some(tokio::spawn(async move { let _ = camera.act().await; camera }));

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
            .filter(|t| t.class_type == args.person_label)
            .cloned()
            .collect();

        // ── 过滤 GT ──
        let gt_items = &all_labels[i];
        let person_gt: Vec<&LabelItem> = gt_items.iter()
            .filter(|item| item.obj_type == args.gt_person_label)
            .collect();

        if gt_items.is_empty() && targets_all.is_empty() {
            continue;
        }

        // 预计算 GT Box3D
        let all_gt_boxes: Vec<Box3D> = gt_items.iter().map(|item| item.to_box3d()).collect();
        let person_gt_boxes: Vec<Box3D> = person_gt.iter().map(|item| item.to_box3d()).collect();

        total_gt_count += gt_items.len();
        total_person_gt_count += person_gt.len();
        total_det_count_all += targets_all.len();
        total_det_count_person += targets_person.len();

        // ─────────────────────────────────────────────────────────────────
        //  Mode A: 全部类别匹配
        // ─────────────────────────────────────────────────────────────────
        let fm_all = match_frame(
            &targets_all, gt_items.len(), &all_gt_boxes,
            args.iou_threshold, args.center_dist, &mut hungarian_buf,
        );
        overall_all.tp += fm_all.tp;
        overall_all.fp += fm_all.fp;
        overall_all.fn_ += fm_all.fn_count;

        // 分类质量分析：匹配的检测是 person 还是非 person
        // 注意：这些分类统计只用于分析识别准确率，不影响 overall_all 的 TP/FP 计数。
        // overall_all 基于空间匹配（忽略类名），所有检测参与，FP 覆盖全部误检。
        // 下面的 overall_person（Mode B）则只统计 person 类检测的 FP。
        // 详见 memory 中跟踪策略有效性的记录：分类质量高（87.9%），瓶颈在空间召回率（65.4%）。
        for (_, det_class) in &fm_all.matched_pairs {
            if det_class == "person" {
                tp_person += 1;
            } else {
                tp_nonperson += 1;
            }
        }

        // 按 GT 类别累计（TP + FN）
        for (gt_idx, item) in gt_items.iter().enumerate() {
            let cls = item.obj_type.clone();
            let stats = per_class_all.entry(cls).or_default();
            if fm_all.matched_gt.contains(&gt_idx) {
                stats.tp += 1;
            } else {
                stats.fn_ += 1;
            }
        }

        // ─────────────────────────────────────────────────────────────────
        //  Mode B: Person 过滤匹配
        // ─────────────────────────────────────────────────────────────────
        if !person_gt.is_empty() || !targets_person.is_empty() {
            let fm_p = match_frame(
                &targets_person, person_gt.len(), &person_gt_boxes,
                args.iou_threshold, args.center_dist, &mut hungarian_buf,
            );
            overall_person.tp += fm_p.tp;
            overall_person.fp += fm_p.fp;
            overall_person.fn_ += fm_p.fn_count;
        }

        if i % 50 == 0 || i == n_frames - 1 || i < 5 {
            println!("  进度: {:>4}/{} | 全部 GT: {:>3} 检测: {:>3}  TP:{} FP:{} FN:{} | Person GT: {:>2} 检测: {:>2}",
                i + 1, n_frames,
                gt_items.len(), targets_all.len(), fm_all.tp, fm_all.fp, fm_all.fn_count,
                person_gt.len(), targets_person.len());
        }
    }

    let total_elapsed = total_start.elapsed().as_secs_f64();

    // ═════════════════════════════════════════════════════════════════════
    //  输出结果
    // ═════════════════════════════════════════════════════════════════════

    let match_desc = if args.center_dist > 0.0 {
        format!("中心距离 ≤ {:.1}m", args.center_dist)
    } else {
        format!("IoU ≥ {:.2}", args.iou_threshold)
    };

    println!();
    println!("╔════════════════════════════════════════════════════╗");
    println!("║       消融实验评估结果                              ║");
    println!("╚════════════════════════════════════════════════════╝");
    println!();
    println!("  配置: {}  |  帧数: {}  |  耗时: {:.1}s",
        match_desc, n_frames, total_elapsed);
    println!();

    println!("  ── 配置概览 ──");
    println!("  地面: {} (expand={})", cfg.ground_strategy, cfg.ground_expand);
    println!("  聚类: {} (patience={}, min_pts={:?}, voxel={})",
        cfg.cluster.strategy, cfg.cluster.merge_patience,
        cfg.cluster.min_points_per_cluster, cfg.cluster.voxel_size);
    println!("  降噪: radius={}, min_pts={}", cfg.cluster.denoise_radius, cfg.cluster.denoise_min_pts);
    println!("  跟踪: max_disappeared={}, min_confidence={}, kf_avg_frames={}",
        cfg.tracker.max_disappeared, cfg.tracker.min_confidence, cfg.tracker.kf_avg_frames);
    println!("  KF:   noise_pos={} noise_vel={} noise_acc={} noise_size={}",
        cfg.tracker.kf_measurement_noise_pos,
        cfg.tracker.kf_measurement_noise_vel,
        cfg.tracker.kf_measurement_noise_acc,
        cfg.tracker.kf_measurement_noise_size);
    println!("  KF:   proc_pos={} proc_vel={} proc_acc={} cov_init={} gate={}",
        cfg.tracker.kf_process_noise_pos,
        cfg.tracker.kf_process_noise_vel,
        cfg.tracker.kf_process_noise_acc,
        cfg.tracker.kf_initial_covariance_scale,
        cfg.tracker.kf_gate_threshold);
    println!();

    // ── 输出文件路径 ──
    let json_path = out_dir.join("eval_result.json");
    let csv_path = out_dir.join("eval_result.csv");
    println!("  输出 → {}/", out_dir.display());
    println!();

    // ── Mode A: 全部类别 ──
    println!("  ═══ 全部类别 (All Classes) ═══");
    print_metrics("总体 (Overall)", &overall_all, total_gt_count, total_det_count_all);
    println!();

    // ── 类别识别分析 ──
    let total_matched = tp_person + tp_nonperson;
    println!("  ── 行人类别识别分析 (在全部GT中) ──");
    println!("    空间匹配正确: {}/{} ({:.1}%)",
        total_matched, total_gt_count,
        total_matched as f64 / total_gt_count.max(1) as f64 * 100.0);
    println!("    ├─ 正确分类为 person: {} ({:.1}%)",
        tp_person, tp_person as f64 / total_gt_count.max(1) as f64 * 100.0);
    println!("    └─ 误分类为 obstacle/其他: {} ({:.1}%)",
        tp_nonperson, tp_nonperson as f64 / total_gt_count.max(1) as f64 * 100.0);
    println!();

    // 按类别明细
    println!("  ── 按类别明细（只计 Recall，FP 只在总体反映）──");
    let mut class_vec: Vec<_> = per_class_all.iter().collect();
    class_vec.sort_by_key(|(_, s)| s.tp + s.fn_);
    class_vec.reverse();
    for (cls, stats) in &class_vec {
        let n = stats.tp + stats.fn_;
        let recall = if n > 0 { stats.tp as f64 / n as f64 } else { 0.0 };
        println!("  {:>15}: TP={:>3} FN={:>3}  Recall={:>5.1}%  ({}/{})",
            cls, stats.tp, stats.fn_, recall * 100.0, stats.tp, n);
    }
    println!();

    // ── Mode B: Person 过滤 ──
    if total_det_count_person > 0 || overall_person.tp > 0 || overall_person.fn_ > 0 {
        println!("  ═══ Person 过滤 (Person Only) ═══");
        print_metrics("Person", &overall_person, total_person_gt_count, total_det_count_person);
        println!();
    }

    // ═════════════════════════════════════════════════════════════════════
    //  保存 JSON
    // ═════════════════════════════════════════════════════════════════════

    #[derive(Serialize)]
    struct ModeOutput {
        n_gt: usize,
        n_detections: usize,
        tp: usize,
        fp: usize,
        #[serde(rename = "fn")]
        fn_: usize,
        precision: f64,
        recall: f64,
        f1: f64,
    }

    #[derive(Serialize)]
    struct Output {
        config: ConfigSummary,
        iou_threshold: f32,
        center_dist: f32,
        n_frames: usize,
        elapsed_s: f64,
        all_classes: ModeOutput,
        person_only: ModeOutput,
        // 分类质量分析
        tp_person: usize,
        tp_nonperson: usize,
        per_class: Vec<ClassOutput>,
    }

    #[derive(Serialize)]
    struct ConfigSummary {
        ground_strategy: String,
        ground_expand: f32,
        cluster_strategy: String,
        cluster_patience: f32,
        cluster_voxel_size: f32,
        cluster_min_pts: Option<usize>,
        denoise_radius: f32,
        denoise_min_pts: usize,
        tracker_max_disappeared: u32,
        tracker_min_confidence: f32,
        tracker_min_appearances: u32,
        tracker_use_point_cloud_voting: bool,
        tracker_moving_speed_threshold: f32,
        kf_process_noise_pos: f64,
        kf_process_noise_vel: f64,
        kf_process_noise_acc: f64,
        kf_process_noise_size: f64,
        kf_measurement_noise_pos: f64,
        kf_measurement_noise_vel: f64,
        kf_measurement_noise_acc: f64,
        kf_measurement_noise_size: f64,
        kf_initial_covariance_scale: f64,
        kf_gate_threshold: f64,
    }

    #[derive(Serialize)]
    struct ClassOutput {
        name: String,
        tp: usize,
        #[serde(rename = "fn")]
        fn_: usize,
        recall: f64,
    }

    fn make_mode_output(stats: &ClassStats, n_gt: usize, n_det: usize) -> ModeOutput {
        ModeOutput {
            n_gt, n_detections: n_det,
            tp: stats.tp, fp: stats.fp, fn_: stats.fn_,
            precision: stats.precision(),
            recall: stats.recall(),
            f1: stats.f1(),
        }
    }

    let output = Output {
        config: ConfigSummary {
            ground_strategy: cfg.ground_strategy.clone(),
            ground_expand: cfg.ground_expand,
            cluster_strategy: cfg.cluster.strategy.clone(),
            cluster_patience: cfg.cluster.merge_patience,
            cluster_voxel_size: cfg.cluster.voxel_size,
            cluster_min_pts: cfg.cluster.min_points_per_cluster,
            denoise_radius: cfg.cluster.denoise_radius,
            denoise_min_pts: cfg.cluster.denoise_min_pts,
            tracker_max_disappeared: cfg.tracker.max_disappeared,
            tracker_min_confidence: cfg.tracker.min_confidence,
            tracker_min_appearances: cfg.tracker.min_appearances,
            tracker_use_point_cloud_voting: cfg.tracker.use_point_cloud_voting,
            tracker_moving_speed_threshold: cfg.tracker.moving_speed_threshold,
            kf_process_noise_pos: cfg.tracker.kf_process_noise_pos,
            kf_process_noise_vel: cfg.tracker.kf_process_noise_vel,
            kf_process_noise_acc: cfg.tracker.kf_process_noise_acc,
            kf_process_noise_size: cfg.tracker.kf_process_noise_size,
            kf_measurement_noise_pos: cfg.tracker.kf_measurement_noise_pos,
            kf_measurement_noise_vel: cfg.tracker.kf_measurement_noise_vel,
            kf_measurement_noise_acc: cfg.tracker.kf_measurement_noise_acc,
            kf_measurement_noise_size: cfg.tracker.kf_measurement_noise_size,
            kf_initial_covariance_scale: cfg.tracker.kf_initial_covariance_scale,
            kf_gate_threshold: cfg.tracker.kf_gate_threshold,
        },
        iou_threshold: args.iou_threshold,
        center_dist: args.center_dist,
        n_frames,
        elapsed_s: total_elapsed,
        all_classes: make_mode_output(&overall_all, total_gt_count, total_det_count_all),
        person_only: make_mode_output(&overall_person, total_person_gt_count, total_det_count_person),
        tp_person,
        tp_nonperson,
        per_class: {
            let mut v: Vec<_> = per_class_all.iter().map(|(name, s)| ClassOutput {
                name: name.clone(),
                tp: s.tp,
                fn_: s.fn_,
                recall: if s.tp + s.fn_ > 0 { s.tp as f64 / (s.tp + s.fn_) as f64 } else { 0.0 },
            }).collect();
            v.sort_by_key(|c| c.tp + c.fn_);
            v.reverse();
            v
        },
    };

    std::fs::write(&json_path, serde_json::to_string_pretty(&output)?)?;
    println!("  JSON → {}", json_path.display());

    // ─── CSV ───────────────────────────────────────────────────────────────
    {
        use std::io::Write;
        let mut f = std::fs::File::create(&csv_path)?;
        writeln!(f, "mode,metric,value")?;

        macro_rules! write_metrics {
            ($mode:expr, $stats:expr, $n_gt:expr, $n_det:expr) => {
                writeln!(f, "{},n_gt,{}", $mode, $n_gt)?;
                writeln!(f, "{},n_detections,{}", $mode, $n_det)?;
                writeln!(f, "{},tp,{}", $mode, $stats.tp)?;
                writeln!(f, "{},fp,{}", $mode, $stats.fp)?;
                writeln!(f, "{},fn,{}", $mode, $stats.fn_)?;
                writeln!(f, "{},precision,{:.4}", $mode, $stats.precision())?;
                writeln!(f, "{},recall,{:.4}", $mode, $stats.recall())?;
                writeln!(f, "{},f1,{:.4}", $mode, $stats.f1())?;
            };
        }

        write_metrics!("all", overall_all, total_gt_count, total_det_count_all);
        write_metrics!("person", overall_person, total_person_gt_count, total_det_count_person);

        writeln!(f, "classification,tp_person,{}", tp_person)?;
        writeln!(f, "classification,tp_nonperson,{}", tp_nonperson)?;

        writeln!(f, "config,iou_threshold,{:.2}", args.iou_threshold)?;
        writeln!(f, "config,center_dist,{:.2}", args.center_dist)?;
        writeln!(f, "config,n_frames,{}", n_frames)?;
        writeln!(f, "config,elapsed_s,{:.1}", total_elapsed)?;
        writeln!(f, "config,ground_strategy,{}", cfg.ground_strategy)?;
        writeln!(f, "config,ground_expand,{:.2}", cfg.ground_expand)?;
        writeln!(f, "config,cluster_strategy,{}", cfg.cluster.strategy)?;
        writeln!(f, "config,cluster_patience,{:.3}", cfg.cluster.merge_patience)?;
        writeln!(f, "config,cluster_voxel_size,{:.2}", cfg.cluster.voxel_size)?;
        writeln!(f, "config,denoise_radius,{:.2}", cfg.cluster.denoise_radius)?;
        writeln!(f, "config,denoise_min_pts,{}", cfg.cluster.denoise_min_pts)?;
    }
    println!("  CSV → {}", csv_path.display());
    println!();
    println!("══════════════════════════════════════════");
    println!("  评估完成");
    println!("══════════════════════════════════════════");

    Ok(())
}
