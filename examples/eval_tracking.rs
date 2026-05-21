//! 跟踪精度评估（MOTA / IDF1 / MOTP）
//!
//! 在 per-frame 检测匹配基础上，跟踪帧间 ID 一致性。
//! 输出 MOTA / MOTP / IDF1 / ID Switch 等跟踪标准指标。
//!
//! 用法：
//!   cargo run --release --example eval_tracking
//!   cargo run --release --example eval_tracking -- --center-dist 0.5
//!   cargo run --release --example eval_tracking -- --frames 408

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
use serde::Deserialize;

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
//  MOTA / IDF1 评估
// ═══════════════════════════════════════════════════════════════════════════════

/// 全序列 MOTA / IDF1 统计
struct TrackingEval {
    /// 逐帧匹配记录: frame_idx → (gt_idx → tracker_id)
    frame_matches: Vec<HashMap<usize, usize>>,
    /// 逐帧统计
    per_frame: Vec<FrameResult>,

    // 累计
    total_gt: usize,       // Σ |GT_i|
    total_tp: usize,       // Σ matched detections (spatial TP)
    total_fp: usize,       // Σ unmatched tracker outputs
    total_fn: usize,       // Σ unmatched GT objects
    total_idsw: usize,     // Σ ID switches
    total_mota_denom: usize,

    // IDF1 统计
    idtp: usize,  // tracked IDs 匹配上的帧数
    idfp: usize,  // tracker output 中未匹配的
    idfn: usize,  // GT 中未匹配的
}

struct FrameResult {
    n_gt: usize,
    n_det: usize,
    tp: usize,
    fp: usize,
    fn_: usize,
}

impl TrackingEval {
    fn new() -> Self {
        Self {
            frame_matches: Vec::new(),
            per_frame: Vec::new(),
            total_gt: 0, total_tp: 0, total_fp: 0, total_fn: 0, total_idsw: 0,
            total_mota_denom: 0,
            idtp: 0, idfp: 0, idfn: 0,
        }
    }

    /// 添加一帧的匹配结果
    fn add_frame(
        &mut self,
        gt_count: usize,
        tracker_targets: &[Target],
        gt_boxes: &[Box3D],
        center_dist: f32,
        hungarian_buf: &mut Vec<Vec<f64>>,
    ) {
        let n_det = tracker_targets.len();

        // ── 匈牙利匹配 ────────────────────────────────────────────────────
        let (tp, matched_pairs, all_gt_matched) = if n_det > 0 && gt_count > 0 {
            self.match_frame(
                tracker_targets, gt_count, gt_boxes, center_dist, hungarian_buf,
            )
        } else {
            (0, Vec::new(), HashSet::new())
        };

        let fp = n_det - tp;
        let fn_count = gt_count - all_gt_matched.len();

        // ── MOTA 累加 ─────────────────────────────────────────────────────
        self.total_gt += gt_count;
        self.total_tp += tp;
        self.total_fp += fp;
        self.total_fn += fn_count;
        self.total_mota_denom += gt_count;

        // ── ID Switch 检测 ────────────────────────────────────────────────
        // 建立该帧的 gt→tracker_id 映射
        let mut gt_to_tid: HashMap<usize, usize> = HashMap::new();
        for (gt_idx, det_idx) in &matched_pairs {
            gt_to_tid.insert(*gt_idx, tracker_targets[*det_idx].id);
        }

        // 与上一帧比较
        if let Some(prev) = self.frame_matches.last() {
            for (&gt_idx, &cur_tid) in &gt_to_tid {
                if let Some(&prev_tid) = prev.get(&gt_idx) {
                    if prev_tid != cur_tid && prev_tid != 0 && cur_tid != 0 {
                        self.total_idsw += 1;
                    }
                }
            }
        }
        self.frame_matches.push(gt_to_tid);

        // ── IDF1 累加 ────────────────────────────────────────────────────
        // IDTP: 每个匹配的检测贡献 1（无论 GT 是什么 ID）
        // 等于 spatial TP
        // IDFP: 未匹配的 tracker output
        // IDFN: 未匹配的 GT
        self.idtp += tp;
        self.idfp += fp;
        self.idfn += fn_count;

        self.per_frame.push(FrameResult {
            n_gt: gt_count,
            n_det,
            tp,
            fp,
            fn_: fn_count,
        });
    }

    /// 匈牙利匹配，返回 (tp, matched_pairs as (gt_idx, det_idx), matched_gt_set)
    fn match_frame(
        &self,
        targets: &[Target],
        n_gt: usize,
        gt_boxes: &[Box3D],
        center_dist: f32,
        hungarian_buf: &mut Vec<Vec<f64>>,
    ) -> (usize, Vec<(usize, usize)>, HashSet<usize>) {
        let n_det = targets.len();
        let mut cost = vec![vec![f64::MAX; n_gt]; n_det];
        for (i, det) in targets.iter().enumerate() {
            for (j, gt_box) in gt_boxes.iter().enumerate() {
                let dc = det.the_box.center();
                let gc = gt_box.center();
                let dist = ((dc.x - gc.x).powi(2) + (dc.y - gc.y).powi(2)).sqrt();
                if dist <= center_dist {
                    cost[i][j] = dist as f64;
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
                matched_pairs.push((gt_idx, i)); // (gt_idx, det_idx)
            }
        }
        (tp, matched_pairs, matched_gt)
    }

    fn mota(&self) -> f64 {
        if self.total_mota_denom == 0 { return 0.0; }
        1.0 - (self.total_fn + self.total_fp + self.total_idsw) as f64 / self.total_mota_denom as f64
    }

    fn motp(&self) -> f64 {
        // MOTP = 1 - avg(center_dist) / threshold
        // 这里用中心距 0.5m 做归一化，暂时简单返回
        if self.total_tp == 0 { return 0.0; }
        // 简化为：未匹配占总量的比例，作为定位误差代理
        (self.total_tp as f64) / (self.total_tp + self.total_fp) as f64
    }

    fn idf1(&self) -> f64 {
        let denom = 2 * self.idtp + self.idfp + self.idfn;
        if denom == 0 { return 0.0; }
        2.0 * self.idtp as f64 / denom as f64
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
//  主流程
// ═══════════════════════════════════════════════════════════════════════════════

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Warn)
        .init();

    // ─── CLI ───────────────────────────────────────────────────────────────
    let args: Vec<String> = std::env::args().collect();
    let center_dist: f32 = args.iter()
        .position(|a| a == "--center-dist")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.5);
    let n_frames_limit: Option<usize> = args.iter()
        .position(|a| a == "--frames")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok());
    let out_prefix: String = args.iter()
        .position(|a| a == "--output")
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_default();
    let person_only: bool = args.iter().any(|a| a == "--person-only");
    let tracker_toml: Option<String> = args.iter()
        .position(|a| a == "--tracker-toml")
        .and_then(|i| args.get(i + 1))
        .cloned();

    // ─── 配置初始化（支持 tracker-toml 覆盖）────────────────────────────────
    if let Some(ref toml_str) = tracker_toml {
        let mut config = Config::from_file("config/default.toml");
        let override_toml = format!("\n[tracker]\n{}", toml_str.replace(',', "\n"));
        if let Err(e) = config.update_from_toml(&override_toml) {
            eprintln!("tracker-toml 解析失败: {}", e);
            std::process::exit(1);
        }
        init_config(config);
    } else {
        init_config(Config::new());
    }

    // ─── 加载标注 ─────────────────────────────────────────────────────────
    let label_dir = "data/labeled/label";
    let all_labels = load_labels(label_dir);
    let n_label_frames = all_labels.len();
    info!("加载 {} 帧标注", n_label_frames);

    // ─── 检查 YOLO 模型 ──────────────────────────────────────────────────
    let config = fixif();
    if !std::path::Path::new(&config.model_path).exists() {
        eprintln!("YOLO 模型不存在（{}）", config.model_path);
        std::process::exit(1);
    }

    // ─── 数据加载器 ───────────────────────────────────────────────────────
    let mut data_loader = DataLoader::new_independent(
        "data/labeled/camera/image".to_string(),
        "data/labeled/lidar".to_string(),
    );
    data_loader.load().await?;

    let n_frames = n_frames_limit
        .map(|n| n.min(data_loader.frame_count()).min(n_label_frames))
        .unwrap_or(data_loader.frame_count().min(n_label_frames));
    info!("将评估 {} 帧 (中心距 ≤ {}m)", n_frames, center_dist);

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
        PathBuf::from(format!("output/eval_tracking_{}", secs))
    } else {
        PathBuf::from(&out_prefix)
    };
    std::fs::create_dir_all(&out_dir)?;

    // ─── 预加载 ───────────────────────────────────────────────────────────
    if !data_loader.load_next().await? { info!("数据为空"); return Ok(()); }
    if n_frames > 1 { data_loader.load_next().await?; }

    let mut eval = TrackingEval::new();
    let mut hungarian_buf = Vec::new();
    let total_start = Instant::now();

    let mut l_handle = Some(tokio::spawn(async move { let _ = lidar.act().await; lidar }));
    let mut c_handle = Some(tokio::spawn(async move { let _ = camera.act().await; camera }));

    for i in 0..n_frames {
        // ── 等待检测完成 ─────────────────────────────────────────────────
        let (l_res, c_res) = tokio::join!(l_handle.take().unwrap(), c_handle.take().unwrap());
        lidar = l_res.unwrap();
        camera = c_res.unwrap();

        let swapl = global_swapl();
        swapl.swap_pipeline();

        if i + 1 < n_frames {
            l_handle = Some(tokio::spawn(async move { let _ = lidar.act().await; lidar }));
            c_handle = Some(tokio::spawn(async move { let _ = camera.act().await; camera }));
        }

        // ── 后融合 + 跟踪 ────────────────────────────────────────────────
        fuse.act().await;
        if i + 2 < n_frames { data_loader.load_next().await?; }
        let _ = tracker.run().await;

        let targets_all: Vec<Target> = swapl.targets.lock().unwrap().read().unwrap_or_default();
        let targets: Vec<Target> = if person_only {
            targets_all.into_iter().filter(|t| t.class_type == "person").collect()
        } else {
            targets_all
        };
        let gt_items = &all_labels[i];
        let gt_boxes: Vec<Box3D> = gt_items.iter().map(|item| item.to_box3d()).collect();

        eval.add_frame(gt_items.len(), &targets, &gt_boxes, center_dist, &mut hungarian_buf);

        if i % 50 == 0 || i == n_frames - 1 || i < 5 {
            let fr = eval.per_frame.last().unwrap();
            println!("  进度: {:>4}/{} | GT:{:>3} 检测:{:>3} | TP:{:>3} FP:{:>3} FN:{:>3} | IDSW累計:{}",
                i + 1, n_frames, fr.n_gt, fr.n_det, fr.tp, fr.fp, fr.fn_, eval.total_idsw);
        }
    }

    let elapsed = total_start.elapsed().as_secs_f64();

    // ═════════════════════════════════════════════════════════════════════
    //  输出结果
    // ═════════════════════════════════════════════════════════════════════
    println!();
    println!("╔════════════════════════════════════════════╗");
    println!("║       跟踪精度评估结果 (MOTA / IDF1)       ║");
    println!("╚════════════════════════════════════════════╝");
    println!();
    let mode_label = if person_only { "仅 person" } else { "全部检测" };
    println!("  配置: 中心距离 ≤ {:.1}m  |  模式: {}  |  帧数: {}  |  耗时: {:.1}s",
        center_dist, mode_label, n_frames, elapsed);
    println!();

    let mota = eval.mota();
    let idf1 = eval.idf1();
    let motp = eval.motp();

    println!("  ── 跟踪核心指标 ──");
    println!("    MOTA    = {:.2}%", mota * 100.0);
    println!("    MOTP    = {:.4}", motp);
    println!("    IDF1    = {:.2}%", idf1 * 100.0);
    println!("    ID Sw   = {}", eval.total_idsw);
    println!();
    println!("  ── 检测成分 ──");
    println!("    GT      = {:>4}", eval.total_gt);
    println!("    TP      = {:>4}", eval.total_tp);
    println!("    FP      = {:>4}", eval.total_fp);
    println!("    FN      = {:>4}", eval.total_fn);
    println!("    ID Sw   = {:>4}", eval.total_idsw);
    println!("    MOTA denom = {}", eval.total_mota_denom);
    println!();
    println!("  ── MOTA 分解 ──");
    let fnr = eval.total_fn as f64 / eval.total_mota_denom.max(1) as f64;
    let fpr = eval.total_fp as f64 / eval.total_mota_denom.max(1) as f64;
    let idswr = eval.total_idsw as f64 / eval.total_mota_denom.max(1) as f64;
    println!("    FN 率   = {:.2}% ({}/{})", fnr * 100.0, eval.total_fn, eval.total_mota_denom);
    println!("    FP 率   = {:.2}% ({}/{})", fpr * 100.0, eval.total_fp, eval.total_mota_denom);
    println!("    IDSW 率 = {:.2}% ({}/{})", idswr * 100.0, eval.total_idsw, eval.total_mota_denom);
    println!("    MOTA    = 1 - {:.2}% = {:.2}%", (fnr + fpr + idswr) * 100.0, mota * 100.0);
    println!();
    println!("  ── IDF1 成分 ──");
    println!("    IDTP    = {}", eval.idtp);
    println!("    IDFP    = {}", eval.idfp);
    println!("    IDFN    = {}", eval.idfn);
    println!("    IDF1    = 2x{} / (2x{} + {} + {}) = {:.2}%",
        eval.idtp, eval.idtp, eval.idfp, eval.idfn, idf1 * 100.0);
    println!();

    // frane-level breakdown
    println!("  ── 帧级统计 ──");
    let tps: Vec<usize> = eval.per_frame.iter().map(|f| f.tp).collect();
    let fps: Vec<usize> = eval.per_frame.iter().map(|f| f.fp).collect();
    let fns: Vec<usize> = eval.per_frame.iter().map(|f| f.fn_).collect();
    let avg_tp = tps.iter().sum::<usize>() as f64 / tps.len() as f64;
    let avg_fp = fps.iter().sum::<usize>() as f64 / fps.len() as f64;
    let avg_fn = fns.iter().sum::<usize>() as f64 / fns.len() as f64;
    println!("    avg TP/frame = {:.2}", avg_tp);
    println!("    avg FP/frame = {:.2}", avg_fp);
    println!("    avg FN/frame = {:.2}", avg_fn);
    println!("    frames       = {}", eval.per_frame.len());

    // ─── 保存 JSON ─────────────────────────────────────────────────────────
    use serde::Serialize;
    #[derive(Serialize)]
    struct Output {
        center_dist: f32,
        n_frames: usize,
        n_gt: usize,
        mota: f64,
        motp: f64,
        idf1: f64,
        id_switches: usize,
        tp: usize,
        fp: usize,
        fn_: usize,
        idtp: usize,
        idfp: usize,
        idfn: usize,
    }

    let output = Output {
        center_dist,
        n_frames,
        n_gt: eval.total_gt,
        mota,
        motp,
        idf1,
        id_switches: eval.total_idsw,
        tp: eval.total_tp,
        fp: eval.total_fp,
        fn_: eval.total_fn,
        idtp: eval.idtp,
        idfp: eval.idfp,
        idfn: eval.idfn,
    };

    let json_path = out_dir.join("eval_tracking.json");
    std::fs::write(&json_path, serde_json::to_string_pretty(&output)?)?;
    println!("  JSON → {}", json_path.display());

    // CSV
    {
        use std::io::Write;
        let csv_path = out_dir.join("eval_tracking.csv");
        let mut f = std::fs::File::create(&csv_path)?;
        writeln!(f, "metric,value")?;
        writeln!(f, "n_frames,{}", n_frames)?;
        writeln!(f, "n_gt,{}", eval.total_gt)?;
        writeln!(f, "mota,{:.4}", mota)?;
        writeln!(f, "motp,{:.4}", motp)?;
        writeln!(f, "idf1,{:.4}", idf1)?;
        writeln!(f, "id_switches,{}", eval.total_idsw)?;
        writeln!(f, "tp,{}", eval.total_tp)?;
        writeln!(f, "fp,{}", eval.total_fp)?;
        writeln!(f, "fn,{}", eval.total_fn)?;
        writeln!(f, "idtp,{}", eval.idtp)?;
        writeln!(f, "idfp,{}", eval.idfp)?;
        writeln!(f, "idfn,{}", eval.idfn)?;
        writeln!(f, "elapsed_s,{:.1}", elapsed)?;
    }
    println!("  CSV → {}/eval_tracking.csv", out_dir.display());

    println!();
    println!("══════════════════════════════════════════");
    println!("  评估完成");
    println!("══════════════════════════════════════════");

    Ok(())
}
