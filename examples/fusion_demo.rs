//! 点云 + 图像融合可视化演示
//!
//! 对指定数据集的每一帧执行：
//! 1. 图像去畸变 → YOLO 目标检测 → 2D 框（**青色**）
//! 2. 点云距离滤波 → 地面检测 → 聚类 → 3D 框 → 针孔重投影（**品红**）
//! 3. 两者叠加输出到 `output/fusion_demo/<data_dir>/`
//!
//! # 用法
//!
//! ```bash
//! cargo run --example fusion_demo --release                    # 默认 data/cloud
//! cargo run --example fusion_demo --release -- data/labeled    # 标注数据 408 帧
//! ```
//!
//! # 投影模型
//!
//! 外参标定设计为 **原始点云 → 去畸变图像** 的直接投影，因此采用纯针孔模型（无畸变）：
//!
//! ```text
//! P_cam   = cam_from_lidar · P_lidar
//! u       = fx · X_cam / Z_cam + cx
//! v       = fy · Y_cam / Z_cam + cy
//! ```
//!
//! YOLO 也在去畸变图像上检测，保证 2D–3D 空间一致。
//!
//! # 重要发现
//!
//! TOML 以行主序存储矩阵，`Matrix3::from()` / `Matrix4::from()` 直接拷贝数据到
//! nalgebra 列主序存储中，导致有效转置。**必须 `.transpose()` 修正**。
//!
//! Bug: `src/fuse.rs:26` 和 `src/cloud/classify/core.rs:129–130` 缺少此转置，
//! 导致 `cx=0, cy=0` —— 主点偏移约 (328, 209) 像素。

use std::path::{Path, PathBuf};

use image::RgbImage;
use nalgebra::{Matrix3, Matrix4, Vector4};
use pcd_rs::DynReader;
use perple::{
    cloud::{
        classify::cluster::Cluster,
        ground::create_ground_strategy,
        CldBud,
    },
    color::{
        image::{scale_image, image_to_tensor, UndistortMap},
        load_model, ClrBud,
        utils::to_input,
    },
    config::fixif,
    utils::boxes::Box2D,
};

/// 解码 YOLO 输出（注意：该模型 sigmoid 已内置在 ONNX 图中，直接读置信度）
fn decode_yolo(data: &[f32], stride: usize, pad_x: f32, pad_y: f32, scale: f32,
    conf_thresh: f32, nms_thresh: f32, cap: usize) -> Vec<ClrBud>
{
    let scale = if scale <= 0.0 { 1.0 } else { scale };
    let mut cand: Vec<(f32, f32, f32, f32, f32)> = (0..stride)
        .map(|i| {
            let conf = data[4 * stride + i];  // 直读：sigmoid 已内置
            let cx = (data[0 * stride + i] - pad_x) / scale;
            let cy = (data[1 * stride + i] - pad_y) / scale;
            let w = data[2 * stride + i] / scale;
            let h = data[3 * stride + i] / scale;
            (conf, cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)
        })
        .filter(|(c, x1, y1, x2, y2)| *c > conf_thresh && x2 > x1 && y2 > y1)
        .collect();
    cand.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
    if cand.is_empty() { return vec![]; }

    // NMS
    let mut keep: Vec<ClrBud> = Vec::with_capacity(cap.min(cand.len()));
    let mut suppressed = vec![false; cand.len()];
    let label = fixif().person_class_label.clone();
    for i in 0..cand.len() {
        if keep.len() >= cap || suppressed[i] { continue; }
        let (c, x1, y1, x2, y2) = cand[i];
        keep.push(ClrBud { the_box: Box2D { x1, y1, x2, y2 }, class_id: 0, class_name: label.clone(), confidence: c });
        let a = (x2 - x1) * (y2 - y1);
        for j in (i + 1)..cand.len() {
            if suppressed[j] { continue; }
            let (_, jx1, jy1, jx2, jy2) = cand[j];
            let ix = x1.max(jx1); let iy = y1.max(jy1);
            let iw = (x2.min(jx2) - ix).max(0.0);
            let ih = (y2.min(jy2) - iy).max(0.0);
            let inter = iw * ih;
            if inter > 0.0 {
                let jarea = (jx2 - jx1) * (jy2 - jy1);
                if inter / (a + jarea - inter) >= nms_thresh { suppressed[j] = true; }
            }
        }
    }
    keep.truncate(cap);
    keep
}

// ─── 重投影与绘制工具 ────────────────────────────────────────────────────

/// 纯针孔投影（无畸变）—— 用于外参标定下的原始点云 → 去畸变图像
fn pinhole_project(
    cam_from_lidar: &Matrix4<f32>,
    intrinsic: &Matrix3<f32>,
    p: [f32; 3],
) -> Option<(f32, f32)> {
    let cam = cam_from_lidar * Vector4::new(p[0], p[1], p[2], 1.0);
    if cam.z <= 0.0 {
        return None;
    }
    let fx = intrinsic[(0, 0)];
    let fy = intrinsic[(1, 1)];
    let cx = intrinsic[(0, 2)];
    let cy = intrinsic[(1, 2)];
    Some((fx * cam.x / cam.z + cx, fy * cam.y / cam.z + cy))
}

/// 安全钳位像素坐标
fn clamp_px(v: f32, imax: u32) -> Option<u32> {
    if v.is_nan() || v.is_infinite() || v < 0.0 || v >= imax as f32 {
        None
    } else {
        Some(v as u32)
    }
}

/// 绘制矩形框
fn draw_box(rgb: &mut RgbImage, b: &Box2D, color: [u8; 3]) {
    let (w, h) = rgb.dimensions();
    let x1 = b.x1.max(0.0).min(w as f32 - 1.0) as u32;
    let y1 = b.y1.max(0.0).min(h as f32 - 1.0) as u32;
    let x2 = b.x2.max(0.0).min(w as f32 - 1.0) as u32;
    let y2 = b.y2.max(0.0).min(h as f32 - 1.0) as u32;
    let c = image::Rgb(color);
    for px in x1..=x2 {
        if y1 < h {
            rgb.put_pixel(px, y1, c);
        }
        if y2 < h {
            rgb.put_pixel(px, y2, c);
        }
    }
    for py in y1..=y2 {
        if x1 < w {
            rgb.put_pixel(x1, py, c);
        }
        if x2 < w {
            rgb.put_pixel(x2, py, c);
        }
    }
}

/// 绘制十字标记
fn draw_cross(rgb: &mut RgbImage, x: f32, y: f32, color: [u8; 3], size: i32) {
    let (w, h) = rgb.dimensions();
    let Some(ix) = clamp_px(x, w) else { return };
    let Some(iy) = clamp_px(y, h) else { return };
    let c = image::Rgb(color);
    for d in -size..=size {
        let px = ((ix as i32 + d).max(0) as u32).min(w - 1);
        let py = ((iy as i32).max(0) as u32).min(h - 1);
        rgb.put_pixel(px, py, c);
        let px = ((ix as i32).max(0) as u32).min(w - 1);
        let py = ((iy as i32 + d).max(0) as u32).min(h - 1);
        rgb.put_pixel(px, py, c);
    }
}

// ─── 主流程 ───────────────────────────────────────────────────────────────

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cfg = fixif();

    // ── 相机矩阵（TOML 行主序 → nalgebra 列主序，必须转置） ──────────────
    let intrinsic = Matrix3::from(cfg.camera.intrinsic).transpose();
    let cam_from_lidar = Matrix4::from(cfg.camera.extrinsic).transpose();

    // ── 初始化各模块 ──────────────────────────────────────────────────────
    let mut session = load_model(&cfg.model_path)?;
    let mut ground_strat = create_ground_strategy();
    let mut cluster = Cluster::new();

    // ── 解析数据路径 ──────────────────────────────────────────────────────
    let data_dir = std::env::args().nth(1).unwrap_or_else(|| "data/cloud".to_string());
    let data_root = Path::new(&data_dir);

    // 支持两种目录布局：
    //   data/cloud/   → camera/*.jpg, lidar/*.pcd
    //   data/labeled/ → camera/image/*.jpg, lidar/*.pcd
    let camera_dir: PathBuf = if data_root.join("camera/image").exists() {
        data_root.join("camera/image")
    } else {
        data_root.join("camera")
    };
    let pcd_dir = data_root.join("lidar");
    let stem_label = format!("fusion_demo/{}", data_root.file_name().unwrap().to_str().unwrap());
    let out_dir = Path::new("output").join(&stem_label);

    // ── 遍历数据 ──────────────────────────────────────────────────────────

    let mut entries: Vec<_> = std::fs::read_dir(&camera_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| matches!(e.path().extension().and_then(|s| s.to_str()), Some("jpg" | "png")))
        .map(|e| e.path())
        .collect();
    entries.sort();

    std::fs::create_dir_all(&out_dir)?;

    let mut total_yolo = 0usize;
    let mut total_cld = 0usize;

    for entry in &entries {
        let stem = entry.file_stem().unwrap().to_str().unwrap();

        // ── 1. 加载并去畸变图像 ───────────────────────────────────────────
        let img = image::open(entry)?;
        let (img_w, img_h) = (img.width(), img.height());

        let undistorted = if let Some(ref dc) = cfg.camera.dist_coeffs {
            let map = UndistortMap::new(&cfg.camera.intrinsic, dc, img_w, img_h);
            map.apply(&img)
        } else {
            img.clone()
        };
        let mut rgb = undistorted.to_rgb8();

        // ── 2. YOLO 检测 ──────────────────────────────────────────────────
        let (resized, msg) = scale_image(
            &undistorted,
            cfg.default_input_width as u32,
            cfg.default_input_height as u32,
        );
        let arr = image_to_tensor(&resized, cfg.default_input_height, cfg.default_input_width);
        let tensor = to_input(&arr);
        let outputs = session.run(ort::inputs!["images" => tensor])?;
        let extracted = outputs[0].try_extract_tensor::<f32>().unwrap();
        let shape = extracted.0;
        let data = extracted.1;

        let yolo_buds: Vec<ClrBud> = decode_yolo(
            data,
            shape[2] as usize,
            msg.pad_x, msg.pad_y, msg.scale,
            cfg.default_confidence_threshold, cfg.default_nms_threshold, cfg.detections_capacity,
        );
        total_yolo += yolo_buds.len();

        // 绘制 YOLO 框（青色）
        for b in &yolo_buds {
            draw_box(
                &mut rgb,
                &Box2D::new(b.the_box.x1, b.the_box.y1, b.the_box.x2, b.the_box.y2),
                [0, 255, 255],
            );
        }

        // ── 3. 点云管线：滤波 → 地面检测 → 聚类 ───────────────────────────
        let pcd_path = pcd_dir.join(format!("{}.pcd", stem));
        let cld_buds: Vec<CldBud> = if pcd_path.exists() {
            // 3a. 加载 PCD
            let mut reader = DynReader::open(&pcd_path)?;
            let mut points: Vec<[f32; 3]> = Vec::new();
            while let Some(Ok(record)) = reader.next() {
                if let Some(xyz) = record.to_xyz() {
                    points.push(xyz);
                }
            }

            // 3b. 距离滤波
            points.retain(|p| {
                let d = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
                d >= cfg.min_range && d <= cfg.max_range
            });

            if points.is_empty() {
                Vec::new()
            } else {
                // 3c. 地面检测（原地重排：前 n_ground 个为地面点，之后为非地面点）
                let (n_ground, _ground_buds, _plane) = ground_strat.pick(&mut points);
                let nonground = &points[n_ground..];

                if nonground.is_empty() {
                    Vec::new()
                } else {
                    // 3d. 聚类
                    cluster.cluster(nonground);
                    cluster.to_cldbuds()
                }
            }
        } else {
            Vec::new()
        };
        total_cld += cld_buds.len();

        // ── 4. 点云 3D 框 → 2D 针孔重投影（品红绘制） ──────────────────
        for cld in &cld_buds {
            // 按张正友标定法，相机平面后方 (Z_cam < 1) 的物体投影无意义，整框丢弃
            let cam_centroid = cam_from_lidar * Vector4::new(
                cld.centroid[0], cld.centroid[1], cld.centroid[2], 1.0,
            );
            if cam_centroid.z < 1.0 {
                continue;
            }

            let verts = cld.the_box.vertices();
            let mut pts_2d = Vec::new();
            for v in &verts {
                if let Some((u, v_)) =
                    pinhole_project(&cam_from_lidar, &intrinsic, [v.x, v.y, v.z])
                {
                    pts_2d.push((u, v_));
                }
            }
            if pts_2d.len() >= 4 {
                let (min_u, max_u) =
                    pts_2d.iter().fold((f32::MAX, f32::MIN), |(mn, mx), &(u, _)| {
                        (mn.min(u), mx.max(u))
                    });
                let (min_v, max_v) =
                    pts_2d.iter().fold((f32::MAX, f32::MIN), |(mn, mx), &(_, v_)| {
                        (mn.min(v_), mx.max(v_))
                    });
                draw_box(
                    &mut rgb,
                    &Box2D::new(min_u, min_v, max_u, max_v),
                    [255, 0, 255],
                );

                // 质心十字
                if let Some((cu, cv)) =
                    pinhole_project(&cam_from_lidar, &intrinsic, cld.centroid)
                {
                    draw_cross(&mut rgb, cu, cv, [255, 0, 255], 3);
                }
            }
        }

        // ── 5. 保存 ────────────────────────────────────────────────────────
        rgb.save(out_dir.join(format!("{}.jpg", stem)))?;
        println!("{:>6}: YOLO={}, 3D={}", stem, yolo_buds.len(), cld_buds.len());
    }

    println!("\n=== Fusion Demo 完成 ===");
    println!("帧数:     {}", entries.len());
    println!("YOLO 检测: {}", total_yolo);
    println!("3D 聚类:  {}", total_cld);
    println!("输出目录: output/fusion_demo/");

    Ok(())
}
