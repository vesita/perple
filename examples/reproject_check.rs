//! 重投影验证：在图像上同时绘制 YOLO 检测框和 GT 3D 框的 2D 投影
//! 用于检查标定是否准确、YOLO 检测是否与 GT 对齐
use std::path::Path;

use image::RgbImage;
use nalgebra::{Matrix3, Matrix4, Vector4};
use perple::{
    color::{load_model, image_to_tensor, image::scale_image, utils::{to_input, decode_yolo_person}, output::ClrBud},
    config::fixif,
};
use serde::Deserialize;

#[derive(Deserialize)]
struct LabelItem { obj_type: String, psr: LabelPsr }
#[derive(Deserialize)] struct LabelPsr { position: LabelVec3, scale: LabelVec3, rotation: LabelVec3 }
#[derive(Deserialize)] struct LabelVec3 { x: f32, y: f32, z: f32 }

#[derive(Clone, Copy)]
struct Box2D { x1: f32, y1: f32, x2: f32, y2: f32 }

/// OpenCV projectPoints: 3D → 2D (含畸变)
/// 符合 cv2.solvePnP + cv2.projectPoints 的重投影流程
fn project_point(
    cam_from_lidar: &Matrix4<f32>, intrinsic: &Matrix3<f32>,
    dist_coeffs: &[f32; 5], p: [f32; 3],
) -> Option<(f32, f32)> {
    let cam = cam_from_lidar * Vector4::new(p[0], p[1], p[2], 1.0);
    if cam.z <= 0.0 { return None; }
    // 归一化坐标
    let xn = cam.x / cam.z;
    let yn = cam.y / cam.z;
    // OpenCV 径向畸变 + 切向畸变
    let [k1, k2, p1, p2, k3] = *dist_coeffs;
    let r2 = xn * xn + yn * yn;
    let r4 = r2 * r2;
    let r6 = r2 * r4;
    let k = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;
    let xd = xn * k + 2.0 * p1 * xn * yn + p2 * (r2 + 2.0 * xn * xn);
    let yd = yn * k + p1 * (r2 + 2.0 * yn * yn) + 2.0 * p2 * xn * yn;
    let fx = intrinsic[(0, 0)];
    let fy = intrinsic[(1, 1)];
    let cx = intrinsic[(0, 2)];
    let cy = intrinsic[(1, 2)];
    Some((fx * xd + cx, fy * yd + cy))
}

fn clamp_px(v: f32, imax: u32) -> Option<u32> {
    if v.is_nan() || v.is_infinite() || v < 0.0 || v >= imax as f32 { None } else { Some(v as u32) }
}

fn draw_cross(img: &mut RgbImage, x: f32, y: f32, color: image::Rgb<u8>, size: i32) {
    let (w, h) = img.dimensions();
    let Some(ix) = clamp_px(x, w) else { return };
    let Some(iy) = clamp_px(y, h) else { return };
    for d in -size..=size {
        let px = ((ix as i32 + d).max(0) as u32).min(w - 1);
        let py = ((iy as i32).max(0) as u32).min(h - 1);
        img.put_pixel(px, py, color);
        let px = ((ix as i32).max(0) as u32).min(w - 1);
        let py = ((iy as i32 + d).max(0) as u32).min(h - 1);
        img.put_pixel(px, py, color);
    }
}

fn draw_box(img: &mut RgbImage, b: &Box2D, color: image::Rgb<u8>) {
    let (w, h) = img.dimensions();
    let x1 = b.x1.max(0.0).min(w as f32 - 1.0) as u32;
    let y1 = b.y1.max(0.0).min(h as f32 - 1.0) as u32;
    let x2 = b.x2.max(0.0).min(w as f32 - 1.0) as u32;
    let y2 = b.y2.max(0.0).min(h as f32 - 1.0) as u32;
    for px in x1..=x2 {
        if y1 < h { img.put_pixel(px, y1, color); }
        if y2 < h { img.put_pixel(px, y2, color); }
    }
    for py in y1..=y2 {
        if x1 < w { img.put_pixel(x1, py, color); }
        if x2 < w { img.put_pixel(x2, py, color); }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cfg = fixif();
    let intrinsic = Matrix3::from(cfg.camera.intrinsic).transpose();
    let cam_from_lidar = Matrix4::from(cfg.camera.extrinsic).transpose();
    let dist = cfg.camera.dist_coeffs.unwrap_or([0.0; 5]);
    let mut session = load_model(&cfg.model_path)?;

    let img_dir = Path::new("data/person/camera/image");
    let label_dir = Path::new("data/person/label");

    let mut entries: Vec<_> = std::fs::read_dir(img_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| matches!(e.path().extension().and_then(|s| s.to_str()), Some("jpg"|"png")))
        .collect();
    entries.sort_by_key(|e| e.path());

    let out_dir = Path::new("output/reproject_check");
    std::fs::create_dir_all(out_dir)?;

    let mut yolo_total = 0usize;
    let mut gt_total = 0usize;
    let mut in_front = 0usize;
    let mut inside_image = 0usize;

    for entry in &entries {
        let stem = entry.path().file_stem().unwrap().to_str().unwrap().to_string();
        let img = image::open(entry.path())?;
        let mut rgb = img.to_rgb8();

        // ── YOLO 检测 ──────────────────────────────────────────────────
        let (resized, msg) = scale_image(&img, cfg.default_input_width as u32, cfg.default_input_height as u32);
        let arr = image_to_tensor(&resized, cfg.default_input_height, cfg.default_input_width);
        let tensor = to_input(&arr);
        let outputs = session.run(ort::inputs!["images" => tensor])?;
        let extracted = outputs[0].try_extract_tensor::<f32>().unwrap();
        let shape = extracted.0;
        let data = extracted.1;
        let buds: Vec<ClrBud> = decode_yolo_person(
            data, shape[1] as usize, shape[2] as usize,
            msg.pad_x, msg.pad_y, msg.scale,
            cfg.default_confidence_threshold, cfg.default_nms_threshold, cfg.detections_capacity,
        );
        yolo_total += buds.len();

        // 绘制 YOLO 检测框（青色）
        for b in &buds {
            draw_box(&mut rgb, &Box2D { x1: b.the_box.x1, y1: b.the_box.y1, x2: b.the_box.x2, y2: b.the_box.y2 }, image::Rgb([0, 255, 255]));
            // 显示置信度
            let cx = ((b.the_box.x1 + b.the_box.x2) / 2.0) as u32;
            let cy = b.the_box.y1.max(2.0) as u32 - 2;
            let label = format!("{:.2}", b.confidence);
            for (dx, c) in label.chars().enumerate() {
                if cx + dx as u32 * 8 < rgb.width() {
                    // simple pixel text would be complex, skip for now
                }
            }
        }

        // ── 加载 GT 标注 ──────────────────────────────────────────────
        let label_path = label_dir.join(format!("{}.json", stem));
        let n_gt = if label_path.exists() {
            let content = std::fs::read_to_string(&label_path).unwrap_or_default();
            let items: Vec<LabelItem> = serde_json::from_str(&content).unwrap_or_default();
            let n = items.len();
            gt_total += n;

            for item in &items {
                let p = &item.psr.position;
                let (img_w, img_h) = rgb.dimensions();
                // 投影 GT 3D 中心点到 2D 图像
                if let Some((u, v)) = project_point(&cam_from_lidar, &intrinsic, &dist, [p.x, p.y, p.z]) {
                    in_front += 1;
                    let inside = u >= 0.0 && u < img_w as f32 && v >= 0.0 && v < img_h as f32;
                    if inside { inside_image += 1; }
                    // GT 中心投影点：红色十字
                    draw_cross(&mut rgb, u, v, image::Rgb([255, 0, 0]), 5);

                    // 投影 GT 包围盒 8 个顶点 → 2D 框
                    let s = &item.psr.scale;
                    let r = &item.psr.rotation;
                    let half = [s.x / 2.0, s.y / 2.0, s.z / 2.0];
                    let (cr, sr) = (r.z.cos(), r.z.sin());

                    // 8 个顶点 (旋转后的 local → world)
                    let corners = [
                        [-half[0], -half[1], -half[2]], [ half[0], -half[1], -half[2]],
                        [ half[0],  half[1], -half[2]], [-half[0],  half[1], -half[2]],
                        [-half[0], -half[1],  half[2]], [ half[0], -half[1],  half[2]],
                        [ half[0],  half[1],  half[2]], [-half[0],  half[1],  half[2]],
                    ];
                    let mut pts_2d = Vec::new();
                    for c in &corners {
                        let wx = cr * c[0] - sr * c[1] + p.x;
                        let wy = sr * c[0] + cr * c[1] + p.y;
                        let wz = c[2] + p.z;
                        if let Some((pu, pv)) = project_point(&cam_from_lidar, &intrinsic, &dist, [wx, wy, wz]) {
                            pts_2d.push((pu, pv));
                        }
                    }
                    if pts_2d.len() >= 4 {
                        let (min_u, max_u) = pts_2d.iter().fold((f32::MAX, f32::MIN), |(mn, mx), &(u,_)| (mn.min(u), mx.max(u)));
                        let (min_v, max_v) = pts_2d.iter().fold((f32::MAX, f32::MIN), |(mn, mx), &(_,v)| (mn.min(v), mx.max(v)));
                        // GT 投影框：红色虚线
                        draw_box(&mut rgb, &Box2D { x1: min_u, y1: min_v, x2: max_u, y2: max_v }, image::Rgb([255, 100, 100]));
                    }
                }
            }
            n
        } else {
            0
        };

        rgb.save(out_dir.join(format!("{}.jpg", stem)))?;
        println!("  {}: YOLO={}  GT={}", stem, buds.len(), n_gt);
    }

    println!("\n========== 重投影检查 ==========");
    println!("检查帧数: {}", entries.len());
    println!("YOLO 检测: {}", yolo_total);
    println!("GT 标注: {} 人", gt_total);
    println!("GT 投影在相机前方: {}", in_front);
    println!("GT 投影在图像内: {}", inside_image);
    println!("标注图已保存到: output/reproject_check/");

    Ok(())
}
