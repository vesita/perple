//! ONNX 模型量化对比测试
//!
//! 并行推理三种量化模型，将检测结果（标注图）保存到 output/yolo/ 各变体目录下。
//!
//! 用法：
//!   cargo run --example model_bench

use std::path::Path;
use std::time::Instant;
use std::thread;

use image::DynamicImage;
use perple::{
    color::{
        load_model, image_to_tensor,
        image::scale_image,
        utils::{to_input, decode_yolo_person, draw_detections},
        output::ClrBud,
    },
    config::fixif,
};

struct BenchResult {
    name: String,
    file_mb: f64,
    total_dets: usize,
    elapsed_ms: f64,
    error: Option<String>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ── 加载测试图像 ──────────────────────────────────────────────────
    let img_dir = Path::new("data/cloud/camera");
    let mut entries: Vec<_> = std::fs::read_dir(img_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| {
            let p = e.path();
            let ext = p.extension().and_then(|s| s.to_str());
            matches!(ext, Some("jpg" | "png"))
        })
        .skip(100) // 前 100 帧是空对象
        .take(50)
        .collect();
    entries.sort_by_key(|e| e.path());
    let n = entries.len();

    let images: Vec<(String, DynamicImage)> = entries
        .into_iter()
        .map(|e| {
            let name = e.file_name().to_string_lossy().to_string();
            (name, image::open(e.path()).unwrap())
        })
        .collect();

    println!("测试图像: {} 张\n", n);

    let out_root = Path::new("output/yolo");

    // ── 模型变体 ──────────────────────────────────────────────────────
    let variants = [
        ("FP32", "model/quantized/yolo11n.onnx".to_string()),
        ("FP16", "model/quantized/yolo11n_fp16.onnx".to_string()),
        ("INT8", "model/quantized/yolo11n_int8.onnx".to_string()),
    ];

    // ── 并行推理 ──────────────────────────────────────────────────────
    let mut handles = vec![];

    for (name, path) in &variants {
        let images = images.clone();
        let out_dir = out_root.join(name.to_lowercase());
        let name = name.to_string();
        let path = path.clone();

        handles.push(thread::spawn(move || {
            let p = Path::new(&path);
            if !p.exists() {
                return BenchResult {
                    name,
                    file_mb: 0.0,
                    total_dets: 0,
                    elapsed_ms: 0.0,
                    error: Some("文件不存在".into()),
                };
            }
            let file_mb = p.metadata().unwrap().len() as f64 / 1_000_000.0;

            let mut session = match load_model(&path) {
                Ok(s) => s,
                Err(e) => {
                    return BenchResult {
                        name: name.clone(),
                        file_mb,
                        total_dets: 0,
                        elapsed_ms: 0.0,
                        error: Some(format!("加载失败: {e}")),
                    };
                }
            };

            std::fs::create_dir_all(&out_dir).ok();

            let mut total_dets = 0usize;
            let t0 = Instant::now();
            let mut buds: Vec<ClrBud> = Vec::new();
            let mut any_error = None;

            for (fname, img) in &images {
                let (resized, msg) = scale_image(img, 640, 640);
                let arr = image_to_tensor(&resized, 640, 640);
                let tensor = to_input(&arr);

                let run_result = session.run(ort::inputs!["images" => tensor]);
                let outputs = match run_result {
                    Ok(o) => o,
                    Err(e) => {
                        any_error = Some(format!("推理失败: {e}"));
                        break;
                    }
                };

                buds.clear();
                {
                    let extracted = outputs[0].try_extract_tensor::<f32>()
                        .expect("无法提取张量");
                    let shape = extracted.0;
                    let data = extracted.1;
                    let num_detections = shape[2] as usize;
                    let num_channels = shape[1] as usize;
                    let cfg = fixif();

                    buds = decode_yolo_person(
                        data, num_channels, num_detections,
                        msg.pad_x, msg.pad_y, msg.scale,
                        cfg.default_confidence_threshold,
                        cfg.default_nms_threshold,
                        cfg.detections_capacity,
                    );
                }

                total_dets += buds.len();

                if !buds.is_empty() {
                    let annotated = draw_detections(img, &buds);
                    let _ = annotated.save(out_dir.join(fname));
                }
            }

            BenchResult {
                name,
                file_mb,
                total_dets,
                elapsed_ms: t0.elapsed().as_secs_f64() * 1000.0,
                error: any_error,
            }
        }));
    }

    // ── 输出结果 ──────────────────────────────────────────────────────
    println!("{:<6} {:>8} {:>12} {:>10} {:>8} {:>10}  {}",
        "模型", "大小", "检测总数", "总耗时", "平均/帧", "帧率", "输出目录");
    println!("{}", "-".repeat(80));

    for handle in handles {
        let r = handle.join().unwrap();
        let avg_ms = if r.elapsed_ms > 0.0 {
            r.elapsed_ms / n as f64
        } else {
            0.0
        };
        let fps = if r.elapsed_ms > 0.0 {
            n as f64 / (r.elapsed_ms / 1000.0)
        } else {
            0.0
        };

        let out_path = out_root.join(r.name.to_lowercase());

        if let Some(err) = &r.error {
            println!("{:<6} {:>6.1}MB {:>12} {:>10} {:>8} {:>10}  {}",
                r.name, r.file_mb, "不支持", "-", "-", "-", out_path.display());
            eprintln!("  {err}");
        } else {
            println!("{:<6} {:>6.1}MB {:>8} {:>8.0}ms {:>7.1}ms {:>7.1}  {}",
                r.name, r.file_mb, r.total_dets, r.elapsed_ms, avg_ms, fps,
                out_path.display());
        }
    }

    Ok(())
}
