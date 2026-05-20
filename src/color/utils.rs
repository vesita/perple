//! 边界框处理模块
//!
//! 负责处理模型输出，进行坐标转换、置信度过滤和非极大值抑制(NMS)等后处理操作。

use ndarray::Array4;
use ort::session::SessionOutputs;
use ort::value::Tensor;
use ort::value::TensorValueType;
use ort::value::Value;

use crate::color::ClrBud;
use crate::color::image::ScaleMessage;
use crate::config::fixif;
use crate::utils::Box2D;

use image::DynamicImage;

/// YOLO 候选检测框（NMS 前中间表示）
#[derive(Clone)]
struct Candidate {
    confidence: f32,
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
}

/// 解码 YOLO ONNX 输出，只保留 person（class 0）检测，应用 NMS。
///
/// 同时支持单类（1,5,8400）和 COCO 多类（1,84,8400）输出布局。
/// 坐标从 letterbox 填充后的模型空间映射回原图空间。
///
/// # 参数
/// * `data` — 原始张量数据（NCHW 布局，通道连续）
/// * `num_channels` — 输出通道数（5 或 84）
/// * `num_detections` — 检测位置数（8400）
/// * `pad_x` — letterbox 水平填充（模型输入像素）
/// * `pad_y` — letterbox 垂直填充（模型输入像素）
/// * `scale` — letterbox 缩放比
/// * `confidence_threshold` — 置信度阈值
/// * `nms_threshold` — NMS 的 IoU 阈值
/// * `detections_capacity` — 最大输出检测数
pub fn decode_yolo_person(
    data: &[f32],
    num_channels: usize,
    num_detections: usize,
    pad_x: f32,
    pad_y: f32,
    scale: f32,
    confidence_threshold: f32,
    nms_threshold: f32,
    detections_capacity: usize,
) -> Vec<ClrBud> {
    let scale = if scale <= 0.0 { 1.0 } else { scale };
    let stride = num_detections;
    let mut candidates: Vec<Candidate> = Vec::with_capacity(num_detections);

    if num_channels == 5 {
        // 单类: 通道 4 是 person 置信度（YOLO11 ONNX 导出已内置 sigmoid，直接读取）
        for i in 0..num_detections {
            let conf = data[4 * stride + i];
            if conf < confidence_threshold {
                continue;
            }
            let cx = (data[0 * stride + i] - pad_x) / scale;
            let cy = (data[1 * stride + i] - pad_y) / scale;
            let w = data[2 * stride + i] / scale;
            let h = data[3 * stride + i] / scale;
            let x1 = cx - w / 2.0;
            let y1 = cy - h / 2.0;
            let x2 = cx + w / 2.0;
            let y2 = cy + h / 2.0;
            if x2 <= x1 || y2 <= y1 {
                continue;
            }
            candidates.push(Candidate { confidence: conf, x1, y1, x2, y2 });
        }
    } else {
        // 多类: 通道 4 是 class 0 (person) 置信度（YOLO11 ONNX 导出已内置 sigmoid，直接读取）
        for i in 0..num_detections {
            let conf = data[4 * stride + i];
            if conf < confidence_threshold {
                continue;
            }
            let cx = (data[0 * stride + i] - pad_x) / scale;
            let cy = (data[1 * stride + i] - pad_y) / scale;
            let w = data[2 * stride + i] / scale;
            let h = data[3 * stride + i] / scale;
            let x1 = cx - w / 2.0;
            let y1 = cy - h / 2.0;
            let x2 = cx + w / 2.0;
            let y2 = cy + h / 2.0;
            if x2 <= x1 || y2 <= y1 {
                continue;
            }
            candidates.push(Candidate { confidence: conf, x1, y1, x2, y2 });
        }
    }

    // 按置信度降序
    candidates.sort_unstable_by(|a, b| b.confidence.total_cmp(&a.confidence));

    // NMS
    let n = candidates.len();
    let mut suppressed = vec![false; n];
    let mut results = Vec::with_capacity(detections_capacity.min(n));
    let class_name = fixif().person_class_label.clone();

    for ci in 0..n {
        if results.len() >= detections_capacity {
            break;
        }
        if suppressed[ci] {
            continue;
        }

        let c = &candidates[ci];
        results.push(ClrBud {
            the_box: Box2D { x1: c.x1, y1: c.y1, x2: c.x2, y2: c.y2 },
            class_id: 0,
            class_name: class_name.clone(),
            confidence: c.confidence,
        });

        let c_area = (c.x2 - c.x1) * (c.y2 - c.y1);
        for cj in (ci + 1)..n {
            if suppressed[cj] {
                continue;
            }
            let j = &candidates[cj];

            let x_left = c.x1.max(j.x1);
            let y_top = c.y1.max(j.y1);
            let x_right = c.x2.min(j.x2);
            let y_bottom = c.y2.min(j.y2);

            if x_right > x_left && y_bottom > y_top {
                let inter = (x_right - x_left) * (y_bottom - y_top);
                let j_area = (j.x2 - j.x1) * (j.y2 - j.y1);
                if j_area <= 0.0 {
                    suppressed[cj] = true;
                    continue;
                }
                if inter / (c_area + j_area - inter) >= nms_threshold {
                    suppressed[cj] = true;
                }
            }
        }
    }


    results
}

/// 对模型输出应用NMS处理
///
/// 从模型输出中提取检测结果，并应用置信度阈值和NMS阈值进行过滤
///
/// # 参数
/// * `from_model` - 模型输出
/// * `bounds` - 存储检测结果的容器
/// * `message` - 图像缩放信息
/// * `picked_indices` - 用于NMS处理的临时索引数组
/// * `confidence_threshold` - 置信度阈值
/// * `nms_threshold` - NMS阈值
pub fn nms_tensor(
    from_model: &mut SessionOutputs,
    bounds: &mut Vec<ClrBud>,
    message: &ScaleMessage,
    _picked_indices: &mut Vec<bool>,
    confidence_threshold: f32,
    nms_threshold: f32,
) {
    bounds.clear();

    let output_tensor = &mut from_model[0];
    let extracted_tensor = output_tensor
        .try_extract_tensor_mut::<f32>()
        .expect("无法提取张量");
    let shape = extracted_tensor.0;
    let data = extracted_tensor.1;

    let num_detections = shape[2] as usize;
    let num_channels = shape[1] as usize;

    *bounds = decode_yolo_person(
        data,
        num_channels,
        num_detections,
        message.pad_x,
        message.pad_y,
        message.scale,
        confidence_threshold,
        nms_threshold,
        fixif().detections_capacity,
    );
}

/// 计算两个边界框的交集面积
///
/// # 参数
/// * `box1` - 第一个边界框
/// * `box2` - 第二个边界框
///
/// # 返回值
/// 返回交集面积
fn intersection(box1: &Box2D, box2: &Box2D) -> f32 {
    let x_left = box1.x1.max(box2.x1);
    let y_top = box1.y1.max(box2.y1);
    let x_right = box1.x2.min(box2.x2);
    let y_bottom = box1.y2.min(box2.y2);

    if x_right <= x_left || y_bottom <= y_top {
        0.0
    } else {
        (x_right - x_left) * (y_bottom - y_top)
    }
}

/// 计算两个边界框的并集面积
///
/// # 参数
/// * `box1` - 第一个边界框
/// * `box2` - 第二个边界框
///
/// # 返回值
/// 返回并集面积
///
pub fn union(box1: &Box2D, box2: &Box2D) -> f32 {
    let area1 = (box1.x2 - box1.x1) * (box1.y2 - box1.y1);
    let area2 = (box2.x2 - box2.x1) * (box2.y2 - box2.y1);
    area1 + area2 - intersection(box1, box2)
}

/// 在图像上绘制检测结果
///
/// # 参数
/// * `image` - 原始图像
/// * `detections` - 检测结果
///
/// # 返回值
/// 返回绘制了检测框的图像
pub fn draw_detections(image: &DynamicImage, detections: &[ClrBud]) -> DynamicImage {
    let mut rgb = image.to_rgb8();
    let (w, h) = rgb.dimensions();

    for det in detections {
        let bbox = &det.the_box;
        let x1 = bbox.x1.max(0.0).min(w as f32 - 1.0) as u32;
        let y1 = bbox.y1.max(0.0).min(h as f32 - 1.0) as u32;
        let x2 = bbox.x2.max(0.0).min(w as f32 - 1.0) as u32;
        let y2 = bbox.y2.max(0.0).min(h as f32 - 1.0) as u32;

        let color = match det.class_id {
            0 => image::Rgb([0, 255, 255]), // 青色 — person
            _ => image::Rgb([255, 0, 0]),   // 红色 — 其他
        };

        // 四条边：各 2 像素宽
        for px in x1..=x2 {
            draw_pixel(&mut rgb, px, y1, w, h, &color); // 上
            draw_pixel(&mut rgb, px, y1.saturating_add(1), w, h, &color);
            draw_pixel(&mut rgb, px, y2, w, h, &color); // 下
            draw_pixel(&mut rgb, px, y2.saturating_sub(1), w, h, &color);
        }
        for py in y1..=y2 {
            draw_pixel(&mut rgb, x1, py, w, h, &color); // 左
            draw_pixel(&mut rgb, x1.saturating_add(1), py, w, h, &color);
            draw_pixel(&mut rgb, x2, py, w, h, &color); // 右
            draw_pixel(&mut rgb, x2.saturating_sub(1), py, w, h, &color);
        }
    }

    DynamicImage::ImageRgb8(rgb)
}

#[inline]
fn draw_pixel(img: &mut image::RgbImage, x: u32, y: u32, w: u32, h: u32, color: &image::Rgb<u8>) {
    if x < w && y < h {
        img.put_pixel(x, y, *color);
    }
}

/// 将ndarray数组转换为ONNX Runtime张量
///
/// # 参数
/// * `mats` - 四维数组，形状为(1, 3, height, width)
///
/// # 返回值
/// 返回对应的ONNX Runtime张量
pub fn to_input(mats: &Array4<f32>) -> Value<TensorValueType<f32>> {
    let shape: Vec<usize> = mats.shape().to_vec();
    let (data, _offset) = mats.clone().into_raw_vec_and_offset();
    let result = Tensor::from_array(([shape[0], shape[1], shape[2], shape[3]], data)).unwrap();
    result
}
