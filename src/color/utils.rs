//! 边界框处理模块
//! 
//! 负责处理模型输出，进行坐标转换、置信度过滤和非极大值抑制(NMS)等后处理操作。

use image::GenericImageView;
use ndarray::Array2;
use ndarray::Axis;
use ort::session::SessionOutputs;

use crate::color::bounds::BoundingBox;
use crate::color::bounds::Bounds;
use crate::color::bounds::Detection;
use crate::color::image::ScaleMessage;
use crate::config::DETECTIONS_CAPACITY;
use crate::config::PERSON_CLASS_LABEL;
use crate::utils::sort::group_sort;
use crate::utils::sort::group_sort_by;

use image::DynamicImage;
use raqote::{DrawOptions, DrawTarget, LineJoin, PathBuilder, SolidSource, Source, StrokeStyle};


/// 处理模型输出，应用置信度和NMS阈值
/// 
/// 对模型输出进行后处理，包括坐标转换、置信度过滤和非极大值抑制。
/// 
/// # 参数
/// * `output` - 模型输出，形状为(num_boxes, 5)
/// * `img_width` - 原始图像宽度
/// * `img_height` - 原始图像高度
/// * `input_width` - 模型输入宽度
/// * `input_height` - 模型输入高度
/// * `confidence_threshold` - 置信度阈值
/// * `nms_threshold` - NMS阈值
/// 
/// # 返回值
/// 返回处理后的检测结果列表
/// 
/// # 示例
/// 
/// ```
/// use ndarray::Array2;
/// use perple::color::posts::process_detections;
/// 
/// let output = Array2::<f32>::zeros((10, 5)); // 示例输出
/// let detections = process_detections(
///     output,
///     1920.0,  // 原始图像宽度
///     1080.0,  // 原始图像高度
///     640,     // 模型输入宽度
///     640,     // 模型输入高度
///     0.5,     // 置信度阈值
///     0.7      // NMS阈值
/// );
/// ```
pub fn process_detections(
    output: Array2<f32>,
    img_width: f32,
    img_height: f32,
    input_width: usize,
    input_height: usize,
    confidence_threshold: f32,
    nms_threshold: f32,
) -> Vec<Detection> {
    let mut detections = Vec::new();
    
    // 预分配容量以减少重新分配
    detections.reserve(output.len_of(Axis(0)));
    
    for row in output.axis_iter(Axis(0)) {
        let row_slice = row.as_slice().expect("Row should be contiguous");
        // 对于只有一个人物检测类别的情况，直接获取置信度
        let prob = row_slice[4]; // 第5个元素是person类别的置信度
            
        if prob < confidence_threshold {
            continue;
        }
        // YOLO模型输出的是相对于输入图像尺寸的坐标 (640x640)
        // 需要将其转换为相对于原始图像尺寸的坐标
        let x1 = row[0];  // 左上角x坐标 (相对于640)
        let y1 = row[1];  // 左上角y坐标 (相对于640)
        let x2 = row[2];   // 右下角x坐标 (相对于640)
        let y2 = row[3];   // 右下角y坐标 (相对于640)

        // 转换为相对于原始图像的坐标
        let scale_x = img_width / input_width as f32;
        let scale_y = img_height / input_height as f32;

        let s_x1 = x1 * scale_x;
        let s_y1 = y1 * scale_y;
        let s_x2 = x2 * scale_x;
        let s_y2 = y2 * scale_y;

        detections.push(Detection {
            bbox: BoundingBox {
                x1: s_x1,
                y1: s_y1,
                x2: s_x2,
                y2: s_y2
            },
            class_id: 0, // 只有一个类别，ID为0
            class_name: PERSON_CLASS_LABEL.to_string(),
            confidence: prob
        });
    }

    // 使用 unstable_sort 提升排序性能，因为我们不关心相等元素的顺序
    detections.sort_unstable_by(|a, b| b.confidence.total_cmp(&a.confidence));
    
    // 应用非极大值抑制(NMS)
    apply_nms(&mut detections, nms_threshold)
}

pub fn to_bounds(
    output: &SessionOutputs,
    message: &ScaleMessage,
    confidence_threshold: f32,
    nms_threshold: f32,
) -> Vec<Detection> {
    let mut detections = Vec::new();
    let (img_width, img_height) = (message.o_width as f32, message.o_height as f32);
    let (input_width, input_height) = (message.s_width, message.s_height);
    
    // 从SessionOutputs中直接提取张量数据
    let output_tensor = &output[0];
    let extracted_tensor = output_tensor.try_extract_tensor::<f32>().expect("无法提取张量");
    let shape = extracted_tensor.0.clone();
    let data = extracted_tensor.1; // 直接使用引用，避免to_vec()的内存复制
    
    // 直接处理原始数据，绕过Array2中间环节
    let num_boxes = shape[1] as usize;
    let num_params = shape[2] as usize;
    
    // 遍历每个检测框
    for i in 0..num_boxes {
        // 计算当前框在数据中的起始索引（只取前5列数据）
        let start_index = i * num_params;
        
        // 提取当前框的数据
        let x1 = data[start_index];
        let y1 = data[start_index + 1];
        let x2 = data[start_index + 2];
        let y2 = data[start_index + 3];
        let confidence = data[start_index + 4];
        
        // 置信度过滤
        if confidence < confidence_threshold {
            continue;
        }
        
        // 转换为相对于原始图像的坐标
        let scale_x = img_width / input_width as f32;
        let scale_y = img_height / input_height as f32;
        
        let scaled_x1 = x1 * scale_x;
        let scaled_y1 = y1 * scale_y;
        let scaled_x2 = x2 * scale_x;
        let scaled_y2 = y2 * scale_y;
        
        detections.push(Detection {
            bbox: BoundingBox {
                x1: scaled_x1,
                y1: scaled_y1,
                x2: scaled_x2,
                y2: scaled_y2,
            },
            class_id: 0,
            class_name: PERSON_CLASS_LABEL.to_string(),
            confidence,
        });
    }
    
    // 按置信度排序
    detections.sort_by(|a, b| b.confidence.total_cmp(&a.confidence));
    
    // 应用非极大值抑制(NMS)
    apply_nms(&mut detections, nms_threshold)
}


/// 应用非极大值抑制
/// 
/// 去除重叠度高的重复检测框，只保留置信度最高的框。
/// 
/// # 参数
/// * `detections` - 检测结果列表（会被修改）
/// * `nms_threshold` - NMS阈值
/// 
/// # 返回值
/// 返回应用NMS后的检测结果列表
fn apply_nms(detections: &mut Vec<Detection>, nms_threshold: f32) -> Vec<Detection> {
    let mut result = Vec::new();
    let mut picked_indices = vec![false; detections.len()];

    for i in 0..detections.len() {
        if picked_indices[i] {
            continue;
        }
        
        result.push(detections[i].clone());
        
        // 缓存当前检测框的面积以避免重复计算
        let area_i = (detections[i].bbox.x2 - detections[i].bbox.x1) * (detections[i].bbox.y2 - detections[i].bbox.y1);
        
        // 提前检查，如果框的面积为0，则跳过
        if area_i <= 0.0 {
            picked_indices[i] = true;
            continue;
        }
        
        for j in (i + 1)..detections.len() {
            if picked_indices[j] {
                continue;
            }
            
            let area_j = (detections[j].bbox.x2 - detections[j].bbox.x1) * (detections[j].bbox.y2 - detections[j].bbox.y1);
            
            // 提前检查，如果框的面积为0，则跳过
            if area_j <= 0.0 {
                picked_indices[j] = true;
                continue;
            }
            
            let inter = intersection(&detections[i].bbox, &detections[j].bbox);
            // 如果交集为0，直接跳过
            if inter == 0.0 {
                continue;
            }
            
            let union_area = area_i + area_j - inter;
            let iou = inter / union_area;
            if iou >= nms_threshold {
                picked_indices[j] = true;
            }
        }
    }

    result
}

pub fn nms_tensor(
    from_model: &mut SessionOutputs,
    bounds: &mut Bounds,
    message: &ScaleMessage,
    picked_indices: &mut [bool; DETECTIONS_CAPACITY],
    confidence_threshold: f32,
    nms_threshold: f32,
) {
    bounds.clear();
    
    let (img_width, img_height) = (message.o_width as f32, message.o_height as f32);
    let (input_width, input_height) = (message.s_width, message.s_height);
    let width_scale = img_width / input_width as f32;
    let height_scale = img_height / input_height as f32;

    // 从SessionOutputs中直接提取张量数据
    let output_tensor = &mut from_model[0];
    let extracted_tensor = output_tensor.try_extract_tensor_mut::<f32>().expect("无法提取张量");
    let shape = extracted_tensor.0;
    let mut data = extracted_tensor.1; // 直接使用引用，避免to_vec()的内存复制
    // 直接处理原始数据，绕过Array2中间环节
    let num_boxes = shape[1] as usize;
    let num_params = shape[2] as usize;
    
    // 按置信度排序，将置信度高的框排在前面
    group_sort_by(&mut data, num_params, 4, |a, b| 
        b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    // 初始化picked_indices数组，但不超过DETECTIONS_CAPACITY的大小
    picked_indices.fill(false);

    // NMS处理
    for i in 0..num_boxes.min(DETECTIONS_CAPACITY) {
        // 如果当前框已经被抑制，则跳过
        if picked_indices[i] {
            continue;
        }

        let i_start = i * num_params;
        let i_confidence = data[i_start + 4];
        
        // 置信度过滤
        if i_confidence < confidence_threshold {
            picked_indices[i] = true;
            continue;
        }

        // 计算当前框的坐标和面积
        let i_x1 = data[i_start];
        let i_y1 = data[i_start + 1];
        let i_x2 = data[i_start + 2];
        let i_y2 = data[i_start + 3];
        let i_area = (i_x2 - i_x1) * (i_y2 - i_y1);

        // 如果面积为0，标记为已选择并跳过
        if i_area <= 0.0 {
            picked_indices[i] = true;
            continue;
        }

        // 将未被抑制的边界框添加到bounds中
        bounds.push(Detection {
            bbox: BoundingBox {
                x1: i_x1 * width_scale,
                y1: i_y1 * height_scale,
                x2: i_x2 * width_scale,
                y2: i_y2 * height_scale,
            },
            class_id: 0,
            class_name: PERSON_CLASS_LABEL.to_string(),
            confidence: i_confidence,
        });

        // 检查后续的框是否与当前框重叠过多
        for j in (i + 1)..num_boxes.min(DETECTIONS_CAPACITY) {
            if picked_indices[j] {
                continue;
            }

            let j_start = j * num_params;
            let j_confidence = data[j_start + 4];
            
            // 提前进行置信度过滤
            if j_confidence < confidence_threshold {
                picked_indices[j] = true;
                continue;
            }

            let j_x1 = data[j_start];
            let j_y1 = data[j_start + 1];
            let j_x2 = data[j_start + 2];
            let j_y2 = data[j_start + 3];
            
            // 计算交集区域
            let x_left = i_x1.max(j_x1);
            let y_top = i_y1.max(j_y1);
            let x_right = i_x2.min(j_x2);
            let y_bottom = i_y2.min(j_y2);

            // 如果有交集
            if x_right > x_left && y_bottom > y_top {
                let inter_area = (x_right - x_left) * (y_bottom - y_top);
                let j_area = (j_x2 - j_x1) * (j_y2 - j_y1);
                
                // 如果任一框面积为0则跳过
                if j_area <= 0.0 {
                    picked_indices[j] = true;
                    continue;
                }
                
                let union_area = i_area + j_area - inter_area;
                let iou = inter_area / union_area;

                // 如果IOU超过阈值，则抑制这个框
                if iou >= nms_threshold {
                    picked_indices[j] = true;
                }
            }
        }
    }
}

/// 计算两个边界框的交集面积
/// 
/// # 参数
/// * `box1` - 第一个边界框
/// * `box2` - 第二个边界框
/// 
/// # 返回值
/// 返回交集面积
fn intersection(box1: &BoundingBox, box2: &BoundingBox) -> f32 {
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
fn union(box1: &BoundingBox, box2: &BoundingBox) -> f32 {
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
pub fn draw_detections(image: &DynamicImage, detections: &[Detection]) -> DynamicImage {
    let (img_width, img_height) = image.dimensions();
    let mut dt = DrawTarget::new(img_width as i32, img_height as i32);
    
    // 将原始图像绘制到DrawTarget上
    let rgba_image = image.to_rgba8();
    let image_data: Vec<u32> = rgba_image.chunks(4).map(|pixel| {
        let b = pixel[2];
        let g = pixel[1];
        let r = pixel[0];
        let a = pixel[3];
        u32::from_le_bytes([b, g, r, a])
    }).collect();
    
    let img = raqote::Image {
        width: img_width as i32,
        height: img_height as i32,
        data: &image_data,
    };
    
    dt.draw_image_at(0.0, 0.0, &img, &DrawOptions::new());

    for detection in detections {
        let bbox = &detection.bbox;

        let mut pb = PathBuilder::new();
        let width = bbox.x2 - bbox.x1;
        let height = bbox.y2 - bbox.y1;

        pb.rect(bbox.x1, bbox.y1, width, height);
        let path = pb.finish();
        
        // 根据类别设置不同颜色
        let color = match detection.class_id {
            0 => SolidSource { r: 0x00, g: 0xFF, b: 0xFF, a: 0xFF }, // 青色 - person类别
            _ => SolidSource { r: 0xFF, g: 0x00, b: 0x00, a: 0xFF }, // 红色 - 其他类别
        };
        
        dt.stroke(
            &path,
            &Source::Solid(color),
            &StrokeStyle {
                join: LineJoin::Round,
                width: 2.0,
                ..StrokeStyle::default()
            },
            &DrawOptions::default()
        );
        
        // 可以添加文本标签显示类别和置信度
        // 这里暂时省略，如需要可后续添加
    }

    // 将DrawTarget转换回图像
    let pixels: Vec<u8> = dt.get_data().iter().flat_map(|&pixel| {
        let bytes = pixel.to_le_bytes();
        vec![bytes[2], bytes[1], bytes[0], bytes[3]] // BGRA to RGBA
    }).collect();
    
    DynamicImage::ImageRgba8(
        image::ImageBuffer::from_raw(img_width, img_height, pixels)
            .expect("Failed to create image from rendered data")
    )
}