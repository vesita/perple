//! 边界框处理模块
//! 
//! 负责处理模型输出，进行坐标转换、置信度过滤和非极大值抑制(NMS)等后处理操作。

use ndarray::Array2;
use ndarray::Axis;
use ort::session::SessionOutputs;

use crate::color::image::ScaleMessage;

/// 边界框结构
/// 
/// 表示一个矩形边界框，用于包围检测到的目标。
#[derive(Debug, Clone, Copy)]
pub struct BoundingBox {
    /// 左上角x坐标
    pub x1: f32,
    /// 左上角y坐标
    pub y1: f32,
    /// 右下角x坐标
    pub x2: f32,
    /// 右下角y坐标
    pub y2: f32,
}

/// 检测结果结构
/// 
/// 包含检测到的目标的完整信息。
#[derive(Debug, Clone)]
pub struct Detection {
    /// 目标的边界框
    pub bbox: BoundingBox,
    /// 类别ID
    pub class_id: usize,
    /// 类别名称
    pub class_name: String,
    /// 置信度
    pub confidence: f32,
}

const PERSON_CLASS_LABEL: &str = "person";

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