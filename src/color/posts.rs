use ndarray::Array2;
use ndarray::Axis;

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
    
    for row in output.axis_iter(Axis(0)) {
        let row: Vec<_> = row.iter().copied().collect();
        // 对于只有一个人物检测类别的情况，直接获取置信度
        let prob = row[4]; // 第5个元素是person类别的置信度
            
        if prob < confidence_threshold {
            continue;
        }
        
        let label = PERSON_CLASS_LABEL.to_string();
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
            class_name: label,
            confidence: prob
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
        
        for j in (i + 1)..detections.len() {
            if picked_indices[j] {
                continue;
            }
            
            let iou = intersection(&detections[i].bbox, &detections[j].bbox) / union(&detections[i].bbox, &detections[j].bbox);
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
    let width = box1.x2.min(box2.x2);
    let height = box1.y2.min(box2.y2);

    if width <= 0.0 || height <= 0.0 {
        0.0
    } else {
        width * height
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
    let area1 = box1.x2 * box1.y2;
    let area2 = box2.x2 * box2.y2;
    area1 + area2 - intersection(box1, box2)
}