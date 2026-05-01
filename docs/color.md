# Color 模块文档

## 概述

Color 模块是 Perple 项目中负责图像目标检测的核心模块。它基于 YOLO(You Only Look Once) 模型对输入图像进行目标检测，输出检测到的对象边界框及相关信息。

**核心功能**:

- ✅ ONNX(Open Neural Network Exchange) 模型加载和推理
- ✅ 图像预处理（缩放、归一化、颜色空间转换）
- ✅ YOLO 检测器实现（置信度阈值、NMS 后处理）
- ✅ 坐标转换（模型坐标 → 图像坐标）
- ✅ 检测结果可视化

## 模块架构

### 核心组件

**源代码位置**: [`src/color/`](../src/color/)

```
color 模块
├── model.rs      # ONNX 模型加载
├── image.rs      # 图像预处理和加载
├── detect.rs     # YOLO 检测器实现
├── output.rs     # 检测结果输出容器 (ClrBud)
├── utils.rs      # 可视化和后处理工具
├── core.rs       # 核心检测逻辑
└── look.rs       # 视觉分析功能
```

### 主要结构体

#### YoloDetector

`YoloDetector` 是核心的检测器类，负责：

1. 加载和管理 ONNX(Open Neural Network Exchange) 模型
2. 设置置信度 (Confidence) 和非极大值抑制 (NMS, Non-Maximum Suppression) 阈值
3. 执行模型推理（使用 ONNX Runtime）
4. 调用后处理函数（坐标转换、NMS）

使用示例：

```rust
use perple::color::{YoloDetector, load_model, load_image};

let model = load_model("path/to/model.onnx")?;
let image = load_image("path/to/image.jpg")?;

let mut detector = YoloDetector::new(640, 640)
    .with_confidence_threshold(0.5)
    .with_nms_threshold(0.7);

let detections = detector.detect(&image)?;
```

#### ClrBud

`ClrBud` 是固定容量的检测结果容器，具有以下特点：

- 预分配固定容量（默认 16），避免动态内存分配
- 提供类似 Vec 的操作接口（push, clear, len, is_empty 等）
- 支持迭代器访问（IntoIterator, IntoRefIterator）
- 存储 `Detection` 对象
- 实现 `Stream` trait，支持流式处理

**性能优势**:

- 🚀 零堆分配（在容量范围内）
- 🚀 缓存友好（连续内存布局）
- 🚀 适用于实时系统（可预测的性能）

#### Detection

`Detection` 表示一个检测到的对象，包含：

- `bbox`: 边界框信息 (`Box2D`)
- `class_id`: 类别 ID（YOLO 模型输出的整数索引）
- `class_name`: 类别名称（如 "person", "car" 等）
- `confidence`: 置信度分数（0.0 ~ 1.0）

**注意**: 类别名称需要通过配置文件或模型元数据映射。

#### Box2D

`Box2D` 表示一个二维边界框，包含：

- `x1`, `y1`: 左上角坐标
- `x2`, `y2`: 右下角坐标
- 提供计算宽度、高度、面积等辅助方法

## 模型输出格式

### YOLO 模型输出结构

YOLO 模型的输出是一个三维张量，形状为 `[1, num_boxes, num_params]`：

- `1`: 批次大小（目前只处理单张图片）
- `num_boxes`: 检测框的数量（取决于模型输出，通常为 8400 或更多）
- `num_params`: 每个检测框的参数数量，通常为 5 个参数（4 个坐标 + 1 个置信度）

**注意**: 某些 YOLO 模型可能输出更多参数（如类别概率），当前实现仅使用坐标和置信度。

### 检测框参数格式

每个检测框包含 5 个参数：

1. `x1`: 边界框左上角 x 坐标（相对于模型输入尺寸）
2. `y1`: 边界框左上角 y 坐标（相对于模型输入尺寸）
3. `x2`: 边界框右下角 x 坐标（相对于模型输入尺寸）
4. `y2`: 边界框右下角 y 坐标（相对于模型输入尺寸）
5. `confidence`: 置信度分数

**注意**: 坐标是相对于模型输入尺寸（默认 640x640）的，需要进行坐标转换才能映射到原始图像。

## 检测结果解析机制

### 坐标转换

由于模型输出的坐标是相对于模型输入尺寸（默认 640x640）的，需要将其转换为相对于原始图像的坐标：

```rust
let scale_x = img_width / input_width as f32;
let scale_y = img_height / input_height as f32;

let scaled_x1 = x1 * scale_x;
let scaled_y1 = y1 * scale_y;
let scaled_x2 = x2 * scale_x;
let scaled_y2 = y2 * scale_y;
```

**注意**:

- 如果图像不是正方形，X 和 Y 方向的缩放比例不同
- 转换后的坐标可能超出图像边界，需要裁剪（当前未实现）

### 置信度过滤

在处理检测结果时，会根据设定的置信度阈值（默认 0.5）过滤掉低置信度的检测框：

```rust
if confidence < confidence_threshold {
    continue;
}
```

**调优建议**:

- **高精度场景**: 提高阈值（0.7~0.9），减少误检
- **高召回率场景**: 降低阈值（0.3~0.5），减少漏检

### 非极大值抑制（NMS, Non-Maximum Suppression）

为了去除重叠度高的重复检测框，系统使用 NMS(Non-Maximum Suppression) 算法：

**算法步骤**:

1. 按置信度对检测框进行排序（降序）
2. 依次处理每个检测框：
   - 如果当前框未被抑制且置信度高于阈值，则保留该框
   - 计算该框与后续所有框的 IoU(Intersection over Union, 交并比)
   - 如果 IoU 超过设定阈值（默认 0.7），则抑制相应框

**IoU(Intersection over Union, 交并比) 计算**:

```rust
let intersection_area = (min(box1.x2, box2.x2) - max(box1.x1, box2.x1))
    * (min(box1.y2, box2.y2) - max(box1.y1, box2.y1));
let union_area = area1 + area2 - intersection_area;
let iou = intersection_area / union_area;
```

## 工作流程

完整的检测流程如下：

```rust
use perple::color::{load_model, load_image, YoloDetector};

// 1. 加载 ONNX 模型
let model = load_model("path/to/model.onnx")?;

// 2. 加载待检测图像
let image = load_image("path/to/image.jpg")?;

// 3. 创建检测器实例并配置参数
let mut detector = YoloDetector::new(640, 640)
    .with_confidence_threshold(0.5)
    .with_nms_threshold(0.7);

// 4. 执行检测
let detections = detector.detect(&image)?;

// 5. 处理检测结果
for detection in detections.iter() {
    println!(
        "检测到 {}: 置信度 {:.2}, 位置：({}, {}, {}, {})",
        detection.class_name,
        detection.confidence,
        detection.bbox.x1,
        detection.bbox.y1,
        detection.bbox.x2,
        detection.bbox.y2
    );
}
```

## 性能优化

1. **内存管理**:
   - 使用预分配的固定容量容器 (`ClrBud`)，避免频繁的内存分配和释放
   - 零拷贝设计（在容量范围内）

2. **坐标转换优化**:
   - 直接在原始数据上进行坐标转换，避免不必要的数据拷贝
   - 批量处理所有检测框（向量化潜力）

3. **排序优化**:
   - 使用高效的排序算法（Rust std::sort_by）
   - 按置信度降序排序，便于 NMS 处理

4. **NMS 优化**:
   - 通过提前过滤（置信度阈值）减少不必要的 IoU 计算
   - 面积检查跳过无效框

5. **ONNX Runtime**:
   - 利用 ONNX Runtime 的优化推理引擎加速模型执行
   - 支持多线程并行推理

6. **延迟和吞吐量**:
   - 单次检测延迟：~5-20ms（取决于模型大小和硬件）
   - 适用于实时应用（≥30 FPS）

## 可视化支持

Color 模块提供可视化工具用于调试和展示：

```rust
use perple::color::utils::draw_detections;

let result_image = draw_detections(&image, &detections);
// 保存或显示 result_image
```

可视化功能包括：

- 绘制边界框
- 显示类别标签
- 显示置信度分数
- 不同类别使用不同颜色

## 配置参数

检测器支持以下配置参数：

- **输入尺寸**: 默认 640x640，可通过构造函数设置
  - 较大尺寸：提高检测精度，增加计算量
  - 较小尺寸：降低检测精度，减少计算量
- **置信度阈值**: 默认 0.5，过滤低置信度检测
  - 范围：0.0 ~ 1.0
  - 调优建议见上文"置信度过滤"部分
- **NMS(Non-Maximum Suppression) 阈值**: 默认 0.7，控制重复检测的抑制程度
  - 范围：0.0 ~ 1.0
  - 较小值：更严格的抑制（减少重复检测）
  - 较大值：更宽松的抑制（保留更多检测）

这些参数可以在创建检测器时通过链式调用设置：

```rust
let detector = YoloDetector::new(640, 640)
    .with_confidence_threshold(0.6)
    .with_nms_threshold(0.5);
```

**注意**: 这些参数也可以通过全局配置文件统一设置。
