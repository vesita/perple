# Cloud 模块文档（LiDAR 点云处理）

## 概述

Cloud 模块是 Perple 项目中负责处理 LiDAR(Light Detection and Ranging) 点云数据的核心模块。它能够读取点云数据，进行聚类分析，并输出 3D 边界框。

**核心功能**:

- ✅ PCD(Point Cloud Data) 格式读取和解析
- ✅ 无效点过滤（NaN、无穷大值）
- ✅ DBSCAN(Density-Based Spatial Clustering of Applications with Noise) 聚类算法
- ✅ KD-Tree 和四叉树空间索引加速
- ✅ 3D 边界框拟合和输出
- ✅ 地面检测和过滤

## 模块架构

### 核心组件

**源代码位置**: [`src/cloud/`](../src/cloud/)、[`src/cloud/classify/`](../src/cloud/classify/)

```
cloud 模块
├── core.rs           # 点云数据核心处理
├── classify/         # 点云分类子模块
│   ├── cluster.rs    # 聚类算法实现
│   ├── core.rs       # 分类核心逻辑
│   ├── environment.rs# 环境分析
│   ├── kdtree.rs     # KD-Tree 空间索引
│   ├── quadtree.rs   # 四叉树空间索引
│   └── somecode.rs   # 其他辅助代码
└── output.rs         # 点云检测结果输出 (CldBud)
```

### 主要结构体

#### CldBud

`CldBud` 是固定容量的 3D 边界框容器，用于存储聚类结果：

- 预分配固定容量（默认 16），避免动态内存分配
- 提供类似 Vec 的操作接口（push, clear, len, is_empty 等）
- 支持迭代器访问（IntoIterator, IntoRefIterator）
- 存储包含 `the_box` (Box3D)、`class_name`、`confidence` 等信息
- 实现 `Stream` trait，支持流式处理

**性能优势**:

- 🚀 零堆分配（在容量范围内）
- 🚀 缓存友好（连续内存布局）
- 🚀 适用于实时系统（可预测的性能）

#### Box3D

`Box3D` 表示一个三维边界框，用于包围点云中的对象：

**字段说明**:

- `x_min`, `x_max`: X 轴的最小和最大值（米）
- `y_min`, `y_max`: Y 轴的最小和最大值（米）
- `z_min`, `z_max`: Z 轴的最小和最大值（米）
- `length`, `width`, `height`: 长宽高尺寸（米）
- `pose`: 姿态信息（欧拉角或四元数）
- `center_x`, `center_y`, `center_z`: 中心点坐标（米）

**功能**:

- 计算体积、表面积
- 计算 IoU(Intersection over Union, 交并比)
- 判断点是否在框内
- 获取中心点坐标
- 从位置和角度创建边界框
- 从点云集合拟合边界框

使用示例：

```rust
use perple::utils::boxes::Box3D;

// 从点云创建边界框（自动拟合）
let box3d = Box3D::from_points(&points);

// 从位置和角度创建（指定尺寸和姿态）
let box3d = Box3D::from_position_and_angles(
    x, y, z,           // 位置（米）
    roll, pitch, yaw,  // 旋转角度（弧度）
    length, width, height  // 尺寸（米）
);

// 获取中心点
let center = box3d.center();

// 计算与另一个边界框的 IoU
let iou = box3d.iou(&other_box3d);
```

## 数据结构

### 点云数据存储

点云数据使用预分配的 Vec 存储，具有以下特点：

- 避免动态内存分配
- 提供点云数据的基本操作（push, clear, len 等）
- 支持迭代器访问
- 自动过滤无效点（NaN 和无穷大值）

## 处理流程

### 点云读取

1. 使用 `pcd-rs` 库读取 PCD 格式的点云文件
2. 过滤无效点（NaN 和无穷大值）
3. 将有效点存储到容器中（预分配容量）

**支持的文件格式**:

- ASCII PCD 格式
- Binary PCD 格式（需要 pcd-rs 支持）

### 聚类分析

聚类处理流程如下：

1. **初始化**: 清空历史聚类结果，准备新的聚类周期
2. **遍历点云**: 对所有点云数据点进行遍历
3. **点聚类**: 对每个点进行聚类处理：
   - 使用 KD-Tree 查找附近的现有聚类（距离阈值内）
   - 如果找到合适的聚类，则将点添加到该聚类
   - 如果未找到合适的聚类，则创建新的聚类
4. **聚类合并**: 执行最终的聚类合并操作，合并距离小于阈值的聚类
5. **边界框拟合**: 为每个聚类计算最小外接 3D 边界框
6. **输出结果**: 将结果写入输出流 (`CldBud`)

**性能优化**:

- 使用空间索引加速近邻搜索（KD-Tree、四叉树）
- 并行处理（可选，通过配置启用）
- 体素网格过滤（减少点数，提高效率）

### 空间索引优化

为了提高聚类效率，模块使用了空间索引结构：

#### KD-Tree(K-Dimensional Tree, kdtree.rs)

- K 维树空间索引（K-Dimensional Tree）
- 用于快速最近邻搜索（O(log n) 复杂度）
- 适用于高维空间查询
- 支持范围查询和 k 近邻查询

#### 四叉树 (quadtree.rs)

- 二维空间分割数据结构
- 用于平面区域的快速查询
- 特别适合地面点云处理（Z 轴变化小的场景）
- 递归细分空间，每个节点最多包含 4 个子节点

**性能对比**:

- KD-Tree: 适合高维、稀疏点云
- 四叉树：适合低维、密集点云（尤其是地面区域）

### 聚类参数

聚类算法使用以下关键参数：

- `CLUSTER_DISTANCE_THRESHOLD`: 聚类距离阈值（米），决定点与聚类间的最大距离
  - 典型值：0.3 ~ 0.7 米
  - 较大值：更少的聚类（点更容易被归为一类）
  - 较小值：更多的聚类（点更难被归为一类）
  
- `CLUSTER_MERGE_THRESHOLD`: 聚类合并阈值（米），决定两个聚类是否应该合并
  - 典型值：0.5 ~ 1.0 米
  - 用于合并相近的聚类，避免过分割
  
- `MIN_POINTS_PER_CLUSTER`: 每个聚类所需的最小点数
  - 典型值：3 ~ 10 个点
  - 过滤噪声和小簇

这些参数可以在全局配置文件 (`config/default.toml`) 中调整。

## 输出格式

### 3D 边界框

聚类结果以 3D 边界框的形式输出，每个边界框包含：

**几何信息**:

- `x_min`, `x_max`: X 轴范围（米）
- `y_min`, `y_max`: Y 轴范围（米）
- `z_min`, `z_max`: Z 轴范围（米）
- `length`, `width`, `height`: 尺寸信息（米）
- `center_x`, `center_y`, `center_z`: 中心点坐标（米）
- `pose`: 姿态信息（可选）

**语义信息**:

- `class_name`: 类别名称（如 "obstacle", "person", "vehicle" 等）
- `confidence`: 置信度分数（0.0 ~ 1.0）

**注意**: 当前实现中，类别和置信度可能需要通过配置文件或后续的分类模块确定。

### 输出容器

处理结果存储在 `CldBud` 容器中，该容器具有以下特点：

- 固定容量，避免动态内存分配
- 提供标准的集合操作接口
- 支持迭代器访问
- 可直接用于后续的跟踪模块

## 性能优化

1. **内存预分配**:
   - 所有容器都预分配固定容量（`CldBud`, `Vec<Point3D>` 等）
   - 避免运行时动态内存分配
   - 减少内存碎片和 GC 压力

2. **无效点过滤**:
   - 在读取阶段就过滤掉无效点（NaN、Inf）
   - 减少后续处理负担
   - 提高聚类效率

3. **空间索引**:
   - 使用 KD-Tree 加速近邻搜索（O(n) → O(log n)）
   - 使用四叉树处理地面区域
   - 显著减少距离计算次数

4. **高效聚类算法**:
   - 优化的 DBSCAN 实现
   - 增量式聚类（无需重新计算）
   - 减少重复计算

5. **聚类合并**:
   - 通过合并相近聚类减少最终输出数量
   - 避免过分割
   - 提高检测质量

6. **并行处理** (可选):
   - 通过配置启用并行聚类
   - 利用多核 CPU 加速处理
   - 适用于大规模点云（>10 万点）

**典型性能**:

- 处理速度：~10-50ms（1 万点，取决于参数和硬件）
- 内存占用：低（预分配 + 零拷贝）

## 可视化支持

使用 `rerun` 库支持点云数据的 3D 可视化：

```
use rerun::{RecordingStream, log_points, log_line_segments};

// 创建记录流
let rec = RecordingStream::new("perple_lidar")?;

// 记录原始点云
log_points("point_cloud", &points)?;

// 记录聚类结果
for cluster in clusters.iter() {
    // 记录 3D 边界框（作为线框）
    let box_lines = cluster.the_box.to_line_segments();
    log_line_segments("clusters", &box_lines)?;
    
    // 记录中心点
    log_points("cluster_centers", &[cluster.the_box.center()])?;
}
```

可视化功能包括：

- 显示原始点云数据（彩色或灰度）
- 显示聚类结果和 3D 边界框（不同聚类使用不同颜色）
- 支持交互式查看和分析（旋转、缩放、平移）
- 显示聚类统计信息（点数、尺寸、置信度）
- 时间序列回放（支持录制和重放）

**提示**: rerun 支持实时可视化和离线分析两种模式。

## 与其他模块的集成

### 与 Tracker 模块集成

Cloud 模块的输出 (`CldBud`) 直接输入到 Tracker 模块进行多目标跟踪：

```rust
// Cloud 模块输出
let cloud_detections: Vec<CldBud> = process_point_cloud(&points)?;

// Tracker 模块使用
tracker.update_cloud_detections(cloud_detections)?;
```

### 与 Color 模块融合

Cloud 模块可以与 Color 模块的输出进行融合，实现多模态检测：

```rust
// 获取图像检测结果
let image_detections = detector.detect(&image)?;

// 获取点云检测结果
let cloud_detections = process_point_cloud(&points)?;

// 融合两种检测结果
let fused_targets = fuse_detections(&image_detections, &cloud_detections)?;
```

## 配置示例

通过配置文件调整聚类参数：

```toml
[cloud]
# 聚类距离阈值（米）
cluster_distance_threshold = 0.5

# 聚类合并阈值
cluster_merge_threshold = 1.0

# 每个聚类的最小点数
min_points_per_cluster = 5

# 点云容量预分配
points_capacity = 10000
```
