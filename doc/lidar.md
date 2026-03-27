# Cloud 模块文档（LiDAR 点云处理）

## 概述

Cloud 模块是 Perple 项目中负责处理 LiDAR 点云数据的核心模块。它能够读取点云数据，进行聚类分析，并输出 3D 边界框。

## 模块架构

### 核心组件

```
cloud 模块
├── core.rs           # 点云数据核心处理
├── classify/         # 点云分类子模块
│   ├── claster.rs    # 聚类算法实现
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
- 预分配固定容量，避免动态内存分配
- 提供类似 Vec 的操作接口（push, clear, len 等）
- 支持迭代器访问
- 存储包含 `the_box` (Box3D)、`class_name`、`confidence` 等信息

#### Box3D

`Box3D` 表示一个三维边界框，用于包围点云中的对象：

**字段说明**:
- `x_min`, `x_max`: X 轴的最小和最大值
- `y_min`, `y_max`: Y 轴的最小和最大值
- `z_min`, `z_max`: Z 轴的最小和最大值
- `length`, `width`, `height`: 长宽高尺寸
- `pose`: 姿态信息
- `center_x`, `center_y`, `center_z`: 中心点坐标

**功能**:
- 计算体积、表面积
- 计算 IoU（交并比）
- 判断点是否在框内
- 获取中心点坐标
- 从位置和角度创建边界框

使用示例：
```rust
use perple::utils::boxes::Box3D;

// 从点云创建边界框
let box3d = Box3D::from_points(&points);

// 从位置和角度创建
let box3d = Box3D::from_position_and_angles(
    x, y, z,           // 位置
    roll, pitch, yaw,  // 旋转角度
    length, width, height  // 尺寸
);

// 获取中心点
let center = box3d.center();
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
3. 将有效点存储到容器中

### 聚类分析

聚类处理流程如下：

1. **初始化**: 清空历史聚类结果
2. **遍历点云**: 对所有点云数据点进行遍历
3. **点聚类**: 对每个点进行聚类处理：
   - 查找附近的现有聚类
   - 如果找到合适的聚类，则将点添加到该聚类
   - 如果未找到合适的聚类，则创建新的聚类
4. **聚类合并**: 执行最终的聚类合并操作，合并相近的聚类
5. **输出结果**: 将结果写入输出流 (`CldBud`)

### 空间索引优化

为了提高聚类效率，模块使用了空间索引结构：

#### KD-Tree (kdtree.rs)
- K 维树空间索引
- 用于快速最近邻搜索
- 适用于高维空间查询

#### 四叉树 (quadtree.rs)
- 二维空间分割
- 用于平面区域的快速查询
- 特别适合地面点云处理

### 聚类参数

聚类算法使用以下关键参数：

- `CLUSTER_DISTANCE_THRESHOLD`: 聚类距离阈值，决定点与聚类间的最大距离
- `CLUSTER_MERGE_THRESHOLD`: 聚类合并阈值，决定两个聚类是否应该合并
- `MIN_POINTS_PER_CLUSTER`: 每个聚类所需的最小点数

这些参数可通过配置文件进行调整。

## 输出格式

### 3D 边界框

聚类结果以 3D 边界框的形式输出，每个边界框包含：

**几何信息**:
- `x_min`, `x_max`: X 轴范围
- `y_min`, `y_max`: Y 轴范围
- `z_min`, `z_max`: Z 轴范围
- `length`, `width`, `height`: 尺寸信息

**语义信息**:
- `class_name`: 类别名称（如 "obstacle", "person" 等）
- `confidence`: 置信度分数

### 输出容器

处理结果存储在 `CldBud` 容器中，该容器具有以下特点：

- 固定容量，避免动态内存分配
- 提供标准的集合操作接口
- 支持迭代器访问
- 可直接用于后续的跟踪模块

## 性能优化

1. **内存预分配**: 所有容器都预分配固定容量，避免运行时内存分配
2. **无效点过滤**: 在读取阶段就过滤掉无效点，减少后续处理负担
3. **空间索引**: 使用 KD-Tree 和四叉树加速近邻搜索
4. **高效聚类算法**: 优化的聚类过程，减少重复计算
5. **聚类合并**: 通过合并相近聚类减少最终输出数量

## 可视化支持

使用 `rerun` 库支持点云数据的 3D 可视化：

```rust
use rerun::{RecordingStream, log_points};

// 记录原始点云
log_points("point_cloud", &points)?;

// 记录聚类结果
for cluster in clusters.iter() {
    log_box("clusters", &cluster.the_box)?;
}
```

可视化功能包括：
- 显示原始点云数据
- 显示聚类结果和 3D 边界框
- 支持交互式查看和分析
- 不同聚类使用不同颜色标识

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
