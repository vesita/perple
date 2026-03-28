# 配置管理文档

## 概述

Perple 使用 TOML(Tom's Obvious, Minimal Language) 格式的配置文件来管理各种参数。系统支持全量配置加载和增量配置更新两种方式。

**核心特性**:

- ✅ TOML(Tom's Obvious, Minimal Language) 格式配置（易读、易写）
- ✅ 全量配置加载（从 default.toml）
- ✅ 增量配置更新（运行时动态更新）
- ✅ 全局配置单例（懒加载、线程安全）
- ✅ 类型安全的配置访问

在代码中，可以通过 `fixif()` 函数获取全局配置单例，这是访问配置的标准方式。

## 全量配置加载

使用 `Config::new()` 方法从 [`config/default.toml`](../config/default.toml) 加载完整配置：

```rust
let config = Config::new();
```

**注意**: `Config::new()` 会读取并解析整个配置文件，适用于初始化阶段。

在实际使用中，应该通过全局配置单例访问配置：

```rust
use perple::config::fixif;

// 获取全局配置单例（懒加载、线程安全）
let config = fixif();
let stream_capacity = config.stream_capacity;
let model_path = &config.model_path;
let camera_intrinsic = &config.camera.intrinsic;
```

**特性**:

- 懒加载：首次访问时才加载配置
- 线程安全：使用 `LazyLock` 和 `Mutex` 保护
- 全局唯一：整个应用程序共享同一配置实例

## 增量配置更新

系统支持在运行时通过增量配置文件更新部分配置项，而不影响其他配置。

### 使用方法

``rust
use perple::config::fixif;

// 注意：由于 fixif() 返回的是静态引用，实际使用时
// 通常需要在初始化阶段就完成所有配置更新
// 或者设计专门的配置管理机制

// 示例：通过 TOML 字符串更新（伪代码，实际需要可变引用）
let update_toml = r#"
dbscan_min_points = 5
default_confidence_threshold = 0.7
"#;
// config.update_from_toml(update_toml)?;

// 示例：通过文件更新（伪代码，实际需要可变引用）
// config.update_from_file("[`config/update_example.toml`](../config/update_example.toml)")?;

```

**重要提示**: 
- 当前实现中，`fixif()` 返回不可变引用
- 增量更新需要在初始化阶段完成
- 未来版本可能支持动态可变配置

### 增量配置文件示例

**示例文件**: [`config/update_example.toml`](../config/update_example.toml)

``toml
# config/update_example.toml
# 只更新需要更改的配置项

dbscan_min_points = 5
default_confidence_threshold = 0.7

[camera]
intrinsic = [
  [ 650.0,    0.0,    320.0  ],
  [ 0.0,      650.0,  240.0  ],
  [ 0.0,      0.0,    1.0    ]
]

[lidar]
extrinsic = [
  [ 1.0,  0.0,  0.0,  0.0 ],
  [ 0.0,  1.0,  0.0,  0.0 ],
  [ 0.0,  0.0,  1.0,  0.0 ],
  [ 0.0,  0.0,  0.0,  1.0 ]
]
```

**注意**: 增量更新只会修改明确指定的字段，其他字段保持原值。

## 配置项详解

### 基础配置

```toml
# 流容量配置（循环缓冲区大小）
stream_capacity = 16

# 检测结果容量（单次检测最大输出数）
detections_capacity = 16

# 人员类别标签（YOLO 模型中的类别名称）
person_class_label = "person"

# 点云容量预分配（最大点数）
points_capacity = 16384

# 分辨率（米，用于体素网格和距离阈值计算）
resolution = 0.07
```

### 目标检测参数

```toml
# 模型输入尺寸（像素）
default_input_width = 640
default_input_height = 640

# 置信度阈值（过滤低置信度检测）
default_confidence_threshold = 0.6

# NMS(Non-Maximum Suppression) 阈值（抑制重复检测）
default_nms_threshold = 0.7

# 模型路径（ONNX 文件位置）
model_path = "module/color/yolo11n.onnx"
```

**调优建议**:

- `default_confidence_threshold`:
  - 高精度场景：0.7~0.9
  - 高召回率场景：0.3~0.5
- `default_nms_threshold`:
  - 严格抑制：0.3~0.5
  - 宽松抑制：0.6~0.8

### DBSCAN(Density-Based Spatial Clustering of Applications with Noise) 聚类参数

```toml
# 每个聚类的最小点数（过滤噪声）
dbscan_min_points = 3

# eps 会基于 resolution 动态计算：eps = resolution * 4.0
# 典型值：0.28 米（当 resolution=0.07 时）
```

**调优建议**:

- `dbscan_min_points`:
  - 密集点云：3~5
  - 稀疏点云：5~10
- `eps` (通过 resolution 间接调整):
  - 小物体检测：减小 resolution
  - 大物体检测：增大 resolution

### 地面检测参数

```toml
# 默认地面法向量（Z 轴向上）
default_ground_vector = [0.0, 0.0, 1.0]

# 地面过滤阈值（点与地面的最小距离）
ground_filter_threshold = 0.3

# 地面叉积耐心值（迭代次数）
ground_cross_product_patience = 3

# 地面采样测试次数（RANSAC 迭代次数）
ground_sample_test_count = 23
```

**说明**:

- `ground_filter_threshold`: 过滤地面点，减少误检
- `ground_cross_product_patience`: 提高法向量估计精度
- `ground_sample_test_count`: RANSAC(Random Sample Consensus) 算法的迭代次数

### 点云聚类参数 (Claster)

```toml
[claster]
# 合并耐心值（距离阈值系数）
merge_patience = 0.20

# 合并阈值（米）
merge_threshold = 0.6

# 体素大小（米，用于下采样）
voxel_size = 0.1

# 每个聚类的最小点数
min_points_per_cluster = 10

# 每个节点的最大点数（可选，用于四叉树/KD-Tree）
max_points_per_node = 50

# 最大树深度（可选，限制搜索深度）
max_tree_depth = 10

# 是否使用并行处理
use_parallel = true
```

**调优建议**:

- `merge_threshold`:
  - 小物体：0.3~0.5 米
  - 大物体：0.6~1.0 米
- `voxel_size`:
  - 保持细节：0.05~0.1 米
  - 提高性能：0.1~0.2 米
- `use_parallel`:
  - 大规模点云（>10 万点）：启用
  - 小规模点云：禁用以减少开销

### 相机配置

```toml
[camera]
# 内参矩阵 (3x3)
# fx, fy: 焦距（像素）
# cx, cy: 主点坐标（像素）
intrinsic = [
  [ fx,  0,  cx ],
  [ 0,   fy, cy ],
  [ 0,   0,  1  ]
]

# 外参矩阵 (4x4) - 从世界坐标系到相机坐标系的变换
# R: 旋转矩阵 (3x3)
# t: 平移向量 (3x1)
extrinsic = [
  [ r11, r12, r13, tx ],
  [ r21, r22, r23, ty ],
  [ r31, r32, r33, tz ],
  [ 0,   0,   0,   1  ]
]
```

**标定方法**:

- 使用棋盘格标定法获取内参
- 使用手眼标定获取外参
- 推荐使用 OpenCV(Open Source Computer Vision Library) 或 MATLAB 标定工具箱

### 激光雷达配置

```toml
[lidar]
# 外参矩阵 (4x4) - 从世界坐标系到雷达坐标系的变换
# R: 旋转矩阵 (3x3)
# t: 平移向量 (3x1)
extrinsic = [
  [ r11, r12, r13, tx ],
  [ r21, r22, r23, ty ],
  [ r31, r32, r33, tz ],
  [ 0,   0,   0,   1  ]
]
```

**标定方法**:

- 使用联合标定板（同时包含相机和雷达特征）
- 使用 NDT(Normal Distributions Transform) 配准
- 推荐使用 LiDAR-Camera Calibration 工具箱

## 支持增量更新的配置项

几乎所有配置项都支持增量更新：

### 基础配置项

- `stream_capacity`
- `detections_capacity`
- `person_class_label`
- `points_capacity`
- `resolution`

### 目标检测

- `default_input_width`
- `default_input_height`
- `default_confidence_threshold`
- `default_nms_threshold`
- `model_path`

### DBSCAN 参数

- `dbscan_min_points`

### 地面检测配置项

- `default_ground_vector`
- `ground_filter_threshold`
- `ground_cross_product_patience`
- `ground_sample_test_count`

### 点云聚类参数 (claster)

- `claster.merge_patience`
- `claster.merge_threshold`
- `claster.voxel_size`
- `claster.use_parallel`

### 相机参数

- `camera.intrinsic`
- `camera.extrinsic`

### 激光雷达参数

- `lidar.extrinsic`

## 完整配置示例

**完整配置文件**: [`config/default.toml`](../config/default.toml)

```
# config/default.toml

# 基础配置
stream_capacity = 16
detections_capacity = 16
person_class_label = "person"
points_capacity = 16384
resolution = 0.07

# 目标检测超参数
default_input_width = 640
default_input_height = 640
default_confidence_threshold = 0.6
default_nms_threshold = 0.7
model_path = "module/color/yolo11n.onnx"

# DBSCAN(Density-Based Spatial Clustering of Applications with Noise) 参数
dbscan_min_points = 3

# 地面检测参数
default_ground_vector = [0.0, 0.0, 1.0]
ground_filter_threshold = 0.3
ground_cross_product_patience = 3
ground_sample_test_count = 23

# 点云聚类参数
[claster]
merge_patience = 0.20
merge_threshold = 0.6
voxel_size = 0.1
min_points_per_cluster = 10
max_points_per_node = 50
max_tree_depth = 10
use_parallel = true

# 相机配置
[camera]
intrinsic = [
  [ 641.03, 0.0,    343.42 ],
  [ 0.0,    640.89, 212.39 ],
  [ 0.0,    0.0,    1.0    ]
]
extrinsic = [
  [ 0.595, -0.758, -0.267, -2.498 ],
  [-0.059,  0.290, -0.955, -0.371 ],
  [ 0.802,  0.584,  0.128,  3.191 ],
  [ 0.0,    0.0,    0.0,    1.0    ]
]

# 激光雷达配置
[lidar]
extrinsic = [
  [ 1.0,  0.0,  0.0, 0.0 ],
  [ 0.0, -1.0,  0.0, 0.0 ],
  [ 0.0,  0.0, -1.0, 0.0 ],
  [ 0.0,  0.0,  0.0, 1.0 ]
]
```

## 注意事项

1. **增量更新特性**: 增量更新只会修改明确指定的字段，其他字段保持原值
2. **配置重置**: 如果需要重置某项配置为默认值，需要重新加载完整配置
3. **配置文件格式**: 增量配置文件的格式与完整配置文件相同，只是只包含需要更新的项
4. **全局单例**: 全局配置通过 `fixif()` 函数访问，这是一个单例模式实现，确保在整个应用程序中使用统一的配置
5. **线程安全**: 配置使用 `LazyLock` 实现懒加载，保证线程安全
6. **错误处理**: 配置文件加载失败会导致程序退出，并输出详细的错误信息

## 动态配置更新示例

```
use perple::config::fixif;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 获取全局配置（不可变引用）
    let config = fixif();
    println!("初始置信度阈值：{}", config.default_confidence_threshold);
    
    // 注意：由于 fixif() 返回的是静态引用，实际使用时
    // 通常需要在初始化阶段就完成所有配置更新
    // 或者设计专门的配置管理机制
    
    // 推荐做法：在程序启动时加载并更新配置
    // let mut config = Config::new();
    // config.update_from_file("config/updates.toml")?;
    
    Ok(())
}
```

**最佳实践**:

1. 在 `main()` 函数开始时加载配置
2. 使用增量配置文件覆盖默认值
3. 验证配置参数的有效性
4. 将配置传递给需要使用的模块
