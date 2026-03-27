# 配置管理文档

## 概述

Perple 使用 TOML 格式的配置文件来管理各种参数。系统支持全量配置加载和增量配置更新两种方式。

在代码中，可以通过 `fixif()` 函数获取全局配置单例，这是访问配置的标准方式。

## 全量配置加载

使用 `Config::new()` 方法从 `config/default.toml` 加载完整配置：

```rust
let config = Config::new();
```

在实际使用中，应该通过全局配置单例访问配置：

```rust
use perple::config::fixif;

// 获取全局配置单例
let config = fixif();
let stream_capacity = config.stream_capacity;
let model_path = &config.model_path;
let camera_intrinsic = &config.camera.intrinsic;
```

## 增量配置更新

系统支持在运行时通过增量配置文件更新部分配置项，而不影响其他配置。

### 使用方法

```rust
use perple::config::fixif;

let mut config = fixif();

// 通过 TOML 字符串更新
let update_toml = r#"
dbscan_min_points = 5
default_confidence_threshold = 0.7
"#;
config.update_from_toml(update_toml)?;

// 通过文件更新
config.update_from_file("config/update_example.toml")?;
```

### 增量配置文件示例

```toml
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

## 配置项详解

### 基础配置

```toml
# 流容量配置
stream_capacity = 16

# 检测结果容量
detections_capacity = 16

# 人员类别标签
person_class_label = "person"

# 点云容量预分配
points_capacity = 16384

# 分辨率（米）
resolution = 0.07
```

### 目标检测参数

```toml
# 模型输入尺寸
default_input_width = 640
default_input_height = 640

# 置信度阈值
default_confidence_threshold = 0.6

# NMS 阈值
default_nms_threshold = 0.7

# 模型路径
model_path = "module/color/yolo11n.onnx"
```

### DBSCAN 聚类参数

```toml
# 每个聚类的最小点数
dbscan_min_points = 3

# eps 会基于 resolution 动态计算：eps = resolution * 4.0
```

### 地面检测参数

```toml
# 默认地面法向量
default_ground_vector = [0.0, 0.0, 1.0]

# 地面过滤阈值
ground_filter_threshold = 0.3

# 地面叉积耐心值
ground_cross_product_patience = 3

# 地面采样测试次数
ground_sample_test_count = 23
```

### 点云聚类参数 (Claster)

```toml
[claster]
# 合并耐心值
merge_patience = 0.20

# 合并阈值
merge_threshold = 0.6

# 体素大小
voxel_size = 0.1

# 每个聚类的最小点数
min_points_per_cluster = 10

# 每个节点的最大点数（可选）
max_points_per_node = 50

# 最大树深度（可选）
max_tree_depth = 10

# 是否使用并行处理
use_parallel = true
```

### 相机配置

```toml
[camera]
# 内参矩阵 (3x3)
intrinsic = [
  [ fx,  0,  cx ],
  [ 0,   fy, cy ],
  [ 0,   0,  1  ]
]

# 外参矩阵 (4x4) - 从世界坐标系到相机坐标系的变换
extrinsic = [
  [ r11, r12, r13, tx ],
  [ r21, r22, r23, ty ],
  [ r31, r32, r33, tz ],
  [ 0,   0,   0,   1  ]
]
```

### 激光雷达配置

```toml
[lidar]
# 外参矩阵 (4x4) - 从世界坐标系到雷达坐标系的变换
extrinsic = [
  [ r11, r12, r13, tx ],
  [ r21, r22, r23, ty ],
  [ r31, r32, r33, tz ],
  [ 0,   0,   0,   1  ]
]
```

## 支持增量更新的配置项

几乎所有配置项都支持增量更新：

### 基础配置项

- `stream_capacity`
- `detections_capacity`
- `person_class_label`
- `points_capacity`
- `resolution`

### 目标检测参数

- `default_input_width`
- `default_input_height`
- `default_confidence_threshold`
- `default_nms_threshold`
- `model_path`

### DBSCAN 参数

- `dbscan_min_points`

### 地面检测参数

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

```toml
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

# DBSCAN 参数
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

```rust
use perple::config::fixif;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 获取全局配置（不可变引用）
    let config = fixif();
    println!("初始置信度阈值：{}", config.default_confidence_threshold);
    
    // 注意：由于 fixif() 返回的是静态引用，实际使用时
    // 通常需要在初始化阶段就完成所有配置更新
    // 或者设计专门的配置管理机制
    
    Ok(())
}
```
