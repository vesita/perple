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
use crate::config::fixif;

// 获取全局配置单例
let config = fixif();
let stream_capacity = config.stream_capacity;
```

## 增量配置更新

系统支持在运行时通过增量配置文件更新部分配置项，而不影响其他配置。

### 使用方法

```rust
use crate::config::fixif;

let mut config = fixif();

// 通过 TOML 字符串更新
let update_toml = r#"
dbscan_min_points = 5
"#;
config.update_from_toml(update_toml)?;

// 通过文件更新
config.update_from_file("config/update_example.toml")?;
```

### 增量配置文件示例

```toml
# 只更新需要更改的配置项
dbscan_min_points = 5

[camera]
intrinsic = [
  [ 650.0,    0.0,    320.0  ],
  [ 0.0,      650.0,  240.0  ],
  [ 0.0,      0.0,    1.0    ]
]

extrinsic = [
  [ 1.0,  0.0,  0.0,  0.0 ],
  [ 0.0,  1.0,  0.0,  0.0 ],
  [ 0.0,  0.0,  1.0,  0.0 ],
  [ 0.0,  0.0,  0.0,  1.0 ]
]

[lidar]
extrinsic = [
  [ 1.0,  0.0,  0.0,  0.0 ],
  [ 0.0,  1.0,  0.0,  0.0 ],
  [ 0.0,  0.0,  1.0,  0.0 ],
  [ 0.0,  0.0,  0.0,  1.0 ]
]
```

### 支持增量更新的配置项

几乎所有配置项都支持增量更新：

- 基础配置: `stream_capacity`, `detections_capacity`, `person_class_label`, `points_capacity`, `resolution`
- 目标检测参数: `default_input_width`, `default_input_height`, `default_confidence_threshold`, `default_nms_threshold`
- DBSCAN 参数: `dbscan_min_points`
- 相机参数: `camera.intrinsic`, `camera.extrinsic`
- 激光雷达参数: `lidar.extrinsic`

### 注意事项

1. 增量更新只会修改明确指定的字段，其他字段保持原值
2. 如果需要重置某项配置为默认值，需要重新加载完整配置
3. 增量配置文件的格式与完整配置文件相同，只是只包含需要更新的项
4. 全局配置通过 `fixif()` 函数访问，这是一个单例模式实现，确保在整个应用程序中使用统一的配置
