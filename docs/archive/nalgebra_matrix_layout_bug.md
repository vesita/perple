# Nalgebra 矩阵布局 Bug：行优先 vs 列优先

## 问题描述

构造相机内外参矩阵时，`nalgebra::Matrix3::from()` 和 `Matrix4::from()` 以**列优先**存储数据，而配置 TOML 中内外参按**行优先**（标准 OpenCV 约定）书写。两者不匹配导致矩阵被隐式转置，所有 3D→2D 投影结果完全错误。

## 根因

### nalgebra 的 `from()` 语义

nalgebra `SMatrix::from<[[T; R]; C]>` 中，**外层数组下标 = 列索引**，内层 = 行索引：

```rust
let m = SMatrix::<f32, 3, 3>::from([[a,b,c], [d,e,f], [g,h,i]]);
// 存储布局 (column-major):
//   col0 = [a, b, c]   →  m[(0,0)]=a  m[(1,0)]=b  m[(2,0)]=c
//   col1 = [d, e, f]   →  m[(0,1)]=d  m[(1,1)]=e  m[(2,1)]=f
//   col2 = [g, h, i]   →  m[(0,2)]=g  m[(1,2)]=h  m[(2,2)]=i
// 等效矩阵:
//   [a  d  g]
//   [b  e  h]
//   [c  f  i]
```

### TOML 配置行优先约定

`config/default.toml` 按 OpenCV 惯例使用行优先：

```toml
intrinsic = [
    [523.08,  0,       327.82],   # row 0
    [0,       523.60,   209.39],   # row 1
    [0,       0,         1.00],   # row 2
]
```

### 组合效果

```rust
// 意图: 构造标准内参矩阵 K
//   [fx   0  cx]
//   [ 0  fy  cy]
//   [ 0   0   1]
let K = Matrix3::from(cfg.camera.intrinsic);

// 实际结果: K^T
//   [fx  0   0 ]
//   [0  fy   0 ]
//   [cx cy   1 ]
```

具体影响：

| 元素 | 期望值 | 实际值 | 误差 |
|------|--------|--------|------|
| `K[(0,2)]` (cx) | 327.82 | 0 | -327.82 px |
| `K[(1,2)]` (cy) | 209.39 | 0 | -209.39 px |

外参同样被转置：旋转矩阵变成 `R^T`，平移向量几乎为零（因为 TOML 末行 `[0,0,0,1]` 变成了末列）。

## 波及范围

所有从配置构造 nalgebra 矩阵的代码：

| 文件 | 行 | 构造方式 |
|------|-----|---------|
| `examples/reproject_check.rs` | 83-84 | `Matrix3::from()` / `Matrix4::from()` |
| `src/fuse.rs` | 26-28 | 同上 |
| `src/extrinsic_monitor.rs` | 59-61 | 同上 |
| `src/cloud/classify/core.rs` | 240-241 | 同上 |
| `src/color/look.rs` | 25-26, 49-50 | `from_iterator()` |

这 5 处代码分别影响：
- `reproject_check` — 重投影验证输出完全错位
- `fuse` — YOLO 2D ↔ LiDAR 3D 融合匹配失效
- `extrinsic_monitor` — 外参偏差监测基于错误投影
- `classify/core` — YOLO 辅助簇分裂投影错误
- `look` — 2D→3D 视线向量计算错误（影响关联/追踪）

## 修复

在所有 `from()` / `from_iterator()` 调用后追加 `.transpose()`：

```rust
// 修复前
let intrinsic = Matrix3::from(config.camera.intrinsic);
let cam_from_lidar = Matrix4::from(config.camera.extrinsic);

// 修复后
let intrinsic = Matrix3::from(config.camera.intrinsic).transpose();
let cam_from_lidar = Matrix4::from(config.camera.extrinsic).transpose();
```

`from_iterator` 同理：

```rust
// 修复前
let intrinsic = Matrix3::from_iterator(camera_config.intrinsic.iter().flatten().cloned());

// 修复后
let intrinsic = Matrix3::from_iterator(camera_config.intrinsic.iter().flatten().cloned()).transpose();
```

### 替代方案

也可用 `from_row_slice` 明确表达行优先意图：

```rust
let intrinsic = Matrix3::from_row_slice(&[
    cfg.camera.intrinsic[0][0], cfg.camera.intrinsic[0][1], cfg.camera.intrinsic[0][2],
    cfg.camera.intrinsic[1][0], cfg.camera.intrinsic[1][1], cfg.camera.intrinsic[1][2],
    cfg.camera.intrinsic[2][0], cfg.camera.intrinsic[2][1], cfg.camera.intrinsic[2][2],
]);
```

但 `.transpose()` 改动最小，不易引入新错误。

## 验证

修复后运行 `reproject_check` 示例确认投影合理：

```
========== 重投影检查 ==========
检查帧数: 78
YOLO 检测: 104
GT 标注: 234 人
GT 投影在相机前方: 120
GT 投影在图像内: 105
```

对比修复前（投影到图像内的比例显著更低）。

## Python 为何不受影响

`check_calib.py` 中 `fx, fy, cx, cy` 以标量硬编码，不经过矩阵构造：

```python
fx, fy = 523.08215322, 523.60224492
cx, cy = 327.81872513, 209.38830507
```

避免了 nalgebra 的列优先问题。但该脚本的 `ext_current` 同样是硬编码旧值，并非从当前 `config/default.toml` 读取。

## 规范

今后从 TOML/JSON 等外部数据构造 nalgebra 矩阵时，所有 `from()` 和 `from_iterator()` 必须意识到列优先语义。建议统一使用 `.transpose()` 后缀并添加注释：

```rust
// 配置存储为行优先 (OpenCV 惯例)，nalgebra 列优先，需转置
let intrinsic = Matrix3::from(config.camera.intrinsic).transpose();
```
