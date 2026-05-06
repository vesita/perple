# Perple 设计文档

## 项目概述

Perple 是一个基于 LiDAR + 相机融合的实时 3D 目标检测与跟踪系统，核心管线为：点云地面提取 → 聚类 → YOLO 辅助分裂 → 多目标跟踪。

## 模块结构

| 模块 | 职责 | 关键类型 |
|------|------|----------|
| `cloud` | 点云处理全流程：地面提取、聚类、融合输出 | `Classify`, `CldBud`, `GroundPickStrategy`, `ClusteringStrategy` |
| `cloud/ground` | 地面提取策略族（5 种实现） | `HistogramExpand`, `HistoseedPlane`, `RansacGround`, `PeakDownExpandUp`, `GpfGround` |
| `cloud/classify` | 聚类策略族 + 分类管线 | `DbscanStrategy`, `RangeImageStrategy`, `Claster` |
| `color` | 图像目标检测（YOLO ONNX 推理） | `ClrBud`, `load_model()` |
| `fuse` | 多模态融合（点云 + 视觉检测结果） | — |
| `tracker` | 多目标跟踪（Kalman + 状态机） | `Target` |
| `swapl` | 全局数据交换总线（LazyLock 单例） | `Swapl`, `global_swapl()` |
| `config` | TOML 配置加载 + 全局静态访问 | `Config`, `fixif()` |
| `utils` | 通用工具：Stream 流、Box3D 包围盒、Sight 投影 | `Stream<T>`, `Cream<I,O>`, `Eap<T>`, `Box3D`, `Sight` |
| `bench` | 策略 benchmark 框架 | `BenchHarness`, `BenchStrategy`, `BenchRecorder`, `FrameData` |
| `optional` | 可选组件（DataLoader 等） | `DataLoader` |
| `ros_bridge` | ROS1 桥接（feature-gated） | — |

## 数据流管线

```
DataLoader
    │
    ▼
Swapl.clouds  (原始点云入)
    │
    ▼
Classify.act()
    ├─ 1. GroundPickStrategy.pick()  → 地面点 + 平面方程
    ├─ 2. 天花板过滤 (ceiling_height)
    ├─ 3. 距离过滤 (max_range)
    ├─ 4. ClusteringStrategy.run()   → 簇索引
    ├─ 5. YOLO refine_with_yolo()    → 簇分裂
    └─ 6. 写入 Swapl.cld_buds_raw
              │
    ┌─────────┼─────────┐
    ▼         ▼         ▼
  Fuse    Tracker    clouds_filtered
  (融合)  (跟踪)    (供跟踪器投票)
```

**Swapl 各通道说明：**

| 通道 | 方向 | 用途 |
|------|------|------|
| `clouds` | 输入 | 原始点云 |
| `clouds_out` | 中间 | 预处理后点云（Classify 读取） |
| `clouds_filtered` | 中间 | 地面滤除后点云（Tracker 点云投票用） |
| `cld_buds_raw` | 输出 | 聚类结果（未融合） |
| `cld_objs` | 输出 | 融合后目标 |
| `colors` | 输入 | 图像帧 |
| `clr_objs` | 输出 | YOLO 检测结果 |
| `sights` | 输出 | 3D 投影结果 |
| `targets` | 输出 | 最终跟踪目标 |
| `ground_plane` | 中间 | 地面平面方程 `[a,b,c,d]` |

## 策略模式

### GroundPickStrategy（地面提取）

```rust
pub trait GroundPickStrategy: Send {
    fn pick(&mut self, cloud: &mut [[f32; 3]]) -> (usize, Vec<CldBud>, Option<[f32; 4]>);
    fn strategy_name(&self) -> &'static str { "unknown" }
}
```

**约定：** 调用后 `cloud[..n_ground]` 为地面点，`cloud[n_ground..]` 为非地面点（原地重排）。

**5 种实现：**
- `HistogramExpand` — Z 直方图峰值 + expand（默认，最简单）
- `HistoseedPlane` — 直方图种子 + RANSAC 平面生长
- `RansacGround` — 纯 RANSAC 平面拟合
- `PeakDownExpandUp` — 峰下扫描 + 上扩
- `GpfGround` — Ground Plane Fitting

**工厂函数：** `create_ground_strategy()` 返回 `Box<dyn GroundPickStrategy>`，当前默认 `HistogramExpand::new()`。

### ClusteringStrategy（聚类）

```rust
pub trait ClusteringStrategy: Send {
    fn run(&mut self, points: &[[f32; 3]]) -> (Vec<[f32; 3]>, Vec<Vec<usize>>);
    fn strategy_name(&self) -> &'static str { "unknown" }
}
```

**工厂函数：** `create_strategy()` 读取 `config.claster.strategy` 字段，支持 `"dbscan"`, `"dbscan_adaptive"`, `"range_image"`。

### BenchStrategy（Benchmark）

```rust
pub trait BenchStrategy {
    fn name(&self) -> &str;
    fn run(&mut self, frame: &FrameData) -> Duration;
    fn write_frame(&mut self, recorder: &mut BenchRecorder, frame: &FrameData);
    fn summarize(&self);
}
```

`FrameData` 提供三层点云视图：`cloud`（原始）、`preprocessed`（默认策略排序后）、`non_ground`（非地面子集）。

## Bench 框架设计原则

1. **串行执行** — 同一帧的所有策略依次执行，避免并发干扰计时
2. **每帧预处理一次** — `create_ground_strategy().pick()` 在策略循环外执行，所有策略共享同一份预处理结果
3. **100ms 告警** — 单帧单策略超过 `WARN_THRESHOLD_MS`（100ms）时打印 `[WARN]`
4. **独立输出** — 每个策略持有独立的 `BenchRecorder`，结果写入独立 `.rdra` 文件
5. **参数扫描** — 通过 `with_params()` 构造函数批量生成参数组合（见 `ground_bench.rs` 示例）

## 配置系统

**加载链：** `config/default.toml` → `Config::new()` (toml::from_str) → `LazyLock<Config>` → `fixif()`

**访问方式：** 任意模块调用 `crate::config::fixif()` 获取 `&'static Config`。

**核心字段分组：**

| 字段组 | 示例字段 | 用途 |
|--------|----------|------|
| 流容量 | `stream_capacity`, `points_capacity` | Stream 环形缓冲区大小 |
| 检测 | `default_confidence_threshold`, `default_nms_threshold` | YOLO 推理参数 |
| 地面 | `ground_strategy`, `ground_expand`, `upside_down` | 地面提取行为 |
| `[claster]` | `strategy`, `voxel_size`, `eps_slope`, `max_range` | 聚类算法参数 |
| `[tracker]` | `max_disappeared`, `moving_speed_threshold` | 跟踪器状态机参数 |
| `[camera]` | `intrinsic` (3x3), `extrinsic` (4x4) | 相机内外参矩阵 |

**注意：** 字段名 `claster` 是历史拼写（非 cluster），代码中统一使用此拼写。

## 关键设计决策

### 1. 模块文件组织

采用 `{module}.rs` + `{module}/` 目录并存的方式：
- `src/cloud.rs` — 模块入口，声明子模块 + `pub use` 重导出
- `src/cloud/ground.rs` — 子模块入口，声明子子模块 + trait 定义
- `src/cloud/ground/histogram.rs` — 具体实现

这避免了 `mod.rs` 嵌套地狱，每个层级的入口文件就是同名 `.rs` 文件。

### 2. 策略工厂模式

三个策略 trait 都配合工厂函数使用：
- `create_ground_strategy()` — 固定返回默认策略（`HistogramExpand`）
- `create_strategy()` — 读取配置动态分发（`cfg.claster.strategy`）
- Bench 中直接构造（`HistogramExpand::with_expand(0.10)`）

工厂函数返回 `Box<dyn Trait>`，实现运行时多态。策略通过 `with_params()` 构造函数接受自定义参数，无参数时使用 `new()` 从 `fixif()` 读取默认值。

### 3. Swapl 全局总线

Swapl 是 `LazyLock<Swapl>` 单例，所有数据流通道都是 `Eap<Stream<T>>`（即 `Arc<Mutex<Stream<T>>>`）。

**设计意图：** 模块间松耦合 — 生产者和消费者只需持有 `Swapl` 中对应通道的 clone，无需显式依赖注入。`global_swapl()` 是唯一的访问入口。

**注意事项：** `Stream<T>` 是固定容量环形缓冲区，写满后旧数据被覆盖。`Cream<I,O>` 是双向流适配器，连接处理单元的输入和输出。

### 4. 数据类型约定

- `CldBud` — 点云聚类结果（包围盒 + 簇标签 + 置信度）
- `ClrBud` — 图像检测结果
- `Box3D` — 3D 包围盒（支持 AABB 和 PCA-OBB）
- `Eap<T>` = `Arc<Mutex<T>>` — 线程安全共享引用
- `Stream<T>` — 固定容量环形缓冲区，`write()` 覆盖旧数据，`read()` 取最新
