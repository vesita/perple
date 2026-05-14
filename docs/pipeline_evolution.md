# 点云处理管线演化

## 概述

本项目的点云聚类处理经历了三个阶段的技术演进，从朴素的直接 DBSCAN 发展到三层级联管线（地面提取 → 墙体提取 → 后聚类）。

## 技术路线对比

在同一数据集（10 帧，统一 DBSCAN eps=0.20 min_pts=5）下的量化对比：

| Era | 管线策略 | 输入点量 | 耗时/帧 | 簇数 | 人检测 | 噪声 | 加速比 |
|:---:|----------|--------:|--------:|----:|------:|----:|------:|
| 1a | 原始全量 → DBSCAN | 20006 | **10096ms** | 41.4 | 12.1 | 308 | 1x |
| 1b | 原始 + 降采样 0.10 → DBSCAN | 20006 | **41ms** | 46.2 | 12.5 | 342 | 244x |
| 2 | 去地面 + 降采样 0.10 → DBSCAN | 15779 | **29ms** | 36.3 | 12.9 | 150 | 353x |
| 3 | **去地面 + 去墙体 → DBSCAN** | 1484 | **29ms** | 29.9 | 6.6 | 139 | 352x |

### 关键发现

1. **降采样是最大的加速贡献者**（244x），但保留地面点导致簇过度分割和噪声
2. **去地面** 减少 21% 的点量，带来 1.4x 额外加速，人检测 recall 不变
3. **去墙体** 减少 90% 的点量，但 DBSCAN 时间复杂度 O(n log n) 下速度收益不显著
4. **去墙体降低人检测 recall**（12.9 → 6.6），因为行人在墙边时会被墙体移除带走

## 降噪集成（Era 4）

2026-05-11 管线升级：在墙体提取两侧各增加一级降噪。

- **预处理降噪**（WallPreprocessor 内部, r=0.30, m=3）：在墙体提取之前去除孤立离群点，改善 XYGrid BFS 连通性，帮助稀疏墙体段的检测。
- **后处理降噪**（DenoisePreprocessor 内, r=0.20, m=3）：在聚类之前清洁非墙面点，去除残留噪点。

`WallPreprocessor` 管线现为：地面 → 降噪 → 墙体。`DenoisePreprocessor` 封装 `WallPreprocessor` 并追加后降噪。降噪默认全程开启，`cluster_bench` 的 `--denoise` 标志已移除。

## 五层级联管线架构

地面提取与墙体提取之间增加降噪层，墙体提取与后聚类之间增加第二降噪层：

```
                    ┌─────────────┐
                    │  原始点云    │
                    │  ~20000 pts │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │   地面提取   │  GroundPickStrategy
                    │  histogram  │  peak_scan / ransac
                    │  expand=0.20│
                    └──────┬──────┘
                           │ 非地面点 ~15779 pts
                    ┌──────▼──────┐
                    │  预处理降噪  │  DenoiseStrategy (gentle)
                    │  r=0.30 m=3 │  改善墙体 BFS 连通性
                    └──────┬──────┘
                           │ 降噪后非地面点
                    ┌──────▼──────┐
                    │   墙体提取   │  WallPickStrategy
                    │  ransac_l2_grid│  cc_pca_qt / cc_pca_grid
                    │  0.08/50/30 │
                    └──────┬──────┘
                           │ 非墙面点 ~1484 pts
                    ┌──────▼──────┐
                    │  后处理降噪  │  DenoiseStrategy (standard)
                    │  r=0.20 m=3 │  聚类前清洁
                    └──────┬──────┘
                           │ 降噪后非墙面点
                    ┌──────▼──────┐
                    │   后聚类     │  ClusteringStrategy
                    │  xy_dbscan  │  range_image / lvdot
                    │  xy_grid    │  dbscan_adaptive
                    └─────────────┘
                           │
                    ┌──────▼──────┐
                    │  检测结果    │
                    │  行人 / 障碍物│
                    └─────────────┘
```

## 地面提取策略对比

地面提取负责从原始点云中分离地面点，输出非地面点供后续处理。

### benchmark 结果

![地面策略精度-速度权衡](../output/bench/analysis/cross/ground_all_strategies.png)

### 各策略参数敏感性

| 策略 | 最佳参数 | 耗时 | 地面占比 | 特点 |
|------|---------|:----:|:-------:|------|
| **histogram** | expand=0.20 | 28.8ms | 23.3% | expand 拐点效应明显，≥0.50 时灾难性退化 |
| **peak_scan** | threshold=0.10, expand=0.20 | 28.0ms | 22.9% | 速度最稳定，精度上限受限于直方图分辨率 |
| **ransac** | distance=0.2, iter=50 | 57.6ms | 51% | distance 直接控制宽松度，d=0.6 过度检测 |

### 参数敏感性曲线

![histogram 参数敏感性](../output/bench/analysis/full/ground/histogram_sweep.png)

![ransac 参数敏感性](../output/bench/analysis/full/ground/ransac_sweep.png)

![peak_scan 参数敏感性](../output/bench/analysis/full/ground/peak_scan_sweep.png)

## 墙体提取策略对比

墙体提取在非地面点上进一步分离墙面点，输出干净的障碍物点给聚类。

benchmark 结果（详细数据见 `output/bench/wall/`）：

| 策略 | 平均耗时 | 特点 |
|------|:-------:|------|
| ransac_l2_grid (默认) | 最快 | TLS 精化+确定性种子 |
| cc_pca_qt | 中等 | 网格自顶向下聚类 |
| cc_pca_grid | 较慢 | XY 哈希网格+BFS 连通域+PCA 判别 |
| seq_pca_grid | — | 所有参数都检测到 0 墙面点（待排查） |

## 后聚类策略对比

后聚类在去除地面和墙体的纯净障碍物点上执行，输出行人、障碍物等目标。

### 人检测效率

![聚类策略 人检测效率](../output/bench/analysis/cross/cluster_all_strategies.png)

点大小代表簇数量，越靠左上越快且人检 recall 越高。

| 策略 | 耗时 | 人检 recall | 噪声 | 适用场景 |
|------|:----:|:----------:|:----:|---------|
| **lvdot** | ~17ms | 3.5 | 低 | 速度稳定，人检天花板低 |
| **xy_dbscan** | ~22-29ms | **7.9** | 高 | 人检 recall 最高，需调参 |
| **range_image** | ~3-10ms | 6.8 | 极高 | 速度之王，噪声大 |
| **wall_cluster** | ~25ms | 3.4 | 可为零 | 网格预过滤+DBSCAN |

### 噪声分析

![聚类策略 噪声分析](../output/bench/analysis/cross/cluster_noise_all.png)

理想区（左下角：少簇+低噪声）内 wall_cluster 表现最佳，range_image 噪声最大。

详细每个策略的参数扫描曲线见 `output/bench/analysis/full/cluster/`。

## 综合推荐管线

```
地面提取: peak_scan threshold=0.10 expand=0.20 (28ms)
  → 墙体提取: bev_edlines distance=0.08 (图像边缘检测, ~15ms)
    → 后聚类: dbscan_qt eps_slope=0.2 min_pts=10 voxel=0.10 (~24ms)
```

如需低噪声场景：
```
后聚类: wall_cluster cell=0.15 eps=0.15 min_pts=3 (0 噪声)
```

如需极速场景：
```
后聚类: range_image azimuth=2.0 elevation=2.0 threshold=0.5 (2.4ms)
```

## Era 5：基于图像的墙体检测 + 降噪内聚

2026-05-14 管线重构：

### 墙体检测：几何 → 图像

经过大量对比测试发现，传统几何方法（RANSAC、连通域、法线聚类、SVD 平面拟合等）在室内点云墙体检测中存在以下问题：

- **RANSAC 系列**：对稀疏点云鲁棒性差，随机采样在降采样后易丢失墙面结构
- **CC 系列**：网格分辨率敏感，薄墙易断裂，厚墙过度合并
- **法线系列**：依赖局部法线估计，对噪声敏感，计算开销大
- **自适应系列**：参数耦合度高，调参困难

**BevEdLines**（BEV 图像 + OpenCV EDLines）将墙体检测转化为 2D 图像边缘检测问题：
1. 点云投影到 BEV 图像（俯视图）
2. EDLines 算法提取图像边缘线段
3. 反投影回 3D 空间得到墙面直线
4. 沿 Z 轴扩展为墙面

优势：利用成熟的图像边缘检测算法，对点云稀疏和密度不均不敏感，速度稳定 ~15ms/帧。
保留 `BevHough` 作为 Hough 变换备选方案，预留 LSD 算法接口。

### 降噪内聚至聚类策略

预处理降噪（地面→墙体间）和后处理降噪（墙体→聚类间）均从核心管线移除，将降噪逻辑内聚到各 ClusteringStrategy 实现内部。
降噪不再是管线层级的固定环节，而是策略的可选特性。

当前管线简化为：

```
原始点云
  → 地面提取 (peak_scan)
    → 墙体提取 (bev_edlines)
      → 体素过滤 (仅跟踪器投票用)
        → 后聚类 (含内部降噪)
```

性能（release 模式）：全管线（含 YOLO 推理 28-44ms）平均 ~42ms/帧。点云处理 5-9ms，YOLO 推理为瓶颈。

## Era 6：Pipeline 并行 + DualBuf 锁域分离

2026-05-14 管线优化：

### DualBuf 锁域分离

引入 DualBuf（双缓冲 + producer/consumer 模式）替代 Stream 作为检测→后融合的跨阶段数据通道：

```rust
pub struct DualBuf<T> {
    producer: Mutex<Vec<T>>,  // 检测阶段写入
    consumer: Mutex<Vec<T>>,  // 后融合阶段读取
}
```

通过 `swap()` 一次性交换 producer/consumer 缓冲区，消除检测阶段（Lidar/Camera）和后融合阶段（Fuse/Tracker）之间的锁竞争。涉及字段：`clouds_filtered`、`cld_buds_raw`、`clr_objs`。

### std::sync::Mutex 替代 tokio::sync::Mutex

所有 Eap/Stream/DualBuf 的内部锁从 `tokio::sync::Mutex` 改为 `std::sync::Mutex`。原因：CPU 密集型检测任务在 tokio 工作线程上运行时，tokio::sync::Mutex 的 async 上下文切换引入 ~10-15ms 调度延迟。`std::sync::Mutex` 的同步锁在短临界区场景下更高效。

### 帧级 Pipeline 并行

检测阶段与后融合阶段通过 spawn 重排序实现并行：

```
帧 N: [  检测(Lidar+Camera)  ] → swap → [ 后融合(Fuse+Tracker) ]
                   ↓ spawn next 帧 N+1 检测
帧 N+1:           [  检测(Lidar+Camera)  ] → swap → ...
```

关键改动：swap 后立即 spawn 下一帧检测，使其与当前帧后融合并行执行。

### 性能收益

| 指标 | Debug（优化前） | Release（优化后） |
|------|:-:|:-:|
| 帧耗时 | ~90-100ms | ~42ms |
| 点云处理 | 5-9ms | 5-9ms |
| YOLO 推理 | 40ms | 28-44ms |
| fps | ~10 | ~24 |

631 帧全量测试：总耗时 26.6s，帧均 42ms。瓶颈完全在 YOLO 端，点云处理 5-9ms 被 pipeline 并行覆盖为零成本。<｜end▁of▁thinking｜>

<｜｜DSML｜｜parameter name="file_path" string="true">E:\code\perple\docs\pipeline_evolution.md
