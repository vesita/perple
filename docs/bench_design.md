# Benchmark 管线设计规范

## 整体架构

点云处理管线分五级串联，每层通过 trait 解耦：

```
原始点云 (raw cloud)
  ↓ GroundPickStrategy::pick()
非地面点 (non_ground)
  ↓ DenoiseStrategy::denoise()  —— 预处理降噪 (gentle, r=0.30 m=3)
降噪后非地面点 (denoised_non_ground)
  ↓ WallPickStrategy::pick()
非墙面点 (non_wall)
  ↓ DenoiseStrategy::denoise()  —— 后处理降噪 (standard, r=0.20 m=3)
降噪后点 (denoised)
  ↓ ClusteringStrategy::run()
障碍物簇 (clusters)
```

Benchmark 框架在此基础上增加 Preprocessor 层，用于共享前序处理结果。

## 各模块输入输出契约

### 1. 地面提取层 — GroundPickStrategy

```
输入:  &mut Vec<[f32; 3]>  原始点云（原地修改）
输出:  usize               地面点数量（前 n 个点为地面点）
       Vec<PlaneEq>        检测到的平面方程（可选）
约定:  pick() 将地面点交换到数组前部，返回地面点计数
       调用方通过 cloud[n_ground..] 获取非地面点
       输入为空时返回 (0, vec![])
```

### 2. 墙体提取层 — WallPickStrategy

```
输入:  &mut Vec<[f32; 3]>  非地面点云（原地修改）
输出:  usize               墙面点数量（前 n 个点为墙面点）
       Vec<PlaneEq>        墙体平面方程（可选）
约定:  同地面提取，墙面点交换到数组前部
       WallPreprocessor 内部依次执行 ground -> denoise -> wall 三级提取
       FrameData.non_wall() = denoised_non_ground[n_wall..]
       输入为空时返回 (0, vec![])
```

### 3. 降噪层 — DenoiseStrategy

```
输入:  &[[f32; 3]]         非墙面点（通常为 non_wall）
输出:  Vec<[f32; 3]>       保留的点（剔除稀疏离群点后的点集）
       Vec<usize>           保留点在原始输入中的索引映射
约定:  RadiusOutlierRemoval：邻域半径内点数 < min_pts → 剔除
       StatisticalOutlierRemoval：k 近邻平均距离 > μ + std_ratio·σ → 剔除
       输入为空时返回 (vec![], vec![])
```

### 4. 聚类层 — ClusteringStrategy

```
输入:  &[[f32; 3]]         待聚类的点集（通常为 non_wall）
输出:  Vec<[f32; 3]>      降采样后的点集（采样点）
       Vec<Vec<usize>>    每个簇在采样点中的索引列表
约定:  返回的 (sampled, objects) 满足：
        - sampled.len() ≥ objects.iter().map(|c| c.len()).sum() （差值为噪声点）
        - 噪声点不出现在任何簇的索引列表中
        - 每个 sampled 索引至多出现在一个簇中
        - 输入为空时返回 (vec![], vec![])
```

### 5. 数据源层级 — FrameData

```
cloud:        &[[f32; 3]]    原始点云（全部点）
non_ground(): &[[f32; 3]]  去地面后的点（AllPreprocessor 缓存）
non_wall():   &[[f32; 3]]  去地面+去墙体后的点（WallPreprocessor 缓存）
denoised():   &[[f32; 3]]  去地面+去墙体+降噪后的点（DenoisePreprocessor 缓存）
frame_idx:    u64          帧序号
```

依赖的 Preprocessor 层级：

| Preprocessor | cloud | non_ground() | non_wall() | denoised() |
|-------------|-------|-------------|-----------|-----------|
| PassthroughPreprocessor | 原始 | fallback cloud | fallback non_ground | fallback non_wall |
| GroundPreprocessor | 原始 | 去地面后 | fallback non_ground | fallback non_wall |
| WallPreprocessor | 原始 | 去地面+预处理降噪后 | 去墙体后 | fallback non_wall |
| DenoisePreprocessor | 原始 | 去地面+预处理降噪后 | 去墙体后 | 后处理降噪后 |

### 6. Benchmark 策略 — BenchStrategy trait

```
fn run(&mut self, frame: &FrameData) -> Duration
  功能: 对单帧执行策略，返回耗时
  时序: 每帧每策略调用一次，run → write_frame 顺序执行
  注意: 内部应自行计时；累计统计累加至 self 字段

fn write_frame(&mut self, recorder: &mut BenchRecorder, frame: &FrameData)
  功能: 将 run() 的检测结果写入 recorder 用于可视化
  时序: run() 之后立即调用，可用 self 中保存的上帧结果
  注意: 降采样点云至 ≤5000 点后再写入，避免 recorder 膨胀

fn summarize(&self)
  功能: 打印策略的平均统计汇总

fn stats(&self) -> BenchStats
  功能: 返回基础统计（名称、帧数、总耗时）

fn extra_metrics(&self) -> Vec<(String, f64)>
  功能: 返回额外量化指标，用于 Python 分析图
```

### 7. Preprocessor trait

```
fn preprocess(&mut self, cloud: &[[f32; 3]]) -> Preprocessed
  功能: 对原始点云执行级联预处理
  约定: 每帧调用一次，结果共享给所有候选策略

Preprocessed 枚举:
  - Passthrough                         — 无预处理
  - Ground { non_ground }               — 仅去地面
  - Wall { non_ground, non_wall }       — 去地面+预处理降噪+去墙体 (non_ground 已降噪)
  - Denoise { non_ground, non_wall, denoised } — 去地面+预处理降噪+去墙体+后处理降噪
```

## 策略特殊说明

### GroundPickStrategy 各实现

| 策略 | 特性 | 限制 |
|------|------|------|
| HistogramExpand | Z 轴直方图 + 峰值膨胀 | expand ≥ 0.5 时灾难性退化 |
| PeakScan | 峰值扫描 + 连通域 | 受限于直方图分辨率 |
| RansacGround | RANSAC 平面拟合 | distance 直接控制松紧度 |

### WallPickStrategy 各实现

| 策略 | 特性 | 限制 |
|------|------|------|
| XYRansacWall | TLS 精化 + 确定性种子 | 设为默认墙体策略 |
| TopDownCluster | 网格自顶向下聚类 | cell_size 控制过分割 |
| QuadtreeWall | 四叉树递归分割 | 较慢，p 值控制分割深度 |
| seq_fit | 顺序平面拟合 | **当前所有参数都检测到 0 墙面点，待排查** |

### DenoiseStrategy 各实现

| 策略 | 特性 | 限制 |
|------|------|------|
| RadiusOutlierRemoval | 半径邻域密度判定，O(n) 平均 | 参数 radius/min_pts 需调优 |
| StatisticalOutlierRemoval | SOR 统计离群点剔除，kNN 距离判定 | 暴力计算，大点云较慢 |

### ClusteringStrategy 各实现

| 策略 | 输入要求 | 特性 | 限制 |
|------|---------|------|------|
| LvdotClusterStrategy | non_ground（内部自带墙提+体素过滤+DBSCAN） | 完整管线 | skip_wall 模式下接收 denoised 跳过墙提 |
| WallClusterStrategy | non_wall（网格降采样+DBSCAN） | 低噪声 | with_pre_extracted_wall() 跳过内部墙提 |
| DbscanStrategy | non_wall（固定/自适应 eps） | 通用 DBSCAN | cluster_fixed 缺 Z 轴过滤（已修复） |
| RangeImageStrategy | non_wall（距离图像分割） | 极速 | 跨 -PI/PI 边界时 FOV 计算异常 |

#### LvdotClusterStrategy 的两种工作模式

1. **独立模式** (默认): 接收 non_ground 点（已预处理降噪），内部执行 墙提→体素过滤→DBSCAN
2. **直连模式** (with_pre_extracted_wall/skip_wall): 接收 denoised 点（已去地面+去墙体+后处理降噪），跳过内部墙提，直接做体素过滤→DBSCAN

bench 中通过 InputSource 区分：
- `InputSource::NonGround` → lvdot 独立模式（接收预处理降噪后的非地面点，内部墙体+聚类）
- `InputSource::Denoised` → 默认输入源，接收已去地面+预处理降噪+去墙体+后处理降噪的点

#### xy_dbscan 实质

`xy_dbscan` 在 bench 中实为 `LvdotClusterStrategy::direct(0.0, 1, eps, min_pts).with_pre_extracted_wall()`：
- voxel_size=0.0 → 跳过体素过滤
- skip_wall=true → 接收已去墙体点
- 仅执行 XY DBSCAN

## TOML 驱动 Benchmark

`config/bench/{task}/*.toml` 定义策略系列，每个 TOML 文件：

```toml
name = "策略名"
type = "策略类型"       # 对应工厂函数识别
preprocessor = "ground+wall"  # 前序预处理器

[quick]
frames = 1
params = [ ... ]       # 快速测试参数网格

[full]
frames = 10
params = [ ... ]       # 全量测试参数网格
```

`run_toml_bench()` 流程：
1. 扫描 `config/bench/{task}/` 下所有 .toml 文件
2. 每个文件调用 `StrategyBuilder::build(type, params)` 创建策略
3. 依次执行 quick/full 模式
4. 结果写入 `output/bench/{task}/{strategy}/info.json`

## 数据处理约定

### 人形检测
点云处理阶段**不输出行人信息**，仅输出障碍物簇。行人分类在融合阶段（Camera+LiDAR）确定。所有 bench 中的人形相关指标（count_human_like）已移除。

### AABB 输出
所有障碍物包围盒使用 **AABB**（轴对齐包围盒），OBB 已被剔除。

### 体素占用过滤
LV-DOT 风格过滤：`voxel_size=0.0` 时跳过处理（已修复除零问题）。
- `voxel_occupancy_filter`: 保留密集体素内所有点
- `voxel_occupancy_downsample`: 每密集体素输出 1 个质心

### 降采样上限
write_frame 中可视化点云统一降采样至 ≤5000 点（避免 recorder 文件膨胀），步长 = max(1, len/5000)。

## 共享 API

为避免 bench 示例间的代码重复，以下函数已提取到公共模块：

| 函数 | 位置 | 用途 |
|------|------|------|
| `xy_dbscan()` | `crate::cloud::wall::xy_dbscan` | XY 平面 DBSCAN（使用 XYGrid） |
| `to_cluster_result()` | `crate::bench::to_cluster_result` | 簇索引→点集+噪声计数 |
| `get_f32()` | `crate::bench::get_f32` | TOML 参数提取（返回 f32） |
| `get_i64()` | `crate::bench::get_i64` | TOML 参数提取（返回 i64） |
| `RadiusOutlierRemoval` | `crate::cloud::denoise` | 半径离群点剔除 |
| `StatisticalOutlierRemoval` | `crate::cloud::denoise` | SOR 统计离群点剔除 |

## 已知问题

1. **RangeImageStrategy 跨 -PI/PI 边界**: 点云跨越 azimuth 边界时 FOV 计算铺满 360°，浪费网格。待修复。
2. **seq_fit 零墙面点**: 所有参数组合都检测不到墙面点，可能是算法或坐标系问题。
3. **cluster_fixed Z 轴**: 修复前四叉树仅做 XY 搜索，不同高度的点被误聚类。2026-05-10 已修复。
4. **pipeline_bench wall_pct 分母**: 修复前使用体素过滤后点数做分母导致占比偏高。2026-05-10 已修复。
