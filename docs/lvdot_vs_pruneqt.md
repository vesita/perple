# LV-DOT vs PruneQt：聚类前降采样/过滤策略对比

## 背景

LV-DOT（LiDAR-Visual Dynamic Obstacle Detection and Tracking）是一种轻量化动
态障碍物检测框架，其视觉深度处理流（`dbscanDetect`）采用 **体素占用过滤 + 3D
DBSCAN** 的思路对深度图反投影点云进行聚类。

PruneQt 是本项目提出的改进方案，在同级处理流位置用 **四叉树自适应降采样 + XY
DBSCAN** 替代 LV-DOT 的均匀体素过滤。两者解决的核心问题相同：在聚类之前对原
始点云进行压缩和噪声过滤，使后续 DBSCAN 既快又准。

---

## 处理流对比

```
LV-DOT (visual):
  Depth Image → back-project → voxelFilter (0.1m uniform) → height filter → 3D DBSCAN

PruneQt:
  Raw Cloud → wall detection (BevEdLines) → [box pre-cluster] →
    quadtree build → dense-leaf centroid → quadtree-accelerated XY DBSCAN → [border-point assign]
```

---

## 核心差异

### 1. 空间划分：均匀 vs 自适应

| | LV-DOT voxelFilter | PruneQt quadtree filter |
|---|---|---|
| 划分方式 | 均匀三维体素 (0.1m) | 自适应 XY 四叉树 (max_depth=10) |
| 近处行为 | 一个体素含几十个点，阈值到达后只存 1 点 → 大量信息丢弃 | 四叉树自动细分到小叶片，保留更精细的局部结构 |
| 远处行为 | 一个体素可能只有 2-3 点，达不到 `voxelOccThresh=5` → 全部丢弃 | 叶片自动变大，`min_occ` 更容易满足 → 远距召回更好 |
| 密度适应性 | 无 — 远近分辨率一致 | 有 — 密度越高叶片越小，分辨率自动匹配 |

LV-DOT 采用的均匀体素网格在自车坐标系下有一个根本问题：**LiDAR 点云的密度分
布在空间上是高度非均匀的**（近密远疏），而均匀体素完全无视这一特性。结果是近
处过采样（明明有很多点但只保留 1 个），远处欠采样（本来点就不多，很可能达不到
阈值而被丢弃）。

PruneQt 的四叉树天然适配非均匀分布：点的密集区域树自动深、叶片小，稀疏区域树
浅、叶片大。这使得过滤阶段的压缩率随密度自适应变化，在全距离范围内保持信息量
的相对均衡。

### 2. 墙面处理

LV-DOT 不做墙面检测，仅靠高度滤波 `[groundHeight_, roofHeight_] = [0.2m,
2.0m]` 来排除地面和天花板。墙面点（如走廊两侧的墙壁）会作为"非地面"点进入后
续聚类，在实际环境中墙面产生的 false positive 需要靠 DBSCAN 或后续
YOLO/LiDAR-visual 融合来消除。

PruneQt 在预处理阶段通过 BevEdLines 墙体检测精确分离墙面/非墙面点。墙面点在聚
类之前就被剔除，下游 DBSCAN 只需要处理行人等真正障碍物的点云。这不仅减少了
FP，也降低了 DBSCAN 的输入规模。

### 3. DBSCAN 维度与加速

| | LV-DOT | PruneQt |
|---|---|---|
| 检索维度 | 3D (xyz) | 2D (XY) — 行人场景 Z 轴变化小 |
| 相邻搜索 | 暴力全量搜索 (O(n²)) | 四叉树范围查询 (O(log n)) |
| 典型参数 | `eps=0.05, minPts=20` | `eps=0.20, min_pts=5` |

LV-DOT 的 `eps=0.05`（5cm）非常小，这是因为它的体素过滤后点很稀疏，且保留的
点原本就在同一体素附近。但这个参数对 LiDAR 点云来说过小，行人的腿和身体容易被
分裂为多个簇。

PruneQt 使用 XY 平面 DBSCAN（行人高度方向上不做区分），`eps=0.20` 的半径更适
合行人的典型尺度（肩宽 ~0.5m）。同时四叉树索引使相邻搜索从 O(n²) 降至
O(n log n)，这对性能至关重要。

### 4. 边界点恢复

LV-DOT 的 voxelFilter 对每个体素的输出策略是"达到阈值就存一个点"——这个点不一
定是体素质心，而是恰好使计数达到阈值的那个点。稀疏体素中的其他点全部丢失，没
有恢复机制。

PruneQt 提供了可选的 `use_border_points` 机制（`prune_qt.rs:199-225`）：稀疏叶
片中的点虽然不满足 `min_occ` 而不产生质心，但如果它们落在某个已形成簇的 DBSCAN
半径内，仍可作为边界点加入该簇。这进一步提高了远距离行人的召回率。

---

## 定量对比

### 408-frame ablation benchmark（本项目中最佳配置）

| 策略 | Person F1 | Precision | Recall |
|------|-----------|-----------|--------|
| **PruneQt** | **0.745** | 82.3% | 68.1% |
| LV-DOT style (`lvdot_grid`) | — | — | — |
| Old `dbscan_qt` (baseline) | — | — | EDLines + prune_qt 相比旧基线 recall +11.4pp, FP -59% |

> 精确的 lvdot_grid 对比数据参考 `eval_ablation` 脚本运行结果。

---

## 总结

LV-DOT 的体素占用过滤思路（够密才保留）和 PruneQt 的叶节点过滤思路（密集叶片
输出质心）本质上都是为了解决同一个问题：**在保留语义结构的前提下压缩点云，使
DBSCAN 只处理有意义的代表点**。

PruneQt 的核心改进在于用**自适应空间划分**替代了**均匀网格**，这使得过滤策略
与 LiDAR 点云固有的近密远疏分布特性对齐。墙面检测和四叉树加速 DBSCAN 则是在
同一框架下的自然延伸。
