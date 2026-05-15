# 聚类点云物理中心确定方案 — 调研与规划

## 问题定义

已聚类的点云，需要确定一个稳定的"物理中心"作为运动测算（Kalman 滤波、轨迹关联）的观测值。核心挑战：LiDAR 只能采集物体面向传感器一侧的点，导致几何质心随观测角度漂移。

## 调研来源

| 来源 | 关键信息 |
|------|---------|
| LV-DOT (arXiv:2502.20607) | 同时维护 bbox 中心 + 点云质心，9D 特征向量关联 |
| CenterPoint (Yin Zhou, CVPR 2021) | BEV 热力图回归预测物体物理中心，不依赖原始质心 |
| AB3DMOT (Weng et al.) | 3D Kalman 滤波直接用 bbox 中心作为观测 |
| Arun (1987) / Besl & McKay (1992) | ICP/SVD 配准中质心减法是标准预处理步骤 |
| 你当前的系统 | density_weight_alpha 加权质心 + 6 状态 KF |

## 现有中心定义对比

| 方法 | 公式 | 优点 | 缺点 |
|------|------|------|------|
| **算术质心** | `C = (1/N) Σ p_i` | 简单、平滑 | 受遮挡/非均匀采样影响大 |
| **AABB 中心** | `C = (min + max) / 2` | 对极端点鲁棒 | 受离群点影响 |
| **OBB 中心 (PCA)** | PCA 旋转后取局部中心 | 贴合物体朝向 | 计算量大，退化情况不稳定 |
| **密度加权质心** (你现在用的) | `w = 1/r^α` | 补偿近密远疏偏差 | 仍受单侧观测影响 |
| **深度学习回归中心** | 网络直接预测 | 理论最优 | 需要训练数据和推理开销 |

## 核心问题分析

你当前系统 (`cluster.rs:47-86`) 的密度加权质心:

```
w_i = 1 / ||p_i||^alpha
C = Σ(w_i * p_i) / Σ(w_i)
```

这个方案解决了 LiDAR **径向密度不均匀** 的问题，但没有解决 **单侧观测** 的根本问题：物体背面没有点，质心始终偏向传感器方向。当物体旋转或传感器移动时，质心会产生系统性漂移，直接污染 KF 的位置观测。

## 建议方案

### 方案 A: OBB 中心替代质心（推荐，改动最小）

**原理**: PCA 拟合的 OBB 中心 = 旋转后的局部坐标中心。对于刚体，OBB 中心近似物体的几何对称中心，不受单侧观测影响。

**你已有基础设施**: `Box3D::from_points_pca()` 已实现。

**改动点**:
- `cluster.rs`: `cluster_box_and_centroid()` 中，当 `use_pca_obb=true` 时，centroid 直接取 OBB 的 pose translation，不再单独算加权质心
- 或者新增一个 `center_source` 配置项：`centroid` / `obb_center` / `bbox_center`

**风险**: 点数过少时 PCA 不稳定 → 可加最小点数阈值，低于阈值回退到 AABB 中心。

### 方案 B: 双中心特征关联（LV-DOT 风格）

**原理**: 同时维护 bbox 中心和点云质心，在关联阶段用 9D 特征向量（位置 + 尺寸 + 质心）做匹配，而非仅用单一中心。

**改动点**:
- `CldBud` 增加 `pc_centroid` 字段（当前已有 `centroid`）
- `tracker/core.rs` 的关联代价矩阵改为 Mahalanobis 距离 + 质心一致性
- KF 观测仍用 bbox/OBB 中心

**效果**: 提高关联鲁棒性，但不直接改善位置观测精度。

### 方案 C: 形状先验补偿（针对已知类别）

**原理**: 对已知类别（车辆、行人），预存典型尺寸。用可见点估算物体实际中心 = 质心 + 尺寸补偿偏移。

**公式**: `C_physical = C_centroid + offset(bbox_size, viewing_angle)`

**改动点**: 需要类别数据库和视角估算。适合自动驾驶场景，你的 bench 框架可以用来对比效果。

### 方案 D: 多帧滑动窗口稳定

**原理**: 不改中心计算方法，而是在 KF 观测层面做额外平滑。对质心做 N 帧加权平均后再喂给 KF。

**改动点**: `tracker/core.rs` 中 KF 的观测值改为滑动窗口均值。

**缺点**: 引入延迟，对快速运动目标不利。

## 推荐实施路径

1. **第一步** (改动最小，效果最直接): 实现方案 A — 让 `centroid` 可选为 OBB 中心
2. **第二步** (bench 对比): 用 bench 框架对比 `centroid` vs `obb_center` vs `bbox_center` 在轨迹稳定性上的差异
3. **第三步** (可选): 如果关联仍有问题，实施方案 B 双中心特征

## 相关论文

- [LV-DOT](https://arxiv.org/abs/2502.20607) — LiDAR-visual dynamic obstacle detection and tracking
- [CenterPoint](https://arxiv.org/abs/2006.11275) — Center-based 3D Object Detection and Tracking (CVPR 2021)
- [AB3DMOT](https://arxiv.org/abs/2008.08063) — A Baseline for 3D Multi-Object Tracking
- [Arun et al. 1987](https://doi.org/10.1109/TPAMI.1987.4767965) — Least-squares fitting of two 3-D point sets
- [Besl & McKay 1992](https://doi.org/1109/34.121791) — A Method for Registration of 3-D Shapes (ICP)
