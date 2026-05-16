# 墙体策略设计文档 — 图像边缘检测

## 设计理念

墙体提取不再使用传统几何方法（RANSAC、连通域、法线聚类、SVD 平面拟合等），
而是将 3D 点云投影为 BEV（Bird's Eye View）图像，利用成熟 2D 图像边缘检测
算法提取墙体。

## 算法演进

### 几何方法（已废弃）

以下策略经过大量对比测试后确认在室内点云墙体检测中存在根本性问题，已全部移除：

| 策略 | 问题 |
|------|------|
| RANSAC 系列 | 对稀疏点云鲁棒性差，随机采样在降采样后易丢失墙面结构 |
| CC 系列 | 网格分辨率敏感，薄墙易断裂，厚墙过度合并 |
| 法线系列 | 依赖局部法线估计，对噪声敏感，计算开销大 |
| 自适应系列 | 参数耦合度高，调参困难 |
| 顺序 SVD | 所有参数均检测到 0 墙面点 |

### BevEdLines（当前默认）

BEV 图像 + OpenCV EDLines 边缘检测：

1. **投影**：非地面点云投影到 XY 平面，按分辨率渲染为二值图像
2. **边缘检测**：EDLines 算法提取图像中的直线段
3. **反投影**：图像线段 → 3D 直线（Z 轴无限延伸的垂直平面）
4. **点分类**：计算每点到最近墙线的距离，小于阈值的标记为墙面点

优势：利用成熟的图像边缘检测算法，对点云稀疏和密度不均不敏感，
速度稳定 ~6ms/帧。

### BevHough（备选）

保留 Hough 变换作为备选方案，当前未激活。

## 文件结构

```
src/cloud/wall/
├── wall.rs           — WallPickStrategy trait + 模块根
├── bev_edlines.rs    — BevEdLines（默认）：BEV + EDLines
├── bev_hough.rs      — BevHough（备选）：BEV + Hough
└── l2_util.rs        — L2 几何检测共享工具
```

## 验证

```bash
cargo check
cargo run --example wall_bench -- --mode=quick
cargo run --example wall_pipeline_bench -- --mode=quick
.venv/Scripts/python.exe scripts/bench_viz.py
```
