# 验证流程文档

## 概览

本项目的验证体系分为三层：
- **管线运行** — 输出 JSONL + .rdra 数据
- **精度评估** — 基于标注数据的定量评测
- **可视化** — Python 脚本生成论文图表

## 环境要求

```bash
# Python 依赖 (venv 已配置)
.venv/Scripts/python.exe scripts/viz_trajectory.py

# YOLO 模型
model/quantized/yolo11n.onnx

# 标注数据
data/labeled/label/          # STPoints JSON 标注
data/labeled/camera/image/   # 相机图像
data/labeled/lidar/          # 点云数据
```

---

## 1. 管线运行

完整管线（检测 → 融合 → 跟踪），输出 JSONL 供可视化。

```bash
# Release 模式运行 408 帧
cargo run --release -- --frames 408 --output ./output/thesis_final

# 跳过头 N 帧（管线预热）
cargo run --release -- --frames 408 --skip 20 --output ./output/skip_test

# 输出内容
#   output/thesis_final/
#     pipeline.jsonl    # 每帧检测/跟踪/延迟数据
#     ground.db         # 地面点 .rdra
#     wall.db           # 墙体点 .rdra
#     cluster.db        # 聚类 .rdra
#     tracker.db        # 跟踪结果 .rdra
```

---

## 2. 精度评估

### 2.1 单阈值评估 (eval_labeled)

基于 data/labeled 标注数据，计算 Precision / Recall / F1。

```bash
# 默认: IoU=0.15, 全部帧
cargo run --example eval_labeled

# 中心距匹配（推荐）: 0.5m
cargo run --release --example eval_labeled -- --center-dist 0.5

# 指定输出目录 + 帧数
cargo run --release --example eval_labeled -- \
    --center-dist 0.5 --frames 408 --output ./output/eval_result

# 输出
#   eval_result.json   # 完整结果（含 per-class, per-distance）
#   eval_result.csv    # 汇总 CSV
```

**关键参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--iou` | 0.15 | IoU 匹配阈值 |
| `--center-dist` | 0.0 (关闭) | 中心距匹配阈值(m)，开启后替代 IoU |
| `--frames` | 全部 | 评测帧数 |
| `--skip` | 0 | 跳过的初始帧数 |
| `--output` | 自动 | 输出目录 |
| `--disable-yolo-smooth` | false | 关闭 YOLO 帧间标签平滑（测试平滑影响时用） |

**评估维度：** 输出两个层级的指标：

| 层级 | 说明 |
|------|------|
| **严格评估** | 仅 `class_type == "person"` 的检测参与匹配（原始行为） |
| **空间评估** | 全部检测参与匹配，衡量实际空间检测能力 |
| **类别识别分析** | 分析有多少 GT 被正确分类为 person、误分类为 obstacle、完全漏检 |

示例输出：
```
  ── 严格评估 (仅 class_type == "person") ──
    GT: 1224  | 检测:  801  | TP:  444  FP:  357  FN:  780
    Precision: 55.4%  | Recall: 36.3%  | F1: 0.4385

  ── 空间评估 (全部检测参与匹配) ──
    GT: 1224  | 检测: 2643  | TP:  807  FP: 1836  FN:  417
    Precision: 30.5%  | Recall: 65.9%  | F1: 0.4174

  ── 行人类别识别分析 ──
    空间匹配正确: 807/1224 (65.9%)
    ├─ 正确分类为 person: 444 (36.3%)
    └─ 误分类为 obstacle/其他: 363 (29.7%)
```

JSON 输出新增字段: `tp_spatial`, `fn_spatial`, `precision_spatial`, `recall_spatial`, `f1_spatial`, `tp_person`, `tp_nonperson`。

### 2.2 消融实验 (eval_ablation)

调参对比工具，支持运行时覆盖任意配置参数，输出与 eval_labeled 一致的双维度指标。

```bash
# 默认参数
cargo run --release --example eval_ablation

# 指定帧数
cargo run --release --example eval_ablation -- --frames 408

# 调参：聚类 min_points（降低提高召回率）
cargo run --release --example eval_ablation -- \
    --cluster-toml 'min_points_per_cluster=3'

# 调参：YOLO 置信度
cargo run --release --example eval_ablation -- \
    --cluster-toml 'min_points_per_cluster=3' \
    --tracker-toml 'min_confidence=0.5'

# 从配置文件加载
cargo run --release --example eval_ablation -- --config ./experiment.toml

# 输出
#   eval_result.json   # 完整结果（含配置快照）
#   eval_result.csv    # 汇总 CSV（含分类质量字段）
```

**关键参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--iou` | 0.15 | IoU 匹配阈值 |
| `--center-dist` | 0.0 (关闭) | 中心距匹配阈值(m) |
| `--frames` | 全部 | 评测帧数 |
| `--output` | 自动 | 输出目录 |
| `--config` | 无 | 完整 TOML 配置文件路径 |
| `--ground-toml` | 无 | 地面参数覆盖（逗号分隔） |
| `--cluster-toml` | 无 | 聚类参数覆盖（逗号分隔） |
| `--denoise-toml` | 无 | 降噪参数覆盖（逗号分隔） |
| `--tracker-toml` | 无 | 跟踪参数覆盖（逗号分隔） |

### 2.3 PR 曲线 (eval_pr_curve)

管线跑一次，在 0.05-1.0m 共 20 个阈值下计算 P/R/F1，同时输出 **person 过滤** 和 **全部检测** 两组曲线。

```bash
cargo run --release --example eval_pr_curve -- --output ./output/pr_curve

# 输出
#   pr_curve.json  # 20 个阈值点的 P/R/F1（points: person过滤, points_all: 全部检测）
```

JSON 中 `points`（向后兼容原格式）+ `points_all`（新增全部检测曲线数据）。

---

## 3. 可视化图表

### 3.1 轨迹与性能图 (viz_trajectory.py)

从 pipeline.jsonl 生成 4 张图：

```bash
# 自动查找最新 pipeline_xxx/pipeline.jsonl
.venv/Scripts/python.exe scripts/viz_trajectory.py

# 指定文件
.venv/Scripts/python.exe scripts/viz_trajectory.py output/thesis_final/pipeline.jsonl

# 输出
#   fig_trajectory_bev.png   — BEV 俯视轨迹图
#   fig_speed_curves.png     — 各目标速度曲线
#   fig_stats.png            — 每帧检测数/聚类数/分类分布
#   fig_latency.png          — 各阶段延迟堆叠图
```

### 3.2 汇总统计 (viz_summary.py)

生成统计表 + 分类饼图：

```bash
.venv/Scripts/python.exe scripts/viz_summary.py output/thesis_final/pipeline.jsonl

# 输出
#   summary.csv               — 汇总 CSV
#   summary_table.tex         — LaTeX 表格（直接插入论文）
#   fig_classification_pie.png — 分类分布饼图
```

### 3.3 PR 曲线 (viz_pr_curve.py)

从 pr_curve.json 生成论文用曲线，**自动绘制 person 过滤 + 全部检测两条对比曲线**。

```bash
.venv/Scripts/python.exe scripts/viz_pr_curve.py output/pr_curve/pr_curve.json

# 输出
#   fig_pr_curve.png           — PR 曲线对比（两条：person 过滤 vs 全部检测）
#   fig_f1_vs_threshold.png    — F1/Precision/Recall 随阈值变化（双模式对比）
```

### 3.4 批量实验 (run_cluster_experiments.py)

自动修改聚类参数、运行 eval、汇总对比：

```bash
.venv/Scripts/python.exe scripts/run_cluster_experiments.py

# 自动修改 config/default.toml → 运行 eval → 还原配置
# 结果保存在 output/experiment_summary.json
```

---

## 4. 管线扫描 (eval_pipeline)

按文件编号分段统计管线输出，用于分析不同场景下的检测密度。

```bash
# 默认统计
cargo run --example eval_pipeline

# 指定编号范围 + 分段大小
cargo run --example eval_pipeline -- --start 200 --end 906 --fold-size 100
```

输出每帧的聚类数、person 数、obstacle 数，以及移动目标数，并可分段绘制趋势图。

---

## 5. 完整验证流程

```bash
# Step 1: 运行管线（获取原始数据）
cargo run --release -- --frames 408 --output ./output/final

# Step 2: 精度评估（中心距 0.5m，输出双维度指标）
cargo run --release --example eval_labeled -- \
    --center-dist 0.5 --output ./output/final_eval

# Step 3: PR 曲线（20 个阈值，双模式）
cargo run --release --example eval_pr_curve -- --output ./output/pr_curve

# Step 4: 生成图表
.venv/Scripts/python.exe scripts/viz_trajectory.py output/final/pipeline.jsonl
.venv/Scripts/python.exe scripts/viz_summary.py output/final/pipeline.jsonl
.venv/Scripts/python.exe scripts/viz_pr_curve.py output/pr_curve/pr_curve.json
```

---

## 6. 配置与参数

关键配置项 (`config/default.toml`):

| 参数 | 当前值 | 说明 |
|------|--------|------|
| `strategy` | `"dbscan_qt"` | 聚类策略（DBSCAN + 四叉树加速） |
| `min_points_per_cluster` | 5 | DBSCAN 核心点最少邻点数，越低 Recall 越高 |
| `merge_patience` | 0.10 | DBSCAN 基础邻域半径 eps（米） |
| `eps_slope` | 0.05 | 自适应 eps 斜率，值越大远处邻域半径越大 |
| `voxel_size` | 0.10 | 体素下采样格子大小（米） |
| `denoise_radius` | 0.20 | 聚类前半径离群点剔除半径（米） |
| `denoise_min_pts` | 3 | 降噪最小邻点数 |
| `density_weight_alpha` | 2.0 | 密度感知质心加权指数：`r^α` 补偿 LiDAR 近密远疏导致的质心偏移 |
| `default_confidence_threshold` | 0.6 | YOLO 置信度阈值，越低检测越多 |
| `default_nms_threshold` | 0.7 | YOLO NMS 阈值 |
| `max_range` | 10.0 | 有效检测距离上限（米），超出此距离的点在聚类前被过滤 |
| `downsample_method` | `"voxel"` | 下采样方法: `"voxel"`（均匀体素）或 `"gaussian"`（距离概率采样） |
| `point_vel_threshold` | 0.08 | 点云投票位移阈值(m)，排除抖动噪声 |
| `moving_speed_threshold` | 0.35 | 运动速度阈值(m/s)，高于此值判定为 Moving |
| `use_point_cloud_voting` | true | 点云投票开关，通过对比历史点云位移判断目标是否真正在运动 |
| `kf_avg_frames` | 5 | Kalman 滤波器平滑窗口帧数 |

**后聚类过滤链** (`clusters_to_cldbuds`):

| # | 条件 | 过滤目标 |
|---|------|----------|
| 1 | `w <= 0.2` 或 `h <= 0.3` | 超小噪点 |
| 2 | `h < 0.15 * w` | 扁平物体（井盖、地面残留） |
| 3 | `length × width × h < 0.03` | 微小体积噪点 |
| 4 | `w > 3.0` | 过大物体（墙残留） |
| 5 | `box3d.center().z < 0.2` | 地面残留噪点（AABB 中心 Z 过低） |
| 6 | `center_dist + half_diag > max_range` | 超出有效检测范围的截断目标 |
| 7 | `volume > 0.5 && n_pts / volume < 20` | 大体积稀疏噪点 |

**调参建议顺序：**

1. 先调 `min_points_per_cluster`（直接影响召回率）
2. 再调 `denoise_radius` / `denoise_min_pts`（控制噪声）
3. 最后调 `default_confidence_threshold`（YOLO 检测灵敏度）

每次调参后运行 `eval_labeled` 验证效果。

---

## 7. 最终评估结果

### 7.1 408 帧全量评测（当前配置）

**当前默认配置：**
- 聚类: `dbscan_qt`（min_pts=5, eps=0.10, denoise_radius=0.20, denoise_min_pts=3）
- 密度加权: `r^α`（α=2.0）
- 后聚类过滤链: 7 道（尺寸、扁度、体积、Z 中心、边界、稀疏度）
- 跟踪: 点云投票 + 几何 fallback（trick.rs）+ 航迹评分 + BTreeMap
- YOLO: 置信度 0.6 + 帧间标签平滑

指标概要（中心距 0.5m 匹配，408 帧，YOLO 非确定性导致波动）：

```
── Person 过滤 (仅 class_type == "person") ──
  GT: 1224  | 检测: ~766-977 | TP: ~631-693  FP: ~135-284  FN: ~531-593
  Precision: ~71-82%  | Recall: ~52-57%  | F1: ~0.63

── 全部类别 (All Classes) ──
  GT: 1224  | 检测: ~1830-1854 | TP: ~865-872  FP: ~958-989  FN: ~352-359
  Precision: ~47%  | Recall: ~71%  | F1: ~0.56
```

**速度：** 408 帧 / ~17-25s = **~16-24 FPS**（Debug 模式较慢，Release 约 24 FPS）。

**非确定性说明：** YOLO (ONNX Runtime on DirectML) 推理不同 run 产生不同检测结果（检测数波动 ~200），是 F1 波动主因。跟踪器已通过 BTreeMap 消除自身非确定性。

### 7.2 消融对比

**早期基线（各阶段渐进改进）：**

| 配置 | Person Precision | Person Recall | Person F1 |
|------|:---:|:---:|:---:|
| 原始代码基线 (100帧) | 76.7% | 26.3% | 0.392 |
| + 几何 fallback + 点云投票 | 44.0% | 52.0% | 0.477 |
| + Static 持久化 fallback (app≥10) | 37.8% | **61.4%** | 0.468 |
| + 后聚类过滤链 (Z中心+边界+10m) | **79.0%** | 49.1% | **0.606** |
| + 密度加权公式修正 `r^α` | 66.1% | 53.9% | 0.594 |
| + YOLO 平滑 + BTreeMap | 60.0% | 53.3% | 0.563 |

**近期参数消融（均基于修复后代码，中心距 0.5m，408 帧）：**

| 实验 | 配置 | All Recall | Person Recall | Person Precision | Person F1 |
|------|------|:---:|:---:|:---:|:---:|
| Exp2 | min_pts=5, denoise=0.20, conf=0.5 | 70.7% | 51.6% | **82.4%** | **0.634** |
| Exp3 | Exp2 + denoise=0.15 | 71.0% | 53.3% | 76.2% | 0.635 |
| Exp4 | min_pts=3, denoise=0.15, conf=0.5 | 71.2% | **56.6%** | 70.9% | 0.630 |

近期实验主要方向及结论：

| 方向 | 结论 |
|------|------|
| `min_points_per_cluster` 3→5 | 5 精度更高(82%→71%), 3 召回更高(52%→57%), F1 基本持平 |
| denoise_radius 0.15→0.20 | 0.20 显著降低 FP（-149），但召回略降（-2pp） |
| 多帧累积聚类 | 放弃 — 累积后簇 box 被拉大导致 geometry 误判，FP 暴增 |
| RANSAC 替代 DBSCAN | 放弃 — RANSAC 线检测不适合行人团状点云，Recall 仅 14% |

### 7.3 关键改动

| 文件 | 改动 | 作用 |
|------|------|------|
| `src/tracker/trick.rs` | 几何 fallback：Floating 速度+几何、Static 纯几何 | 盲区行人检出 |
| `src/tracker/object.rs` | `geo_labeled` 字段 + correct() 覆盖逻辑 | 几何标签可被 YOLO 修正 |
| `src/tracker/analysis.rs` | 动态点云投票（全历史累积 + 方向过滤 + 位移阈值） | 提高运动检测稳定性 |
| `src/tracker/association.rs` | 排序 obj_ids | 消除关联顺序波动 |
| `src/tracker/core.rs` | `HashMap` → `BTreeMap` | 消除 tracker 内部迭代非确定性 |
| `src/cloud/classify/cluster.rs` | 密度加权公式 `1/r^α` → `r^α` | 修复质心偏向传感器 Bug，F1 +8.3% |
| `src/yolo_smooth.rs` | YOLO 帧间标签动量平滑 | 减少 YOLO 间歇性漏检导致的标签闪烁 |
| `src/cloud/classify/cluster.rs` | 中心 Z 过滤 + 边界过滤 + 点云投票方向一致性 | 大幅降低 FP |
| `config/default.toml` | max_range 12→10m, 相机标定参数更新 | 收紧检测范围，降低远端噪声 |
| `src/config.rs` | `update_from_toml()` + `init_config()` + `OnceLock` | 支持 CLI 运行时参数覆盖 |
| `examples/eval_ablation.rs` | 消融实验工具，支持任意配置参数运行时覆盖 | 自动化调参对比 |
| `src/main.rs` | 接入 YoloSmoother（`clr_objs.swap()` → `smooth()` → `fuse.act()`） | 主线启用帧间标签平滑 |
| `examples/eval_labeled.rs` | 添加 `--disable-yolo-smooth` 开关 | 支持关闭平滑做对比实验 |

### 7.4 误差分析

408 帧共 1224 个 GT Pedestrian（均在 0~10m 范围内）：

- **误分类（空间命中但 label 错误）**: ~135-198 (11-16%) → 空间匹配到但被标记为 obstacle（几何 fallback 未覆盖或 trick 条件不满足）
- **漏检（空间未命中）**: ~352-389 (29-31%) → LiDAR 预处理阶段丢失（地面/墙体误删、遮挡、聚类截断）
- **误报 FP**: 在 min_pts=5, denoise=0.20 配置下约 135（person 过滤后），主要是 YOLO 误检 + 噪声聚类经几何 fallback 误判

**当前主要瓶颈：** Person Recall ~52-57%（空间 Recall ~71%），差距主要在稀疏远距离行人（8-10m）的聚类点数不足、以及被地面/墙体过滤误删。高 precision 配置（min_pts=5, denoise=0.20）的 FP 已控制在较低水平但 recall 受限。
