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
| `--iou` | 0.15 | IoU 匹配阈值（仅 `--center-dist 0` 时生效） |
| `--center-dist` | 0.5 | **中心距匹配阈值(m)，默认开启。** 中心距 0.5m 是行人类小目标的最稳定匹配方式（见下方说明） |
| `--bev-iou` | false | 使用 BEV 2D IoU 替代 AABB IoU。与 `--center-dist 0` 配合使用 |
| `--frames` | 全部 | 评测帧数 |
| `--skip` | 0 | 跳过的初始帧数 |
| `--output` | 自动 | 输出目录 |
| `--disable-yolo-smooth` | false | 关闭 YOLO 帧间标签平滑（测试平滑影响时用） |

**三种匹配方式说明：**

| 方式 | 命令 | 适用场景 | 行人 0~10m 典型 F1 |
|------|------|----------|:---:|
| **中心距 0.5m（默认/推荐）** | `--center-dist 0.5` | **行人类小目标，与 nuScenes 一致** | **0.68~0.75** |
| BEV 2D IoU | `--center-dist 0 --bev-iou --iou 0.30` | 需要 2D 空间精度的场景，与 KITTI BEV 评估类似 | 0.39~0.43 |
| 3D AABB IoU | `--center-dist 0 --iou 0.15` | 传统 3D 目标检测评估（注意 AABB 膨胀导致 IoU 虚高） | 0.42~0.57 |

> **关于评估方法的选择：** 行人检测评估中，3D IoU（无论是 AABB 近似还是真 OBB）对小目标过于敏感——行人体积约 0.6×0.6×1.8m（~0.65m³），15cm 的定位误差即可使 3D IoU 从 1.0 降至 ~0.45。成熟基准（nuScenes、KITTI 行人）均采用中心距或 BEV IoU。本系统默认使用**中心距 0.5m**，与 nuScenes 一致，同时在可用时提供 `--bev-iou` 选项。

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
| `--iou` | 0.15 | IoU 匹配阈值（仅在 `--center-dist 0` 时生效） |
| `--center-dist` | 0.5 | 中心距匹配阈值(m)，开启后替代 IoU |
| `--bev-iou` | false | 使用 BEV 2D IoU 替代 AABB IoU（需配合 `--center-dist 0`） |
| `--frames` | 全部 | 评测帧数 |
| `--output` | 自动 | 输出目录 |
| `--config` | 无 | 完整 TOML 配置文件路径 |
| `--ground-toml` | 无 | 地面参数覆盖（逗号分隔） |
| `--cluster-toml` | 无 | 聚类参数覆盖（逗号分隔） |
| `--denoise-toml` | 无 | 降噪参数覆盖（逗号分隔） |
| `--tracker-toml` | 无 | 跟踪参数覆盖（逗号分隔） |

**KF 调参 CLI 示例：**

```bash
# 调大速度测量噪声 → 更平滑的速度跟踪
cargo run --release --example eval_ablation -- \
    --tracker-toml 'kf_measurement_noise_vel=1.2,kf_measurement_noise_acc=3.0' \
    --center-dist 0.5 --frames 408

# 调大门控阈值 → 接受更多测量
cargo run --release --example eval_ablation -- \
    --tracker-toml 'kf_gate_threshold=4.5' \
    --center-dist 0.5 --frames 408

# 调小位置噪声 → 位置跟踪更紧
cargo run --release --example eval_ablation -- \
    --tracker-toml 'kf_measurement_noise_pos=0.2' \
    --center-dist 0.5 --frames 408

# 完整 KF 参数调整
cargo run --release --example eval_ablation -- \
    --tracker-toml 'kf_measurement_noise_pos=0.4,kf_measurement_noise_vel=1.0,kf_measurement_noise_acc=2.5,kf_process_noise_vel=0.08,kf_gate_threshold=4.0,kf_avg_frames=10' \
    --center-dist 0.5 --frames 408
```

**评估方法：** 每次配置运行 **2 次** 取平均（因 YOLO ONNX 推理非确定性）。

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

# Step 2: 精度评估（中心距 0.5m，默认）
cargo run --release --example eval_labeled -- \
    --center-dist 0.5 --output ./output/final_eval

# Step 2b: BEV IoU 评估（可选）
cargo run --release --example eval_labeled -- \
    --center-dist 0 --bev-iou --iou 0.30 --output ./output/bev_eval

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
| `strategy` | `"prune_qt"` | 聚类策略（四叉树剪叶过滤 + DBSCAN） |
| `min_occ` | 4 | 剪叶最小点数（叶节点 ≥ min_occ 才保留质心） |
| `min_points_per_cluster` | 5 | DBSCAN 核心点最少邻点数 |
| `merge_patience` | 0.20 | DBSCAN 邻域半径 eps（米） |
| `eps_slope` | 0.05 | 自适应 eps 斜率 |
| `voxel_size` | 0.10 | 体素下采样格子大小（米） |
| `denoise_radius` | 0.20 | 聚类前半径离群点剔除半径（米） |
| `denoise_min_pts` | 3 | 降噪最小邻点数 |
| `density_weight_alpha` | 2.0 | 密度感知质心加权指数：`r^α` 补偿 LiDAR 近密远疏导致的质心偏移 |
| `default_confidence_threshold` | **0.5** (原 0.6) | YOLO 置信度阈值 |
| `default_nms_threshold` | 0.7 | YOLO NMS 阈值 |
| `max_range` | 10.0 | 有效检测距离上限（米），超出此距离的点在聚类前被过滤 |
| `downsample_method` | `"voxel"` | 下采样方法 |
| `min_appearances` | **1** (原 2) | 轨迹出现帧数低于此值不输出 |
| `point_vel_threshold` | 0.08 | 点云投票位移阈值(m) |
| `moving_speed_threshold` | 0.35 | 运动速度阈值(m/s) |
| `use_point_cloud_voting` | true | 点云投票开关 |
| `kf_avg_frames` | **8** (原 5) | Kalman 滤波器平滑窗口帧数 |
| `geo_pass_threshold` | 6 | 几何验证连续通过帧数 |
| `geo_fail_threshold` | 5 | 几何验证连续失败帧数 |
| `geo_speed_threshold` | 0.6 | 速度激活阈值(m/s) |
| `wall_distance` | 0.08 | BevEdLines 墙体提取点到直线距离阈值（米） |
| `wall_strategy` | `"bev_edlines"` | 墙体提取策略: `bev_lsd` / `bev_edlines` / `bev_hough` |

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
- 聚类: `prune_qt`（四叉树剪叶过滤 + DBSCAN, eps=0.20, min_occ=4, min_pts=5, denoise_radius=0.20）
- 密度加权: `r^α`（α=2.0）
- 后聚类过滤链: 7 道（尺寸、扁度、体积、Z 中心、边界、稀疏度）
- 跟踪: 点云投票 + 几何 fallback + 航迹评分 + BTreeMap
- 跟踪过滤: `min_appearances=1`（原 2），短轨迹也输出以提升召回
- YOLO: 置信度 0.5（原 0.6）+ 帧间标签平滑
- 几何后端: `geo_pass_threshold=6`（连续通过帧数），`geo_fail_threshold=5`，`geo_speed_threshold=0.6 m/s`

指标概要（中心距 0.5m 匹配，408 帧，二进制方向 EDLines）：

```
── Person 过滤 (仅 class_type == "person") ──
  GT: 1224  | 检测:  992  | TP:  844  FP:  148  FN:  380
  Precision: 85.1%  | Recall: 69.0%  | F1: 0.762

── 全部类别 (All Classes) ──
  GT: 1224  | 检测: 1639  | TP:  964  FP:  675  FN:  260
  Precision: 58.8%  | Recall: 78.8%  | F1: 0.673
```

> **注：** 以上为中心距 0.5m 匹配的最新结果（二进制方向 EDLines）。不同评估方式对指标有显著影响：中心距 0.5m 的 F1（0.762/0.673）显著高于 AABB IoU（~0.57）和 BEV IoU 0.3（~0.39），这是因为 IoU 类评估对行人小目标过于敏感，15cm 定位误差即可使 IoU 从 1.0 降至 ~0.45。中心距匹配（nuScenes 标准）消除了 IoU 的体积敏感性，是行人类检测最稳定的评估方式。

**改进对比（与旧默认 dbscan_qt 相比，二进制方向 EDLines）：**

| 指标 | dbscan_qt（旧） | prune_qt（新） | 变化 |
|------|:---:|:---:|:---:|
| Person Precision | 61.4% | **85.1%** | **+23.7pp** |
| Person Recall | 56.7% | **69.0%** | **+12.3pp** |
| Person F1 | 0.589 | **0.762** | **+0.173** |
| All Recall (空间召回) | 65.0% | **78.8%** | +13.8pp |
| FP (Person) | 437 | **148** | **-66%** |
| 正确分类率 | 55.6% | **67.8%** | +12.2pp |

**跟踪器性能评估：**

从空间评估与严格评估的交叉分析可以看出，跟踪器在"检测到→输出正确标签"这一核心链路中表现优异：

- 空间匹配到的 964 个 GT 行人中，**830 个被正确分类为 person**，正确率 **86.1%**
- 134 个（13.9%）被误分类为 obstacle
- 这说明：**只要上游聚类把人的点云检出，跟踪器几乎总能把它标对**。硬锁标签保护 + 几何累加器 + 点云投票的组合策略有效发挥了跨帧标签传播的作用，跟踪分类不是当前系统的性能瓶颈。

**prune_qt 策略优势分析：** prune_qt 的核心优势在于：
1. **墙体预处理**：BevEdLines 先移除墙面点，避免墙体残留产生大量 FP
2. **四叉树剪叶过滤**：仅保留密集叶节点（min_occ≥4），天然抑制稀疏噪声
3. **四叉树加速 DBSCAN**：在叶节点质心上运行 DBSCAN（eps=0.20），聚类质量高

**当前瓶颈：** Person Recall 69.0%，即 31.0% 的 GT 行人在聚类阶段未被检出。主要原因：(1) 墙面提取误删行人点，(2) 远处行人（8-10m）点数不足被剪叶丢弃。二进制方向 EDLines 将 Recall 从 66.4% 提升到 69.0%，FP 维持低位（148）。

**速度：** 408 帧 / ~17-25s = **~16-24 FPS**（Debug 模式较慢，Release 约 24 FPS）。

**非确定性说明：** YOLO (ONNX Runtime on DirectML) 推理不同 run 产生不同检测结果（检测数波动 ~200），是 F1 波动主因。跟踪器已通过 BTreeMap 消除自身非确定性。

### 7.2 聚类策略对比实验

2026-05-16 对所有可用聚类策略进行 408 帧全量评估（中心距 0.5m），按 Person F1 排序：

| 排名 | 策略 | Person P | Person R | Person F1 | All R | TP | FP | FN |
|:---:|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | **prune_qt** | **79.3%** | 58.6% | **0.674** | **70.5%** | 717 | **187** | 507 |
| 2 | xy_grid_dbscan | 60.1% | 59.6% | 0.598 | 65.0% | 729 | 484 | 495 |
| 3 | dbscan_qt | 61.4% | 56.7% | 0.589 | 65.0% | 694 | 437 | 530 |
| 4 | lvdot | 74.9% | 47.3% | 0.580 | 57.4% | 579 | 194 | 645 |
| 5 | dbscan_light | 58.1% | 57.7% | 0.579 | 66.4% | 706 | 510 | 518 |
| 6 | cc | 51.7% | **61.2%** | 0.560 | **67.3%** | **749** | 700 | 475 |
| 7 | dbscan_grid | 50.3% | 60.6% | 0.550 | 70.1% | 742 | 734 | 482 |
| 8 | range_image | 67.7% | 34.6% | 0.458 | 53.0% | 423 | 202 | **801** |
| 9 | seq | 70.3% | 10.0% | 0.176 | 10.0% | 123 | 52 | 1101 |

**关键发现：**

- **prune_qt 全面领先**：相比 dbscan_qt，Precision +17pp, F1 +0.085, FP 降低 54%（437→199）
- **cc 召回最高**（61.2%）但 Precision 仅 51.7%，FP 过多
- **lvdot** 精度高（74.9%）但召回低（47.3%）
- **seq / range_image** 不适合行人检测场景

**prune_qt + wall_distance 联合调参**（YOLO 非确定性导致 ~0.02 F1 波动，二进制方向 EDLines）：

| 配置 | Person P | Person R | Person F1 | FP |
|------|:---:|:---:|:---:|:---:|
| wd=0.08, min_occ=4, eps=0.20（当前最优） | **85.1%** | **69.0%** | **0.762** | **148** |
| wd=0.08, min_occ=4, eps=0.30 | 72.5% | 58.5% | 0.647 | 272 |
| wd=0.05, min_occ=4, eps=0.20 | 73.4% | 57.7% | 0.646 | 256 |
| wd=0.08, min_occ=3, eps=0.30（旧默认） | 78.4% | 59.1% | 0.674 | 199 |
| wd=0.05, min_occ=3, eps=0.30 | 71.0% | 61.5% | 0.659 | 307 |

参数建议：wd=0.08 / min_occ=4 / eps=0.20 在 FP 控制和 F1 间取得最佳平衡。

**跟踪过滤参数调优（conf=0.5 基础上）：**

| 参数变化 | Person P | Person R | F1 | FP |
|:--|:---:|:---:|:---:|:---:|
| 基线 | ~90% | ~49% | ~0.63 | ~66 |
| `min_appearances=1` | 92.3% | **54.5%** | 0.685 | 56 |
| `track_score_output_threshold=1.0` | 92.2% | **53.8%** | 0.680 | 56 |
| `geo_pass_threshold=4` | 86.3% | **57.6%** | **0.691** | 112 |

`min_appearances=1` 在召回 +6pp 同时 FP 未增加，已设为默认值。

### 7.3 消融对比

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

### 7.4 标签传播方案对比实验

尝试用三种贝叶斯标签置信度滤波器替代 `correct()` 中的硬锁 `if !(self.class_type=="person" && new!="person")`：

| 方案 | 机制 | Person P | Person R | Person F1 | 误分类率 | 结论 |
|------|------|:---:|:---:|:---:|:---:|------|
| 基线硬锁 | 一旦 person 永不翻转 | 61.2% | **57.5%** | **0.593** | **12.1%** | 当前最优 |
| Log-Odds | `l += ±1.2~1.8`, 阈值 -0.5 | 62.1% | 54.7% | 0.582 | 16.4% | 软阈值翻转过激 |
| 离散贝叶斯 | 转移矩阵 + 观测似然 | 62.1% | 55.2% | 0.585 | 15.5% | 略好于 Log-Odds |
| Beta 分布 | `α+=3`(person) / `β+=3`(not) | **62.9%** | 54.2% | 0.583 | 16.8% | Precision 最高 |

**结论：硬锁策略在该场景下最优。** 三个贝叶斯方案 Precision 微升但 Recall 下降 ~2-3pp，误分类率反升。原因是 YOLO "obstacle" 误分类远多于真实 obstacle，软概率在 YOLO 间歇性漏检时过早翻转正确标签。当下瓶颈在空间 Recall（65%），标签传播方案无法弥补这个 gap。

### 7.5 聚类策略工厂优化

**改动** (`src/cloud/classify/strategy.rs`): `prune_qt` 策略从硬编码参数改为从 `config/default.toml` 读取 `merge_patience`（→ eps）和 `min_points_per_cluster`（→ min_pts），使 prune_qt 支持 CLI 运行时参数覆盖，提升了可调性。

### 7.6 关键改动

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

### 7.7 误差分析

408 帧共 1224 个 GT Pedestrian（均在 0~10m 范围内）：

- **误分类（空间命中但 label 错误）**: ~135-198 (11-16%) → 空间匹配到但被标记为 obstacle（几何 fallback 未覆盖或 trick 条件不满足）
- **漏检（空间未命中）**: ~352-389 (29-31%) → LiDAR 预处理阶段丢失（地面/墙体误删、遮挡、聚类截断）
- **误报 FP**: 在 min_pts=5, denoise=0.20 配置下约 135（person 过滤后），主要是 YOLO 误检 + 噪声聚类经几何 fallback 误判

**当前主要瓶颈：** Person Recall ~52-57%（空间 Recall ~71%），差距主要在稀疏远距离行人（8-10m）的聚类点数不足、以及被地面/墙体过滤误删。高 precision 配置（min_pts=5, denoise=0.20）的 FP 已控制在较低水平但 recall 受限。
