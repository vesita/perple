# 聚类策略优化计划 — LV-DOT 启发

## Context

室内场景下当前 DBSCAN 产生过多无效簇（细柱/桌腿被误认为目标），跟踪器收到太多假阳性。LV-DOT（CMU 2025）的工程策略提供了可借鉴的优化方向。本计划将优化拆为 5 个独立 phase，每个 phase 产出可测试的增量改进。

---

## Phase 1: 预处理 — 高斯概率下采样（替代均匀体素）

**目标**：保留远处稀疏点、稀疏近处密集点，提升远距离检测能力

**LV-DOT 参考**：`exp(-d²/2σ²)` 保留概率随距离衰减，`σ=gaussian_downsample_rate=6`

**改动文件**：
- `src/cloud/classify/strategy/dbscan.rs` — `voxel_downsample()` 增加高斯模式
- `src/config.rs` — `ClasterConfig` 增加 `downsample_method: String` + `gaussian_downsample_rate: f32`
- `config/default.toml` — 增加对应字段

**实现方式**：
```
enum DownsampleMethod { Voxel(f32), Gaussian { leaf: f32, sigma: f32 } }
```
在 `DbscanStrategy` 初始化时根据 config 选择方法。Gaussian 模式下：先用 0.1m 体素粗略过滤，再以概率 exp(-d²/2σ²) 保留每个点，实现远密近疏。

**验证**：`cargo run --example cluster_bench` 对比 `voxel=0.1` 和 `gaussian` 的大目标数 + 耗时

---

## Phase 2: 策略后处理 — YOLO 辅助簇分裂/合并

**目标**：用 YOLO 2D 检测结果修正 3D 聚类错误（过分割/欠分割）

**LV-DOT 参考**：1 个 3D 簇匹配多个 YOLO 框时按 2D 投影点分配分裂；多个 3D 簇匹配 1 个 YOLO 框时合并

**改动文件**：
- `src/cloud/classify/claster.rs` — 新增 `refine_with_yolo()` 方法
- `src/cloud/classify/core.rs` — `Classify::act()` 中聚类后调用 YOLO 精炼
- `src/fuse.rs` — 调整执行时序到 Tracker 之前
- `src/perple.rs` — 调整 fuse_loop 在 tracker_loop 之前启动

**算法步骤**：
1. 调整 Fuse 时序：fuse_loop 在 tracker_loop 之前运行（perple.rs 中调整启动顺序）
2. `claster()` 完成后，Fuse 将 3D 簇与 2D YOLO 框融合（利用 swapl.clr_objs）
3. Tracker 消费已融合的 `Vec<CldBud>`（含 YOLO 类别标签）
4. **分裂**：1 个 3D 簇匹配多个 YOLO 框 → 将簇内点按投影到哪个 2D 框分配成子簇
5. **合并**：多个 3D 簇匹配同 1 个 YOLO 框 → 合并这些簇

**配置新增**：
- `claster.yolo_refine: bool = true`（是否启用）
- `claster.yolo_iou_threshold: f32 = 0.2`

**验证**：运行 pipeline，对比启用/关闭 YOLO refine 后的目标数

---

## Phase 3: 跟踪器 — 点云投票动态分类（替代速度 DBSCAN）

**目标**：更鲁棒的动/静分类，避免速度空间 DBSCAN 的间接误差

**LV-DOT 参考**：当前帧 vs N 帧前点云 → 最近邻找点对 → 方向点乘投票 → 连续 15 帧一致性检查

**改动文件**：
- `src/tracker/core.rs` — 新增 `classify_by_point_cloud_voting()` 方法
- `src/cloud/classify/core.rs` — 需要传递历史点云帧给跟踪器

**关键设计**：
- `TrackedObject` 增加 `point_cloud_history: Vec<Vec<[f32;3]>>`（存最近 5 帧）
- 对每个目标：当前帧点云 vs `skipFrame`(5) 帧前点云
- 每个点找最近邻 → 点对方向与 KF 速度做点乘 → 方向一致为有效投票
- `votes / total >= 0.8` 且 KF 速度 ≥ 0.2m/s → dynamic 候选
- 连续 15 帧为候选 → 标记 dynamic

**配置新增**：
- `tracker.use_point_cloud_voting: bool = false`
- `tracker.dynamic_vote_threshold: f32 = 0.8`
- `tracker.dynamic_consistency_frames: usize = 15`

**验证**：回放数据，对比新旧方法的速度分类结果一致性

---

## Phase 4: 跟踪器 — 箱体尺寸锁定 fix_size

**目标**：减少跟踪框抖动，稳定下游导航

**LV-DOT 参考**：跟踪 10 帧后，若尺寸变化比例 < 0.4，冻结箱体尺寸

**改动文件**：
- `src/tracker/core.rs` — `TrackedObject` 增加 `fix_size` 逻辑

**实现**：
- `TrackedObject` 增加 `fixed_box: Option<Box3D>`, `frame_count: usize`
- `correct()` 中：递增 `frame_count`，若 `frame_count > 10` 且尺寸变化率 < 0.4，锁定到上一帧尺寸
- `predict()` 输出使用 `fixed_box` 替代实时计算框

**配置新增**：
- `tracker.fix_size_frames: usize = 10`
- `tracker.fix_size_dim_thresh: f32 = 0.4`

**验证**：观察可视化中框体抖动是否减少

---

## Phase 5: 聚类参数调优 — 室内场景默认值

**目标**：收紧默认参数减少误报，同时保持召回率

**LV-DOT 参考**：eps=0.05, min_points=10(激光)/20(视觉)

**改动文件**：
- `config/default.toml` — 调整默认参数

**建议值**：
```
merge_patience = 0.15       # 之前 0.35（室内场景不需要太大 eps）
min_points_per_cluster = 8  # 之前 5
eps_slope = 0.03            # 之前 0.05（自适应幅度减小）
voxel_size = 0.1            # 不变
```

**验证**：`cargo run --example cluster_bench` 对比新旧参数的大目标数/噪声/耗时

---

## 实施顺序与依赖关系

```
Phase 1 (高斯采样)     ← 独立，可先做
    ↓
Phase 5 (参数调优)     ← 独立，可先做（或与 Phase 1 并行）
    ↓
Phase 2 (YOLO 精炼)   ← 依赖 Fuse/YOLO 模块运行
    ↓
Phase 3 (点云投票)     ← 独立，依赖历史帧存储
    ↓
Phase 4 (fix_size)    ← 依赖跟踪器，最后做
```

## 用户确认的决策

1. **Phase 2 YOLO 数据流**：调整 Fuse 到 Tracker 之前执行，让 Tracker 消费已融合的检测结果
2. **Phase 3 点云历史存储**：直接做，每个目标存最近 5 帧点云，20 目标 × 5 帧 × 100 点 ≈ 10KB

---

## Phase 6: 分类修复 — ground 强制 Static + 噪声抑制

**目标**：消除「地面 dynamic」的误判，降低 LiDAR 噪声引发的速度假阳性

### 6A: ground 强制 Static（高优先级）
**根因**：`Classify::act()` 将地面检测结果以 `CldBud` 形式写入 `cld_objs` 流，跟踪器将其与普通目标一起做速度聚类和点云投票。地面点云的微小质心漂移（~0.03m）经过 k 帧速度计算产生非零速度，导致 DBSCAN 将其归为 dynamic。

**改动文件**：
- `src/tracker/core.rs` — 在 `run()` 输出阶段添加硬编码规则：
  ```rust
  // ground 永远为 static
  if target.class_type == "ground" {
      target.is_dynamic = false;
      target.classification = "static".to_string();
  }
  ```
- 可选：`classify_by_velocity()` / `classify_by_point_cloud_voting()` 中跳过 ground 标注的目标

### 6B: 增大 R_vel 抑制速度噪声（中优先级）
**根因**：`measurement_noise_vel = 0.2` 意味着 KF 对速度观测的信任度过高。室内场景下 k 帧速度测量包含大量 centroid jitter 噪声，KF 没有充分滤波。

**改动文件**：
- `src/tracker/kalman.rs` — 调整默认值：
  ```rust
  measurement_noise_vel: 0.2 → 0.8   # 降低 KF 对速度观测的信任
  measurement_noise_pos: 0.1 → 0.2   # 位置观测噪声略增，适应室内遮挡
  process_noise_vel: 0.05 → 0.02     # 降低过程噪声，让预测更保守
  ```

### 6C: 提高 DBSCAN 速度阈值（低优先级）
**根因**：`0.5 m/s` 阈值过低，室内场景中噪声就能达到此值。

**改动文件**：
- `src/tracker/core.rs` — `classify_by_velocity()` 中：
  ```rust
  // 速度阈值 0.5 → 0.8 m/s（仅标记明确运动的目标为 Dynamic）
  if spd > 0.8 { Dynamic } else if spd > 0.2 { Movable }
  ```
- `classify_by_point_cloud_voting()` 速度下限 0.2 → 0.5 m/s

**验证**：运行完整 685 帧 pipeline，检查 ground 目标是否仍被标记为 dynamic；统计 dynamic/movable/static 分布

---

## Phase 7: 管线速度优化

**目标**：将单帧处理时间从 ~843ms 降至 ~200ms 以内

### 7A: pipeline_test.rs 去掉 spawn_blocking 开销（高优先级）
**根因**：每帧对 Lidar / Fuse / Tracker 依次调用 `tokio::task::spawn_blocking`，每次在线程池中做上下文切换。3 次 spawn_blocking × 685 帧 = 2000+ 次线程切换，每次 ~0.1ms，加上锁竞争，浪费大量时间。且模块本身是同步操作，不需要异步包装。

**改动文件**：
- `examples/pipeline_test.rs` — 移除 `Arc<Mutex>` 和 `spawn_blocking`，改为直接 `lock().unwrap()` + 同步调用

### 7B: 流读取优化（中优先级）
**根因**：每帧多次调用 `blocking_lock()` + `read()` 访问 Stream，高频锁竞争。

**改动文件**：
- `src/tracker/core.rs` — 缓存读取结果，减少 stream 访问次数
- `examples/pipeline_test.rs` — 合并统计信息读取，一次 lock 读取多个值

### 7C: 地面检测+RANSAC 优化（低优先级）
**根因**：histoseed + RANSAC 每次 ~40ms，可优化 RANSAC 迭代次数。

**改动文件**：
- `config/default.toml` — 减少 RANSAC 迭代次数 100 → 50

**验证**：`cargo run --example pipeline_test -- --frames 100` 对比优化前后的平均帧耗时

---

## Phase 8: 分类系统重构 — 单向 promotion 改为双向状态机

**目标**：解决「一旦标记 Dynamic 永不回退」的问题，提高分类鲁棒性

### 根因
当前点云投票是单向 promotion（只设 Dynamic，不设 Static）、DBSCAN 覆盖时没有降级逻辑。目标一旦被标记为 Dynamic，即使停止运动也永远停留在 Dynamic 状态。

### 改动文件
- `src/tracker/core.rs` — 重构分类逻辑为状态机：
  ```
  Unknown → [速度聚类] → Static / Movable / Dynamic
  Static → [点云投票] → Dynamic（连续 15 帧一致）
  Dynamic → [速度聚类重置] → Static（连续 30 帧 speed < 0.2 m/s）
  Movable → [速度聚类] → Static（属最大簇）/ Dynamic（speed > 0.8）
  ```
- 点云投票不再无条件 promotion，增加降级检测：
  - 若连续 30 帧 speed < 0.2 m/s 且点云投票 ratio < 0.5，降级为 Static

### 验证
对比 685 帧中首次被标记 Dynamic 后又静默的目标数

---

## 实施顺序与依赖关系

```
Phase 6A (ground 强制 Static)  ← 独立，最优先
    ↓
Phase 6B (KF 噪声参数)        ← 独立，可与 6A 并行
    ↓
Phase 6C (DBSCAN 阈值)        ← 依赖 6B 完成后确认效果
    ↓
Phase 7A (去掉 spawn_blocking) ← 独立
    ↓
Phase 7B (流读取优化)          ← 独立，可与 7A 并行
    ↓
Phase 7C (RANSAC 参数)         ← 独立，低优先级
    ↓
Phase 8 (分类状态机)           ← 依赖 6A-6C 的效果评估
```

## 验证总览

每个 Phase 验证方式：
1. `cargo build` — 编译通过
2. `cargo run --example pipeline_test -- --frames 50 --csv log/phaseX.csv` — 确认分类正确性 + 速度
3. 检查 CSV 中 ground 目标的 is_dynamic 是否始终为 false
4. 确认 dynamic 目标速度 > 0.5 m/s 且持续运动

---

## Phase 9: 碰撞箱中心位置优化

**根因**：LiDAR 扫描存在固有缺陷——只有物体的近端被采样，远端完全空白。加上径向扫描模式导致近密远疏，质心/中心被系统性拉向传感器方向，引入持续误差。

### 参考工作

- **CornerPoint3D (2025)**：提出预测最近角代替中心，因为 LiDAR 只能看到物体近端，中心预测跨域不稳定。参见 [arXiv:2504.02464](http://arxiv.org/abs/2504.02464v1)
- **Corner-Aware 3D Detection (2025)**：系统对比 5 种角编码方案，中心对齐回归在 BEV 稀疏区域有根本性不稳定。F-Corner 编码在 KITTI 上 +3.5% AP。参见 [arXiv:2511.17619](https://arxiv.org/abs/2511.17619)
- **Chinese Patent (2024)**：凸包 + B 样条插值 → 最小外接矩形 + 高度，修正预标注框。参见 [CN202410948876](http://www.xjishu.com/zhuanli/55/202410948876.html)
- **Dynamic Clustering (2024)**：椭圆邻域自适应 DBSCAN 处理 LiDAR 非均匀分布，减少 70.6% 过聚类。参见 [汽车安全与节能学报](https://www.journalase.com/CN/10.3969/j.issn.1674-8484.2024.02.015)

### Phase 9A: 密度感知质心加权（低 effort，高 impact）

**问题**：当前 `cluster_box_and_centroid()` 中质心是点的算术平均。LiDAR 近密远疏导致近端点占主导，质心被拉向传感器。同样，AABB 中心也受近端密集点影响。

**方案**：对簇内每个点的坐标贡献按 1/r 或 1/r² 加权，补偿径向扫描密度差异：

```
加权质心 = Σ(w_i * p_i) / Σ(w_i)
w_i = 1.0 / (r_i^α)     // α=1 或 2，r_i 为点距传感器距离
```

**改动文件**：
- `src/cloud/classify/claster.rs` — `cluster_box_and_centroid()` 中质心计算增加加权选项
- 或在 `to_cldbuds()` 中增加 `centroid_mode: CentroidMode { Arithmetic, DensityWeighted { alpha: f32 } }`

### Phase 9B: AABB → OBB（定向包围盒）

**问题**：当前 `Box3D` 用 AABB（Axis-Aligned Bounding Box），框体方向与 LiDAR 帧轴对齐而非目标朝向。斜向物体的 AABB 远大于实际物体，中心误差大。

**方案**：用 PCA 计算主方向，拟合 OBB（Oriented Bounding Box）：

```
1. 对簇内点去中心化
2. 计算协方差矩阵 3×3
3. 特征值分解 → 主方向向量
4. 点云旋转到主方向坐标系 → AABB → 反旋转 → OBB
5. OBB 中心 = 反旋转后的 AABB 中心
```

**改动文件**：
- `src/utils/boxes.rs` — `Box3D` 增加 `from_points_pca(points: &[[f32; 3]]) -> Self` 构造方法
- `src/cloud/classify/claster.rs` — `cluster_box_and_centroid()` 或 `to_cldbuds()` 调用 PCA 版本

### Phase 9C: BEV 凸包 + 最小面积外接矩形

**问题**：PCA-OBB 对 L 形或不规则点云仍不稳定（主方向受 outlier 影响）。

**方案**：投影到 BEV（去掉 Z），计算凸包，旋转卡壳求最小面积外接矩形：

```
1. 取簇内点 (x, y)，忽略 z
2. 计算 2D 凸包（Graham scan / Andrew monotone chain）
3. 旋转卡壳（Rotating Calipers）求最小面积外接矩形
4. 矩形中心 + Z 方向 AABB → 3D 框中心
```

`Box3D` 已有 `vertices()` 和 `pose` 矩阵，可直接复用。凸包 + 旋转卡壳在点云数 < 1000 时极快。

**改动文件**：
- `src/cloud/classify/claster.rs` — 新增 `convex_hull_2d()` 和 `min_area_rect()` 辅助函数
- `src/utils/boxes.rs` — 新增 `from_points_convex_hull(points: &[[f32; 3]]) -> Self`

### Phase 9D: KF 位置优先覆盖（已部分实现）

**问题**：当前 Tracker 输出用 `kalman_filter.get_position()` 作为框中心，但 `TrackedObject::correct()` 的测量仍用原始 centroid。KF 虽然平滑了输出，但还没消除测量输入的系统偏差。

**方案**：
- KF 的 `measurement_noise_pos` 设为各向异性：远距离目标给更高的位置噪声（因为远距点更稀疏，框中心更不可靠）
- 或增加 `min_appearances` 帧后完全用 KF 预测位置替代测量（类似 fix_size 逻辑对位置生效）

**改动文件**：
- `src/tracker/kalman.rs` — `measurement_noise_pos` 改为距离自适应的函数
- `src/tracker/core.rs` — `correct()` 中增加 KF 位置信任度随 appearance_count 增长

### 验证方法

| Sub-phase | 验证方式 |
|-----------|---------|
| 9A | pipeline_test 对比加权前后同一目标的 centroid 偏移量 |
| 9B | 可视化对比 AABB vs OBB 框体方向（redra 可显示） |
| 9C | 与 9B 对比框面积/中心稳定性 |
| 9D | 长时间跟踪同一静止目标，计算 KF 位置 vs 原始 centroid 的摆动幅度 |

### 依赖关系

```
Phase 9A (密度加权质心)    ← 独立，最优先
    ↓
Phase 9B (PCA-OBB)        ← 依赖 9A 的效果评估
    ↓
Phase 9C (凸包+最小矩形)   ← 9B 不够好时的备选方案
    ↓
Phase 9D (KF 位置优先)     ← 独立，可与 9A 并行
```
