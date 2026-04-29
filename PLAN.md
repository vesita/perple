# 开发计划：地面检测 + 跟踪器优化 + 融合

## 场景设定

- 小车载有 LiDAR + Camera，两者固连，在室内移动
- 三类目标：行人（dynamic）、墙体/固定物（static）、可移动物（movable）
- Perple 内部统一使用 **LiDAR 原生帧**，不做轴转换
- 轴转换只发生在可视化边界（redra 服务端）

---

## 设计决策

### 坐标系：仅 LiDAR 原生帧

| 模块 | 坐标系 | 说明 |
|------|--------|------|
| LiDAR 检测 | LiDAR 帧 | 保持传感器原始坐标 |
| 点云分类 | LiDAR 帧 | 聚类、地面检测都在此帧 |
| 目标跟踪 | LiDAR 帧 | Kalman 状态在此帧 |
| 2D→3D 融合 | LiDAR → Camera | `cam_from_lidar = camera.extrinsic`，P_cam = cam_from_lidar * P_lidar，固定标定参数 |
| 可视化 | LiDAR 帧 → redra | redra 服务端做 LiDAR→Bevy Y-up 转换 |

世界帧概念已被移除。LiDAR 帧就是 perple 的权威帧。

### 地面检测：Histoseed 混合策略

```
步骤 1: Z 直方图种子 — 建直方图找峰值 Z，±expand 取种子区域
步骤 2: RANSAC 平面拟合 — 仅对种子区域迭代拟合最佳平面
步骤 3: 生长 — 用最佳平面距离阈值筛选全点云
```

支持 **倒装 LiDAR**（Z 轴朝下），检测前对 Z 取反归一化。天花板检测沿用直方图方法。

**最优参数**：expand=0.20, ransac_distance=0.3, iterations=100

### Kalman 状态

```
状态: [x, y, z, vx, vy, vz]        6 维，常速模型（CV Model）
测量: 检测框中心 [x, y, z]          LiDAR 帧
速度: 由 Kalman 隐式推断           无需外部输入
```

### 动/静分类方法：速度空间聚类

每帧收齐所有 tracked object 的 Kalman 速度向量，做聚类：

```
LiDAR 帧速度分布（小车以 v_car 前进）：

  墙面（x10 个目标）:   v = -v_car           ← 最大簇 → STATIC
  柱子（x3 个目标）:     v = -v_car           ← 同一簇
  行人:                 v = +0.5 - v_car      ← 偏离簇 → DYNAMIC
  被推的椅子:            v = +0.2 - v_car      ← 边缘 → MOVABLE
```

**具体算法**：对速度向量 `[vx, vy, vz]` 做 DBSCAN 聚类（ε 默认 ~0.3 m/s）：

| 簇大小 | 分类 | 依据 |
|--------|------|---------|
| 最大簇 | static | 占据场景大多数，反映自车速度 |
| 偏离最大簇 | dynamic | 速度显著不同于自车 |
| 边缘 | movable | 暂不确定，需多帧观察 |

不需要 ICP、不需要 IMU、不需要里程计。墙面占室内场景多数，聚类天然鲁棒。

---

## 已完成

### ✅ LiDAR 帧改造
- `cloud/core.rs`：移除 `world` 坐标变换，点云保持 LiDAR 原生帧
- `fuse.rs`：新增 `cam_from_lidar` 直接投影，移除 `cloud_in_world` → `swapl` 重构

### ✅ 地面检测（environment.rs）
- `single_pick_ground` 实现 histoseed 混合策略（直方图种子 + RANSAC + 生长）
- `pick_ground` 旧版保留兼容
- 倒装 LiDAR 支持：Z 取反归一化
- 天花板检测（直方图峰值，配置开关）
- `histoseed_plane` 辅助函数：种子 RANSAC + 全点云生长 + 原地交换

### ✅ 跟踪器优化（tracker/）
- **kalman.rs**: 复用 `KalmanFilterNoControl`，动态 dt，降低初始协方差
- **core.rs**: 匈牙利算法 + 马氏距离关联，卡方门控，速度聚类分类
- **hungarian.rs**: 新增匈牙利算法实现
- **output.rs**: Target 扩展 velocity, speed, is_dynamic, classification 字段

### ✅ 数据加载器（data_loader.rs）
- 帧数限制 `frame_limit` / `set_frame_ratio`
- 下采样 `downsample`
- 独立路径模式 `new_independent`
- 缓冲区满处理（load 正常退出 / load_loop 静默跳过）

### ✅ 基准测试
- `ground_bench.rs`：5 种策略 × 29 参数组合对比
- `doc/ground_detection_conclusion.md`：结论报告
- 最优策略 histoseed（expand=0.20, distance=0.3）

---

## 当前状态

核心管线已整合，可视化通过 rerun 展示检测结果，redra 服务端做轴转换。

---

## 待办

### 短期
- [ ] 匈牙利算法参数调优（马氏距离门限）
- [ ] 速度聚类防抖动（多帧投票）
- [ ] 可视化着色（dynamic→红, static→绿, movable→黄）

### 中长期
- [ ] 外参验证工具（FAST-Calib 投影验证）
- [ ] ROS1 桥接（rosrust / C++ 桥节点）
- [ ] 多传感器时间同步
