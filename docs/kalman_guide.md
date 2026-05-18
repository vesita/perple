# 卡尔曼滤波模块使用指南

## 概述

Perple 项目主推基于**恒加速度模型**（Constant Acceleration Model，9D CA 模型）的卡尔曼滤波器 `KalmanFilterCA`，用于行人目标跟踪中的状态估计和预测。同时也保留了一个 6D 常速模型（CV 模型）`KalmanFilterWrapper` 作为备选。

**核心功能**：

- 9 维状态向量（位置 + 速度 + 加速度 + 三维尺寸）估计
- 恒加速度运动模型预测
- 全状态直接观测更新（含新息门控降级）
- Z 轴通过独立 EMA 跟踪（不纳入卡尔曼状态，降低维度并避免 Z 轴噪声污染平面估计）
- 距离自适应 / 置信度自适应测量噪声
- 速度 + 加速度 + 尺寸限幅

## 架构设计

### 核心组件

**源代码位置**: [`src/tracker/kalman.rs`](../src/tracker/kalman.rs)（模块根 + `kalman/` 子模块）

```
tracker::kalman 模块
├── ca.rs
│   └── KalmanFilterCA (主推)         # 9D 恒加速度滤波器
│       ├── ConstantAccelerationModel   # 恒加速度运动模型
│       └── FullStateObservationModel9  # 9D 全状态观测模型
├── cv.rs
│   └── KalmanFilterWrapper (备选)     # 6D 常速度滤波器
│       ├── ConstantVelocityModel       # 常速度运动模型
│       └── FullStateObservationModel6  # 6D 全状态观测模型
└── KalmanConfigCA / KalmanConfig  # 配置参数
│   ├── ConstantAccelerationModel   # 恒加速度运动模型
│   └── FullStateObservationModel9  # 9D 全状态观测模型
├── KalmanFilterWrapper (备选)     # 6D 常速度滤波器
│   ├── ConstantVelocityModel       # 常速度运动模型
│   └── FullStateObservationModel6  # 6D 全状态观测模型
└── KalmanConfigCA / KalmanConfig  # 配置参数
```

### 状态向量定义

**9D 状态向量 (CA 模型)**: `[x, y, vx, vy, ax, ay, l, w, h]ᵀ`

| 索引 | 分量 | 含义 | 单位 |
|------|------|------|------|
| 0, 1 | x, y | XY 平面质心位置 | m |
| 2, 3 | vx, vy | 平面速度 | m/s |
| 4, 5 | ax, ay | 平面加速度 | m/s² |
| 6, 7, 8 | l, w, h | 三维尺寸（长宽高） | m |

**Z 轴** 不纳入卡尔曼状态，而是通过独立的指数移动平均（EMA）在 `TrackObject` 层跟踪：

```rust
// object.rs
self.z_ema = Z_ALPHA * centroid[2] + (1.0 - Z_ALPHA) * self.z_ema;
```

这样降低了状态维度，避免了 Z 轴噪声（如地面反射波动）污染 XY 平面估计。

### 两个滤波器的选择

| 特性 | KalmanFilterCA (主推) | KalmanFilterWrapper (备选) |
|------|----------------------|---------------------------|
| 状态维度 | 9 | 6 |
| 运动模型 | 恒加速度 (CA) | 常速度 (CV) |
| 观测维度 | 9 (全状态) | 6 (位置+速度) |
| Z轴 | 不包含 (外部EMA) | 包含 (z, vz) |
| 门控降级 | correct_with_gating | correct_position |
| 自适应噪声 | adjust_noise_for_distance/confidence | adjust_noise_for_distance |

## 数学模型 (CA 模型)

### 状态转移方程

恒加速度模型假设目标在两个时刻之间以恒定加速度运动。状态转移矩阵 F：

```
x_{k|k-1} = F · x_{k-1|k-1},  F 如下：
```

```
[1  0  Δt  0  ½Δt²  0  0  0  0]
[0  1  0  Δt  0  ½Δt²  0  0  0]
[0  0  1  0   Δt   0  0  0  0]
[0  0  0  1   0   Δt  0  0  0]
[0  0  0  0   1    0  0  0  0]
[0  0  0  0   0    1  0  0  0]
[0  0  0  0   0    0  1  0  0]
[0  0  0  0   0    0  0  1  0]
[0  0  0  0   0    0  0  0  1]
```

对应代码 (kalman/ca.rs:35-53)：

```rust
f[(0, 2)] = dt;   // x ← vx
f[(1, 3)] = dt;   // y ← vy
f[(0, 4)] = dt2;  // x ← ax  (dt2 = 0.5*dt*dt)
f[(1, 5)] = dt2;  // y ← ay
f[(2, 4)] = dt;   // vx ← ax
f[(3, 5)] = dt;   // vy ← ay
// 尺寸 (l, w, h) 为恒等（无动力学耦合）
```

### 过程噪声协方差 Q

Q 为对角矩阵，各分量独立配置，且随 dt 线性缩放：

```
Q_diag = [q_pos², q_pos², q_vel², q_vel², q_acc², q_acc², q_size², q_size², q_size²] · dt
```

| 参数 | 默认值 | 物理含义 |
|------|--------|----------|
| `kf_process_noise_pos` | 0.1 | 位置过程噪声 |
| `kf_process_noise_vel` | 0.05 | 速度过程噪声 |
| `kf_process_noise_acc` | 1.0 | 加速度过程噪声（较大，允许灵活变化） |
| `kf_process_noise_size` | 0.01 | 尺寸过程噪声（尺寸变化慢） |

### 观测模型

观测模型为全状态直接观测，观测矩阵 H 为单位阵 I₉：

```
z_t = [x, y, vx, vy, ax, ay, l, w, h]ᵀ
```

观测量构成：
- **位置 (x, y)**: 直接来自融合模块输出的点云质心
- **速度 (vx, vy)**: 由多帧位置差分推导（非直接观测量，因此测量噪声较大）
- **加速度 (ax, ay)**: 由速度差分推导（极不可靠，测量噪声最大）
- **尺寸 (l, w, h)**: 直接来自点云聚类结果（相对可靠）

测量噪声协方差 R 为对角矩阵：

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `kf_measurement_noise_pos` | 0.3 | 位置测量噪声 |
| `kf_measurement_noise_vel` | 0.8 | 速度测量噪声（由差分推导，不可靠） |
| `kf_measurement_noise_acc` | 2.0 | 加速度测量噪声（极不可靠） |
| `kf_measurement_noise_size` | 0.2 | 尺寸测量噪声（相对可靠） |

### 新息门控 (Gating)

`correct_with_gating()` 计算位置 (x,y) 的马氏距离，超过门控阈值时降级为仅位置修正，跳过速度和加速度的更新，防止异常观测导致状态发散。

```rust
pub fn correct_with_gating(&mut self, measurement: SVector<f64, 9>, gate_threshold: f64)
```

- 默认阈值：`kf_gate_threshold = 3.5`
- 马氏距离 ≤ 3.5：正常全状态修正
- 马氏距离 > 3.5：仅用 (x,y) 位置修正，速度/加速度/尺寸不变

这一机制在目标频繁遮挡、检测跳变场景下提升了滤波器健壮性。

### 自适应测量噪声

两个自适应机制，按调用顺序组合：

1. **距离自适应** (`adjust_noise_for_distance`): 远距离点云稀疏 → 质心不可靠 → 增大测量噪声
   - `scale = 1 + distance / 10.0`（10m 处 2x，20m 处 3x）

2. **置信度自适应** (`adjust_noise_for_confidence`): 低置信度检测 → 增大测量噪声
   - `scale = (1 + d/10) * (1 + (1-c)*3)`
   - confidence ∈ [0, 1]，0.5 时噪声放大约 2.5x

### 状态限幅

每次更新后调用 `clamp_state()` 对状态进行物理合理性约束：

| 参数 | 默认值 | 限幅对象 |
|------|--------|----------|
| max_speed | 3.0 m/s | vx, vy |
| max_accel | 10.0 m/s² | ax, ay |
| min_size | 0.05 m | l, w, h |
| max_size | 20.0 m | l, w, h |

## 快速开始

### 1. 使用 KalmanFilterCA

```rust
use perple::tracker::kalman::{KalmanFilterCA, KalmanConfigCA};
use nalgebra::SVector;

// 创建默认配置的 9D CA 滤波器
let mut kf = KalmanFilterCA::new(KalmanConfigCA::default())?;

// 初始化状态：9 维 [x, y, vx, vy, ax, ay, l, w, h]
let initial_state = SVector::<f64, 9>::from_column_slice(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.3, 1.7]);
kf.init_with_state(initial_state);

// 预测 - 更新循环
loop {
    let dt = get_time_delta();  // 从时间戳计算
    kf.predict(dt)?;

    if let Some(measurement) = get_detection() {
        // 标准全状态修正
        kf.correct(measurement)?;
        // 或使用门控修正（推荐）
        // kf.correct_with_gating(measurement, 3.5)?;

        // 状态限幅
        kf.clamp_state(3.0, 10.0, 0.05, 20.0);
    }

    let pos = kf.get_position();
    let vel = kf.get_velocity();
    let acc = kf.get_acceleration();
    let size = kf.get_size();
}
```

### 2. 自定义配置

```rust
let config = KalmanConfigCA {
    dt: 0.04,                    // 25Hz 采样
    process_noise_pos: 0.1,
    process_noise_vel: 0.05,
    process_noise_acc: 1.0,
    process_noise_size: 0.01,
    measurement_noise_pos: 0.3,
    measurement_noise_vel: 0.8,
    measurement_noise_acc: 2.0,
    measurement_noise_size: 0.2,
    initial_covariance_scale: 1.0,
};
let mut kf = KalmanFilterCA::new(config)?;
```

## 参数调优指南

### 时间步长 dt

- **推荐值**: 与传感器采样频率匹配（perple 默认 25Hz → dt=0.04）
- **动态调整**: 滤波器在每次 `predict()` 时根据帧间时间戳实时计算 dt
- **影响**: dt 越大，预测步长越大，Q 也按比例放大（不确定性增加）

### 过程噪声

- `kf_process_noise_pos` (0.1): 位置预测信任度，越小越相信预测位置
- `kf_process_noise_vel` (0.05): 速度变化灵活性，目标转向频繁可适当增大
- `kf_process_noise_acc` (1.0): **加速度过程噪声较大**，因为行人运动并非理想 CA，加速度变化通过较大的 Q 吸收模型误差
- `kf_process_noise_size` (0.01): 尺寸变化慢，保持小值

### 测量噪声

- `kf_measurement_noise_pos` (0.3): 点云质心有一定噪声，不过度信任单帧观测
- `kf_measurement_noise_vel` (0.8): 速度由 k 帧位置差推导，不可靠，噪声较大
- `kf_measurement_noise_acc` (2.0): 加速度由速度差推导，极不可靠，噪声最大
- `kf_measurement_noise_size` (0.2): 聚类框尺寸相对稳定，噪声适中

调参可通过 `--tracker-toml` 运行时覆盖：
```bash
cargo run --release --example eval_ablation -- --tracker-toml \
  'kf_measurement_noise_pos=0.5 kf_process_noise_acc=2.0'
```

### 门控阈值

- `kf_gate_threshold` (3.5): 马氏距离门控
  - 调小 → 更多观测被门控拒绝（更保守，防止异常观测污染状态）
  - 调大 → 更多观测被接受（更激进，适合噪声较小的场景）

## 高级功能

### 丢失观测处理

当目标被遮挡时，滤波器仅执行预测步骤，不进行修正：

```rust
if let Some(measurement) = get_detection() {
    kf.correct_with_gating(measurement, 3.5)?;
} // 无检测：仅 predict 已在上层调用
```

长时间丢失观测时协方差不断增大，可通过 `TrackObject.max_disappeared` 控制轨迹删除时机（默认 12 帧）。

### 噪声自适应

远距离 + 低置信度检测的双重自适应：

```rust
// 距离自适应（LV-DOT 风格）
kf.adjust_noise_for_distance(distance);

// 置信度 + 距离自适应（推荐）
kf.adjust_noise_for_confidence(distance, confidence);
```

### 状态限幅

防止滤波器发散（被极端观测拉偏）：

```rust
kf.clamp_state(max_speed, max_accel, min_size, max_size);
// 当前配置：3.0 m/s, 10.0 m/s², 0.05 m, 20.0 m
```

## 性能指标

- **单次 predict + correct**: < 1μs（微秒级）
- **内存占用**: 单个滤波器约 500 bytes（9D 相对于 6D 增加约 2x）
- **适用于实时系统**: >100Hz 采样频率无忧

## 故障排查

| 症状 | 可能原因 | 解决方案 |
|------|----------|----------|
| 滤波器发散 | dt 与实际采样不匹配 | 检查时间戳计算 |
| 响应过慢 | Q 太小或 R 太大 | 增大 process_noise_vel/acc |
| 抖动严重 | 测量噪声过小 | 增大 measurement_noise_pos |
| 异常观测拉偏 | 门控阈值过大 | 调小 kf_gate_threshold |
| Z 轴波动大 | 独立 EMA 参数 Z_ALPHA 不合适 | 调整 object.rs 中 Z_ALPHA |

## 配置参考

完整的 tracker 配置项见 `config/default.toml` `[tracker]` 节：

```toml
[tracker]
# Kalman 9D CA 模型参数
kf_process_noise_pos = 0.1
kf_process_noise_vel = 0.05
kf_process_noise_acc = 1.0
kf_process_noise_size = 0.01
kf_measurement_noise_pos = 0.3
kf_measurement_noise_vel = 0.8
kf_measurement_noise_acc = 2.0
kf_measurement_noise_size = 0.2
kf_initial_covariance_scale = 1.0
kf_gate_threshold = 3.5
```

所有参数可在运行时通过 `eval_ablation --tracker-toml` 覆盖，便于自动调参。
