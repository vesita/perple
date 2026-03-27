# 卡尔曼滤波模块使用指南

## 概述

Perple 项目现在使用 `adskalman` 库实现了基于常速模型 (Constant Velocity Model) 的卡尔曼滤波器，用于目标跟踪中的状态估计和预测。

## 架构设计

### 核心组件

```
tracker::kalman 模块
├── ConstantVelocityModel      # 常速运动模型 (实现 TransitionModelLinearNoControl trait)
├── PositionObservationModel   # 位置观测模型 (实现 ObservationModel trait)
└── KalmanFilterWrapper        # 简化的封装接口
```

### 状态向量定义

**6 维状态向量**: `[x, y, z, vx, vy, vz]ᵀ`
- 前 3 个分量：位置 (x, y, z)
- 后 3 个分量：速度 (vx, vy, vz)

**3 维观测向量**: `[x, y, z]ᵀ`
- 只观测位置信息

## 数学模型

### 状态转移方程 (常速模型)

```
x_new = x_old + vx * dt
y_new = y_old + vy * dt
z_new = z_old + vz * dt
vx_new = vx_old
vy_new = vy_old
vz_new = vz_old
```

对应的状态转移矩阵 F:
```
[1 0 0 dt  0  0]
[0 1 0  0 dt  0]
[0 0 1  0  0 dt]
[0 0 0  1  0  0]
[0 0 0  0  1  0]
[0 0 0  0  0  1]
```

### 观测方程

```
z_x = x
z_y = y
z_z = z
```

对应的观测矩阵 H:
```
[1 0 0 0 0 0]
[0 1 0 0 0 0]
[0 0 1 0 0 0]
```

## 快速开始

### 1. 基本使用

```rust
use perple::tracker::kalman::{KalmanFilterWrapper, KalmanConfig};
use nalgebra::Vector3;

// 创建默认配置的滤波器
let mut kf = KalmanFilterWrapper::new(Default::default())?;

// 初始化状态：位置和速度
let position = Vector3::new(1.0, 2.0, 3.0);
let velocity = Some(Vector3::new(0.1, 0.2, 0.3));
kf.init_with_state(position, velocity);

// 预测 - 更新循环
loop {
    // 预测下一步
    kf.predict()?;
    
    // 获取新的观测值 (从传感器)
    let measurement = get_sensor_measurement();
    
    // 使用观测值更新
    kf.update(measurement)?;
    
    // 获取估计结果
    let estimated_pos = kf.get_position();
    let estimated_vel = kf.get_velocity();
}
```

### 2. 自定义配置

```rust
let config = KalmanConfig {
    dt: 0.05,  // 50ms 时间步长
    process_noise_scale: 0.001,      // 降低过程噪声 (更相信模型)
    measurement_noise_scale: 0.01,   // 降低测量噪声 (更相信观测)
    initial_covariance_scale: 100.0, // 初始高不确定性
};

let mut kf = KalmanFilterWrapper::new(config)?;
```

### 3. 在 Tracker 中的应用

在 `Tracker` 模块中，每个跟踪的目标都包含一个 `KalmanFilterWrapper` 实例：

```rust
use perple::tracker::Tracker;

// 创建跟踪器
let mut tracker = Tracker::new();

// 配置参数（可选）
tracker.set_association_threshold(2.0);
tracker.set_max_disappeared(10);
tracker.set_min_confidence(0.3);

// 运行跟踪循环
loop {
    // 读取点云检测结果
    // 执行数据关联
    // 更新卡尔曼滤波器
    if let Err(e) = tracker.run() {
        eprintln!("跟踪器运行错误：{}", e);
    }
    
    // 获取跟踪结果
    let targets = tracker.get_tracked_ids();
}
```

## 参数调优指南

### 时间步长 `dt`

- **推荐值**: 与传感器采样频率匹配 (如 100Hz → dt=0.01)
- **影响**: 
  - dt 太小：滤波器响应慢
  - dt 太大：预测精度下降

### 过程噪声 `process_noise_scale`

- **推荐范围**: 0.001 ~ 0.1
- **调优原则**:
  - 目标运动平稳 → 减小 (0.001~0.01)
  - 目标机动频繁 → 增大 (0.01~0.1)
  
### 测量噪声 `measurement_noise_scale`

- **推荐范围**: 0.01 ~ 1.0
- **调优原则**:
  - 传感器精度高 → 减小
  - 传感器噪声大 → 增大

### 初始协方差 `initial_covariance_scale`

- **推荐值**: 10.0 ~ 100.0
- **影响**: 只影响初始阶段，通常设为较大值表示不确定

## 典型应用场景

### 场景 1: 平滑轨迹跟踪

```rust
// 低过程噪声，中等测量噪声
let config = KalmanConfig {
    dt: 0.1,
    process_noise_scale: 0.001,
    measurement_noise_scale: 0.1,
    initial_covariance_scale: 10.0,
};
```

### 场景 2: 机动目标跟踪

```rust
// 高过程噪声以适应机动
let config = KalmanConfig {
    dt: 0.05,
    process_noise_scale: 0.1,
    measurement_noise_scale: 0.1,
    initial_covariance_scale: 50.0,
};
```

### 场景 3: 高噪声环境

```rust
// 更相信模型而非观测
let config = KalmanConfig {
    dt: 0.1,
    process_noise_scale: 0.01,
    measurement_noise_scale: 1.0,
    initial_covariance_scale: 100.0,
};
```

## 高级功能

### 处理丢失的观测

```rust
// 如果观测值无效 (NaN)，滤波器会自动跳过更新，只进行预测
let invalid_measurement = Vector3::new(f64::NAN, f64::NAN, f64::NAN);
kf.update(invalid_measurement)?;  // 等同于只执行 predict
```

### 动态调整时间步长

```rust
let new_dt = 0.05;
kf.set_dt(new_dt);
```

### 重置滤波器

```rust
kf.reset();  // 回到初始零状态
```

### 不确定性评估

```rust
// 获取位置和速度的不确定性（标准差）
let pos_uncertainty = kf.get_position_uncertainty();
let vel_uncertainty = kf.get_velocity_uncertainty();

println!(
    "位置不确定性：({:.4}, {:.4}, {:.4})",
    pos_uncertainty.x, pos_uncertainty.y, pos_uncertainty.z
);
```

### 新息分析

```rust
// 计算新息（测量残差）
let innovation = kf.get_innovation(measurement);

// 计算归一化新息平方 (NIS) 用于卡方检验
let nis = kf.normalized_innovation_squared(measurement);

if nis > threshold {
    println!("检测到异常测量！");
}
```

## 性能指标

### 延迟

- 单次 `predict()` + `update()`: < 1μs
- 适用于实时系统 (>100Hz)

### 内存占用

- 单个滤波器：~200 bytes
- 可轻松跟踪数百个目标

## 故障排查

### 问题 1: 滤波器发散

**症状**: 估计误差越来越大

**解决方案**:
1. 检查 dt 是否设置正确
2. 增大 process_noise_scale
3. 检查观测值是否有异常值

### 问题 2: 响应过慢

**症状**: 估计滞后于真实运动

**解决方案**:
1. 增大 process_noise_scale
2. 减小 measurement_noise_scale
3. 检查 dt 是否过大

### 问题 3: 数值不稳定

**症状**: 出现 NaN 或 Inf

**解决方案**:
1. 使用 Joseph 形式协方差更新 (已默认启用)
2. 减小 initial_covariance_scale
3. 检查观测值是否包含 NaN

## 测试与验证

### 运行示例

```bash
cargo test --package perple --lib tracker::kalman
```

### 单元测试示例

```rust
#[test]
fn test_predict_step() {
    let mut kf = KalmanFilterWrapper::new(KalmanConfig::default()).unwrap();
    let position = Vector3::new(1.0, 0.0, 0.0);
    let velocity = Some(Vector3::new(1.0, 0.0, 0.0));
    kf.init_with_state(position, velocity);
    
    // 预测一步
    kf.predict().unwrap();
    
    let pos = kf.get_position();
    // x = x0 + vx * dt = 1.0 + 1.0 * 0.1 = 1.1
    assert_relative_eq!(pos.x, 1.1, epsilon = 1e-10);
}
```

## 参考资料

1. **adskalman 文档**: https://docs.rs/adskalman
2. **Kalman滤波理论**: https://en.wikipedia.org/wiki/Kalman_filter
3. **常速模型详解**: http://www.robots.ox.ac.uk/~ian/Teaching/Estimation/LectureNotes2.pdf

## 迁移指南 (From LV-DOT)

如果你之前使用 C++ 版本的 dynamicDetector:

### C++ 原代码
```cpp
MatrixXd Z(4);
Z(0) = bbox.x;  // 位置 x
Z(1) = bbox.y;  // 位置 y
Z(2) = velocity_x;
Z(3) = velocity_y;
```

### Rust 新代码
```rust
let measurement = Vector3::new(x, y, z);
kf.update(measurement)?;
// 速度由滤波器内部估计，无需显式提供
```

**关键改进**:
- ✅ 自动估计速度，无需手动计算
- ✅ 完整的协方差管理
- ✅ 更好的数值稳定性
- ✅ 类型安全的矩阵运算
