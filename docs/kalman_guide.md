# 卡尔曼滤波模块使用指南

## 概述

Perple 项目使用 `adskalman` 库实现了基于**常速模型**（Constant Velocity Model，CV 模型）的**卡尔曼滤波器**（Kalman Filter），用于目标跟踪中的状态估计和预测。

**核心功能**：

- ✅ 6 维状态向量（位置 + 速度）估计
- ✅ 常速运动模型预测
- ✅ 位置观测更新
- ✅ 协方差管理和不确定性评估
- ✅ 数值稳定性优化（Joseph 形式更新）
- ✅ 新息分析和异常检测

## 架构设计

### 核心组件

**源代码位置**: [`src/tracker/kalman.rs`](../src/tracker/kalman.rs)

```
tracker::kalman 模块
├── ConstantVelocityModel      # 常速运动模型 (实现 TransitionModelLinearNoControl trait)
├── PositionObservationModel   # 位置观测模型 (实现 ObservationModel trait)
└── KalmanFilterWrapper        # 简化的封装接口
```

### 状态向量定义

**6 维状态向量**: `[x, y, z, vx, vy, vz]ᵀ`

- 前 3 个分量：位置 (Position) - x, y, z（米）
- 后 3 个分量：速度 (Velocity) - vx, vy, vz（米/秒）

**3 维观测向量**: `[x, y, z]ᵀ`

- 只观测位置信息
- 速度由滤波器内部估计

**状态空间维度**:

- 状态维度：6
- 观测维度：3
- 控制维度：0（无外部控制输入）

## 数学模型

### 状态转移方程 (常速模型)

常速模型假设目标在两个时刻之间以恒定速度运动：

```
x_new = x_old + vx * dt
y_new = y_old + vy * dt
z_new = z_old + vz * dt
vx_new = vx_old
vy_new = vy_old
vz_new = vz_old
```

对应的状态转移矩阵 F(State Transition Matrix):

```
[1 0 0 dt  0  0]
[0 1 0  0 dt  0]
[0 0 1  0  0 dt]
[0 0 0  1  0  0]
[0 0 0  0  1  0]
[0 0 0  0  0  1]
```

**过程噪声协方差 Q**:

- 建模不确定性（目标加速度、模型误差）
- 通过 `process_noise_scale` 参数调整
- 典型值：0.001 ~ 0.1

### 观测方程

观测方程描述如何从状态空间映射到观测空间：

```
z_x = x
z_y = y
z_z = z
```

对应的观测矩阵 H(Observation Matrix):

```
[1 0 0 0 0 0]
[0 1 0 0 0 0]
[0 0 1 0 0 0]
```

**测量噪声协方差 R**:

- 建模传感器噪声
- 通过 `measurement_noise_scale` 参数调整
- 典型值：0.01 ~ 1.0

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
    // 预测下一步（基于常速模型）
    kf.predict()?;
    
    // 获取新的观测值 (从传感器)
    let measurement = get_sensor_measurement();
    
    // 使用观测值更新（如果测量有效）
    if measurement.is_finite() {
        kf.update(measurement)?;
    }
    // 如果测量无效，滤波器会自动跳过更新，只进行预测
    
    // 获取估计结果
    let estimated_pos = kf.get_position();
    let estimated_vel = kf.get_velocity();
}
```

### 2. 自定义配置

```rust
let config = KalmanConfig {
    dt: 0.05,  // 时间步长 50ms（对应 20Hz 采样频率）
    process_noise_scale: 0.001,      // 降低过程噪声 (更相信模型)
    measurement_noise_scale: 0.01,   // 降低测量噪声 (更相信观测)
    initial_covariance_scale: 100.0, // 初始高不确定性
};

let mut kf = KalmanFilterWrapper::new(config)?;
```

**配置参数说明**:

- `dt`: 时间步长（秒），必须与传感器采样频率匹配
- `process_noise_scale`: 过程噪声缩放因子，控制模型置信度
- `measurement_noise_scale`: 测量噪声缩放因子，控制观测置信度
- `initial_covariance_scale`: 初始协方差缩放因子，影响收敛速度

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

- **推荐值**: 与传感器采样频率匹配 (如 100Hz → dt=0.01, 20Hz → dt=0.05)
- **影响**:
  - dt 太小：滤波器响应慢，可能滞后于真实运动
  - dt 太大：预测精度下降，数值稳定性降低
- **调优方法**:
  - 测量传感器的实际采样间隔
  - 如果采样不均匀，考虑动态调整 dt

### 过程噪声 `process_noise_scale`

- **推荐范围**: 0.001 ~ 0.1
- **物理意义**: 建模目标加速度的不确定性
- **调优原则**:
  - 目标运动平稳（如匀速直线运动）→ 减小 (0.001~0.01)
  - 目标机动频繁（如转弯、加减速）→ 增大 (0.01~0.1)
  - 较大值：滤波器更灵活，但可能过度响应噪声
  - 较小值：滤波器更平滑，但可能滞后于真实运动
  
### 测量噪声 `measurement_noise_scale`

- **推荐范围**: 0.01 ~ 1.0
- **物理意义**: 建模传感器的测量误差
- **调优原则**:
  - 传感器精度高（如激光雷达）→ 减小
  - 传感器噪声大（如低成本超声波）→ 增大
  - 较大值：更相信模型（滤波效果好，但响应慢）
  - 较小值：更相信观测（响应快，但可能抖动）

### 初始协方差 `initial_covariance_scale`

- **推荐值**: 10.0 ~ 100.0
- **影响**: 只影响初始阶段，通常设为较大值表示不确定
- **调优原则**:
  - 较大值：初始收敛快，但可能超调
  - 较小值：初始收敛慢，但更稳定

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

当传感器无法提供有效测量时（如目标被遮挡），滤波器可以仅进行预测：

```rust
// 如果观测值无效 (NaN)，滤波器会自动跳过更新，只进行预测
let invalid_measurement = Vector3::new(f64::NAN, f64::NAN, f64::NAN);
kf.update(invalid_measurement)?;  // 等同于只执行 predict

// 或者显式检查
if measurement.is_finite() {
    kf.update(measurement)?;
} else {
    kf.predict()?;  // 仅预测
}
```

**注意**: 长时间丢失观测会导致不确定性（协方差）不断增大。

### 动态调整时间步长

当传感器采样频率不固定时，可以动态调整 dt：

```rust
let new_dt = calculate_dt_from_timestamps();
kf.set_dt(new_dt);
```

**注意**: dt 应该在每次预测前根据实际时间间隔更新。

### 重置滤波器

```rust
kf.reset();  // 回到初始零状态
```

### 不确定性评估

滤波器提供状态估计的不确定性（标准差）：

```rust
// 获取位置和速度的不确定性（标准差）
let pos_uncertainty = kf.get_position_uncertainty();
let vel_uncertainty = kf.get_velocity_uncertainty();

println!(
    "位置不确定性：({:.4}, {:.4}, {:.4})",
    pos_uncertainty.x, pos_uncertainty.y, pos_uncertainty.z
);
println!(
    "速度不确定性：({:.4}, {:.4}, {:.4})",
    vel_uncertainty.x, vel_uncertainty.y, vel_uncertainty.z
);
```

**应用**:

- 门控检测（Gating）：过滤掉不确定性过大的测量
- 数据关联：使用不确定性加权距离
- 轨迹质量评估：判断跟踪是否可靠

### 新息分析

新息 (Innovation) 是观测值与预测值的差，用于检测异常测量：

```rust
// 计算新息（测量残差）
let innovation = kf.get_innovation(measurement);

// 计算归一化新息平方 (NIS, Normalized Innovation Squared) 用于卡方检验
let nis = kf.normalized_innovation_squared(measurement);

// 卡方检验（自由度=3，95% 置信度）
if nis > 7.81 {
    println!("检测到异常测量！拒绝更新。");
    kf.predict()?;  // 仅预测，不更新
} else {
    kf.update(measurement)?;  // 正常更新
}
```

**应用**:

- 异常检测：识别传感器故障或野值
- 自适应滤波：根据 NIS 调整噪声参数
- 数据关联：多目标跟踪中的门控判断

## 性能指标

### 延迟

- 单次 `predict()` + `update()`: < 1μs（微秒级）
- 适用于实时系统 (>100Hz 采样频率)
- 内存占用：单个滤波器约 200 bytes

### 计算复杂度

- 预测步骤：O(n²)，n 为状态维度（6）
- 更新步骤：O(m³)，m 为观测维度（3）
- 总体复杂度：常数级（固定维度）

### 数值稳定性

- 使用 Joseph 形式协方差更新（默认）
- 避免协方差矩阵失去正定性
- 适用于长时间运行和病态条件

## 故障排查

### 问题 1: 滤波器发散

**症状**: 估计误差越来越大，协方差趋于无穷

**解决方案**:

1. 检查 dt 是否设置正确（与采样频率匹配）
2. 增大 process_noise_scale（增加模型灵活性）
3. 检查观测值是否有异常值（使用 NIS 检验）
4. 验证状态转移矩阵和观测矩阵是否正确

### 问题 2: 响应过慢

**症状**: 估计滞后于真实运动，跟踪不及时

**解决方案**:

1. 增大 process_noise_scale（更相信观测）
2. 减小 measurement_noise_scale（降低对模型的信任）
3. 检查 dt 是否过大（预测步长太大）
4. 考虑使用更复杂的运动模型（如匀加速模型）

### 问题 3: 数值不稳定

**症状**: 出现 NaN 或 Inf，协方差矩阵非正定

**解决方案**:

1. 使用 Joseph 形式协方差更新（已默认启用）
2. 减小 initial_covariance_scale（避免初始数值过大）
3. 检查观测值是否包含 NaN 或 Inf
4. 添加数值保护（如协方差平方根滤波）

## 测试与验证

### 运行示例

```bash
# 运行卡尔曼滤波模块的单元测试
cargo test --package perple --lib tracker::kalman

# 运行集成测试
cargo test --test integration kalman

# 运行所有测试（包含文档测试）
cargo test --all
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

## 迁移指南 (From LV-DOT)

如果你之前使用 C++ 版本的 dynamicDetector:

### C++ 原代码

```
MatrixXd Z(4);
Z(0) = bbox.x;  // 位置 x
Z(1) = bbox.y;  // 位置 y
Z(2) = velocity_x;
Z(3) = velocity_y;
```

### Rust 新代码

```
let measurement = Vector3::new(x, y, z);
kf.update(measurement)?;
// 速度由滤波器内部估计，无需显式提供
```

**关键改进**:

- ✅ 自动估计速度，无需手动计算
- ✅ 完整的协方差管理
- ✅ 更好的数值稳定性
- ✅ 类型安全的矩阵运算
