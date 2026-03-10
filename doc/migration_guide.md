# dynamicDetector 迁移指南(From LV-DOT)

## 概述

本指南旨在将 C++ ROS 版本的 `dynamicDetector` 迁移到 Rust 项目 `perple` 中。迁移将聚焦于**点云检测**和**目标跟踪**部分，深度图像（depth image）相关的 UV 检测功能将被移除。

## 架构对比

### 原 C++ 架构 (dynamicDetector.cpp)

```module
dynamicDetector (ROS Node)
├── 数据输入
│   ├── 激光雷达点云 (lidarCloudSub_)
│   ├── 深度图像 (depthSub_) - 已移除
│   └── 位姿/里程计 (poseSub_/odomSub_)
├── 检测模块
│   ├── DBSCAN 聚类 (dbscanDetect)
│   └── UV 检测器 (uvDetect) - 已移除
├── 跟踪模块
│   ├── 卡尔曼滤波 (kalmanFilterAndUpdateHist)
│   ├── 数据关联 (boxAssociation)
│   └── 历史管理 (boxHist_, pcHist_)
└── 输出
    ├── 动态障碍物边界框 (dynamicBBoxes_)
    ├── 轨迹发布 (publishHistoryTraj)
    └── 速度可视化 (publishVelVis)
```

### 目标 Rust 架构 (perple)

```module
perple 系统
├── cloud 模块 (点云处理)
│   ├── core.rs (点云数据流)
│   └── classify/claster.rs (DBSCAN 聚类) ✓ 已实现
├── tracker 模块 (目标跟踪)
│   ├── core.rs (跟踪器主体) 🔄 迁移中
│   ├── kalman.rs (卡尔曼滤波) ✓ 已实现
│   └── output.rs (输出结构)
└── utils 模块
    └── boxes.rs (Box3D 结构) ✓ 已实现
```

## 核心组件映射

| C++ 组件 | Rust 对应模块 | 状态 | 备注 |
| --------- | ------------- | ------ | ------ |
| `lidarDetector_` | `cloud::classify::Claster` | ✅ 完成 | [`claster.rs`](src/cloud/classify/claster.rs) |
| `UVdetector_` | - | ❌ 移除 | 深度图像相关功能 |
| `Kalman Filter` | `tracker::kalman::KalmanFilter` | ✅ 完成 | [`kalman.rs`](src/tracker/kalman.rs) |
| `boxAssociation` | `tracker::core::Tracker` | 🔄 设计中 | 需实现 IOU 匹配 |
| `boxHist_` | `tracker::core::TrackedObject` | 🔄 设计中 | 历史记录管理 |
| `dynamicBBoxes_` | `tracker::output::Target` | ✅ 完成 | [`output.rs`](src/tracker/output.rs) |

## 迁移步骤

### 第一步：点云检测 (已完成)

✅ **DBSCAN 聚类** - `cloud::classify::Claster`

C++ 原代码中的 `dbscanDetect()` 功能已迁移至 Rust:

```rust
// src/cloud/classify/claster.rs
pub fn claster(&mut self, lifra: &[[f32; 3]]) {
    // 1. 体素下采样
    // 2. 构建四叉树
    // 3. DBSCAN 聚类
    // 4. 转换为 Box3D
}
```

**关键改进**:

- 使用四叉树加速邻域查询 (原 C++ 使用线性搜索)
- 体素下采样预处理
- 更高效的 HashSet 用于簇扩展

### 第二步：跟踪器设计 (当前任务)

🔄 **Tracker 核心逻辑** - `tracker::core::Tracker`

需要实现的功能模块：

#### 1. 数据关联 (Box Association)

```rust
/// 基于 IOU 的数据关联
fn box_association(
    &mut self,
    current_detections: &[Box3D],
    previous_tracks: &[TrackedObject],
) -> Vec<(usize, Option<usize>)> {
    // TODO: 实现 IOU 计算
    // TODO: 匈牙利算法或贪婪匹配
    // 返回：(当前检测 ID, 匹配的上一帧跟踪 ID)
}
```

**参考 C++ 代码**:

```cpp
void dynamicDetector::boxAssociation(std::vector<int>& bestMatch){
    // 计算 IOU 矩阵
    // 贪婪匹配：为每个当前检测框找到最佳匹配的历史框
}
```

#### 2. 卡尔曼滤波更新

```rust
/// 卡尔曼滤波预测与更新
fn kalman_filter_update(
    &mut self,
    matched_pairs: &[(usize, usize)],
    current_detections: &[Box3D],
) -> Result<(), TrackerError> {
    // TODO: 对每个匹配的对象：
    // 1. 提取位置 (x, y) 和速度 (vx, vy) 作为观测
    // 2. 调用 KalmanFilter::update()
    // 3. 更新 TrackedObject 的状态
}
```

**观测模型** (参考 C++):

```cpp
void dynamicDetector::getKalmanObservationVel(
   const onboardDetector::box3D& currDetectedBBox, 
    int bestMatchIdx, 
    MatrixXd& Z
){
    Z(0) = currDetectedBBox.x;  // 位置 x
    Z(1) = currDetectedBBox.y;  // 位置 y
    // 使用前 k 帧计算速度
    Z(2) = (currDetectedBBox.x - prevMatchBBox.x) / (dt_*k);  // 速度 vx
    Z(3) = (currDetectedBBox.y - prevMatchBBox.y) / (dt_*k);  // 速度 vy
}
```

#### 3. 历史管理

```rust
/// 管理跟踪历史和对象生命周期
struct TrackedObject {
    id: usize,
    kalman_filter: KalmanFilter,
    class_type: String,
    history: Vec<Box3D>,      // 边界框历史
    pc_history: Vec<Vec<[f32; 3]>>, // 点云历史 (可选)
    disappeared_count: u32,   // 丢失计数
    last_seen: SystemTime,
}

impl TrackedObject {
    /// 更新对象状态
    fn update(&mut self, detection: Box3D) {
        self.history.push(detection);
        self.disappeared_count = 0;
    }
    
    /// 预测下一帧位置（丢失时使用）
    fn predict(&mut self, dt: f32) -> Result<Box3D, TrackerError> {
        // 使用卡尔曼滤波预测
    }
}
```

#### 4. 坐标变换

C++ 中的坐标变换需要迁移:

```cpp
// C++ 原代码
void dynamicDetector::transformUVBBoxes(std::vector<onboardDetector::box3D>& uvBBoxes){
    // 从相机坐标系变换到世界坐标系
    // 使用 pose 矩阵：positionDepth_, orientationDepth_
}
```

Rust 实现建议:

```rust
use nalgebra::{Isometry3, Point3};

/// 坐标变换工具
pub mod transform {
    use super::*;
    
    /// 将局部坐标系下的边界框变换到世界坐标系
    pub fn transform_box_to_world(
        local_box: &Box3D,
        sensor_pose: &Isometry3<f32>,  // 传感器在世界系中的位姿
    ) -> Box3D {
        // 1. 变换中心点
        let center = local_box.center();
        let world_center = sensor_pose * center;
        
        // 2. 变换方向 (仅旋转部分)
        let world_rotation = sensor_pose.rotation.to_rotation_matrix();
        
        // 3. 构建新的 Box3D
        Box3D::from_center_and_rotation(
            world_center,
            world_rotation,
            local_box.length,
            local_box.width,
            local_box.height,
        )
    }
}
```

### 第三步：集成到主流程

#### 数据流设计

```rust
// main.rs 或 perple.rs 中的主循环
async fn perception_loop() {
    let mut tracker= Tracker::new();
    
    loop {
        // 1. 获取点云数据
        let point_cloud = get_lidar_cloud().await;
        
        // 2. 聚类检测
        let mut claster = Claster::new();
        claster.claster(&point_cloud);
        let detections = claster.to_cldbuds();
        
        // 3. 坐标变换 (从雷达到世界系)
        let lidar_pose = get_lidar_pose();  // 从机器人位姿 + 外参计算
        let world_detections: Vec<Box3D> = detections
            .iter()
            .map(|cld_bud| transform_box_to_world(&cld_bud.the_box, &lidar_pose))
            .collect();
        
        // 4. 跟踪关联
        tracker.associate_and_track(world_detections).await?;
        
        // 5. 发布结果
        let targets = tracker.get_tracked_targets();
        publish_targets(targets);
        
        // 6. 可视化 (可选)
        publish_trajectories(tracker.get_histories());
    }
}
```

## 关键差异与优化

### 1. 异步 vs 同步

**C++ (ROS)**: 基于回调的异步模型

```cpp
// 多个定时器回调
this->detectionTimer_ = nh_.createTimer(ros::Duration(dt_), &dynamicDetector::detectionCB, this);
this->trackingTimer_ = nh_.createTimer(ros::Duration(dt_), &dynamicDetector::trackingCB, this);
```

**Rust (tokio)**: 基于 async/await 的异步模型

```rust
// 建议使用 tokio 定时器
use tokio::time::{interval, Duration};

let mut detection_interval = interval(Duration::from_millis(100));
loop {
    detection_interval.tick().await;
    // 执行检测
}
```

### 2. 内存安全

**C++**: 手动管理指针 (`boost::shared_ptr`)

```cpp
this->lidarDetector_.reset(new lidarDetector());
```

**Rust**: 所有权系统自动管理

```rust
let claster = Claster::new();  // 自动 Drop
```

### 3. 性能优化点

| 优化项 | C++ 实现 | Rust 改进 |
| ------- | --------- | ---------- |
| 邻域查询 | O(n²) 线性搜索 | 四叉树 O(n log n) |
| 聚类包含检查 | O(n) vector 遍历 | O(1) HashSet |
| 体素下采样 | 未实现 | HashMap 加速 |
| 并发安全 | mutex 锁 | tokio::sync::Mutex |
| 内存拷贝 | 频繁 clone() | 借用 + 切片引用 |

## 待实现功能清单

### Tracker 模块 (优先级高)

- [ ] **IOU 计算函数**

  ```rust
  fn calculate_iou(box1: &Box3D, box2: &Box3D, ignore_z: bool) -> f32
  ```
  
- [ ] **贪婪匹配算法**

  ```rust
  fn greedy_matching(
      iou_matrix: &[Vec<f32>], 
      threshold: f32
  ) -> Vec<(usize, Option<usize>)>
  ```

- [ ] **卡尔曼观测提取**

  ```rust
  fn extract_observation(
      current: &Box3D, 
      history: &[Box3D], 
      k: usize
  ) -> Vector4<f32>
  ```

- [ ] **轨迹管理**

  ```rust
  fn update_trajectory(&mut self, target_id: usize, new_box: Box3D)
  fn remove_lost_target(&mut self, target_id: usize)
  ```

### 可视化和输出 (优先级中)

- [ ] **轨迹发布** (对应 C++ `publishHistoryTraj`)
- [ ] **速度可视化** (对应 C++ `publishVelVis`)
- [ ] **点云发布** (对应 C++ `publishPoints`)

### 配置参数 (优先级低)

从 C++ 的 `initParam()` 迁移配置项到 `config/default.toml`:

```toml
# config/default.toml
[tracker]
kf_avg_frames = 3          # 速度估计平均帧数
max_disappeared = 5        # 最大丢失帧数
iou_threshold = 0.3        # IOU 匹配阈值
voxel_size = 0.1           # 体素大小
```

## 测试验证

### 单元测试

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_iou_calculation() {
        // 测试 IOU 计算正确性
    }
    
    #[test]
    fn test_kalman_prediction() {
        // 测试卡尔曼滤波预测
    }
}
```

### 集成测试

```rust
// examples/tracker_test.rs
use perple::tracker::Tracker;
use perple::cloud::Claster;

#[tokio::main]
async fn main() {
    // 加载测试点云数据
    // 运行完整检测跟踪流程
    // 验证输出合理性
}
```

## 常见问题

### Q1: 如何处理坐标系转换？

**A**: 使用 `nalgebra` 库的 `Isometry3` 统一表示位姿：

```rust
use nalgebra::Isometry3;

// 雷达坐标系到世界坐标系
let lidar_to_world = Isometry3::from_parts(
    Translation3::new(x, y, z),
    UnitQuaternion::from_euler_angles(rx, ry, rz)
);
```

### Q2: 卡尔曼滤波的状态向量如何设计？

**A**: 参考 C++ 实现，使用 4 维或 6 维状态：

```rust
// 4 维：位置 + 速度
state = [x, y, vx, vy]ᵀ

// 6 维：位置 + 速度 + 加速度
state = [x, y, vx, vy, ax, ay]ᵀ
```

### Q3: 如何处理遮挡和丢失？

**A**: 实现消失计数器机制：

```rust
if detection_matched {
   target.disappeared_count = 0;
} else{
   target.disappeared_count += 1;
    if target.disappeared_count > MAX_DISAPPEARED {
        remove_target(id);
    }
}
```

## 下一步行动

1. **立即任务**: 完善 `tracker::core::Tracker` 的关联逻辑
2. **短期任务**: 添加 IOU 计算和匈牙利算法
3. **中期任务**: 集成到主感知循环并测试
4. **长期任务**: 优化性能和添加可视化
