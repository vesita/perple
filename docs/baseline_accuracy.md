# 基线精度评估

## 评估条件

- 数据集：408 帧标注数据，1224 个 GT（Pedestrian）
- 匹配方式：中心距离 ≤ 0.5m + 匈牙利最优指派
- 测试次数：**2 次取平均**（因 YOLO ONNX 推理非确定性）

## 聚类策略

当前默认策略为 **lvdot_qt**（四叉树叶节点过滤 + DBSCAN），替代了原来的 dbscan_qt。主要指标对比（408 帧, 中心距 0.5m）：

| 指标 | dbscan_qt（旧） | lvdot_qt（新） | 变化 |
|------|:---:|:---:|:---:|
| Person Precision | 61.2% | **78.4%** | **+17.2pp** |
| Person Recall | 57.5% | **59.1%** | +1.6pp |
| Person F1 | 0.593 | **0.674** | **+0.081** |
| FP (Person) | 447 | **199** | **-55%** |

## 卡尔曼滤波器配置（当前优化后）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `kf_avg_frames` | **8** (原 5) | 速度观测窗口，更大窗口更平滑 |
| `kf_process_noise_pos` | 0.1 | 位置过程噪声 |
| `kf_process_noise_vel` | **0.05** (原 0.02) | 速度过程噪声，允许更大速度变化 |
| `kf_process_noise_acc` | **1.0** (原 0.5) | 加速度过程噪声 |
| `kf_process_noise_size` | 0.01 | 尺寸过程噪声 |
| `kf_measurement_noise_pos` | **0.3** (原 0.2) | 位置测量噪声（降低对质心信任） |
| `kf_measurement_noise_vel` | **0.8** (原 0.1) | 速度测量噪声（速度由位置差推导） |
| `kf_measurement_noise_acc` | **2.0** (原 0.5) | 加速度测量噪声（极不可靠） |
| `kf_measurement_noise_size` | **0.2** (原 0.5) | 尺寸测量噪声（框尺寸较可靠） |
| `kf_initial_covariance_scale` | **1.0** (原 0.5) | 初始协方差，更大加快收敛 |
| `kf_gate_threshold` | **3.5** (新增) | 新息门控，马氏距离超限降级为位置修正 |
| `geo_pass_threshold` | **6** | 几何后端连续通过帧数（复合策略） |
| `geo_fail_threshold` | **5** | 几何后端连续失败回退帧数 |
| `geo_speed_threshold` | **0.6** m/s | 速度激活阈值，与几何判断 OR |

## 几何后端复合策略

当前策略为两条路径 OR：
1. **几何累计路径**：连续 6 帧几何验证通过 → person
2. **速度激活路径**：平滑速度 > 0.6 m/s → 即时标记 person（无需累计）

回退：连续 5 帧几何失败 → obstacle。运动中自动清空失败计数。

## 严格评估（仅 person 类别参与）— lvdot_qt 策略

| 指标 | 当前值 | 目标 | 差距 |
|------|:---:|:---:|:---:|
| Precision | **78.4%** | - | - |
| Recall | **59.1%** | **60%** | **-0.9pp** |
| F1 | **0.674** | - | - |
| TP | 723 | - | - |
| FP | **199** | - | - |
| FN | 501 | - | - |

## 空间评估（全部检测参与匹配）— lvdot_qt 策略

| 指标 | 当前值 |
|------|:---:|
| Precision | 52.4% |
| Recall | **70.5%** |
| F1 | 0.601 |
| TP_spatial | 863 |
| FP_spatial | 785 |
| FN_spatial | 361 |

## 分类分析

空间匹配的 TP 中：

| 类别 | 数量 |
|------|:---:|
| 正确分类为 person | **718 (58.7%)** |
| 误分类为 obstacle | 145 (11.8%) |

## 测试 CLI

```bash
# 基线测试（默认已使用 lvdot_qt）
cargo run --release --example eval_ablation -- --center-dist 0.5 --frames 408

# 测试不同聚类策略
cargo run --release --example eval_ablation -- --cluster-toml 'strategy="dbscan_qt"' --center-dist 0.5 --frames 408

# 测试 lvdot_qt 参数
cargo run --release --example eval_ablation -- \
    --tracker-toml 'kf_measurement_noise_vel=1.0,kf_avg_frames=10' \
    --center-dist 0.5 --frames 408

# 调参测试（几何后端复合策略）
cargo run --release --example eval_ablation -- \
    --tracker-toml 'geo_pass_threshold=8,geo_fail_threshold=3,geo_speed_threshold=0.5' \
    --center-dist 0.5 --frames 408
```
