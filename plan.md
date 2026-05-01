# Perple 开发计划（更新于 2026-05-01）

## 已完成

### Phase 2: YOLO 辅助簇分裂/合并（Fuse 模块）
- `src/fuse.rs` — 2D→3D 语义融合：3D 簇投影到 2D 与 YOLO 框做 IoU 匹配，多簇匹配同框时合并，未匹配簇保留
- `src/perple.rs` — fuse_loop 在 tracker_loop 之前启动

### Phase 3: 跟踪器点云投票动态分类
- `src/tracker/core.rs` — `TrackedObject` 增加 `point_cloud_history: VecDeque<Vec<[f32;3]>>`
- 当前帧 vs skip_frames 帧前点云 → 最近邻找点对 → 方向与 KF 速度点乘投票
- `analyze_point_cloud_voting_direct()` 直接引用 `tracked_objects` 无拷贝

### Phase 4: 箱体尺寸锁定 fix_size
- `src/tracker/core.rs` — `TrackedObject.fixed_box` 冻结尺寸，`apply_fix_size()` 在跟踪稳定后锁定

### Phase 6A: ground 强制 Static
- `src/tracker/core.rs` — 输出阶段 ground 覆写为 static

### Phase 6B: KF 噪声参数
- `src/tracker/kalman.rs` — `adjust_noise_for_distance()` 远距离自动增大测量噪声

### Phase 7: 管线速度优化（80ms → 45ms/帧，22 FPS）
- **两级流水**：`examples/pipeline_test.rs` 中 frame i+1 的 lidar|cam 与 frame i 的 fuse+tracker 并行
- **匈牙利预分配**：`src/tracker/hungarian.rs` 复用 `sq_buf` 避免每帧堆分配
- **VecDeque 替换 Vec**：`src/tracker/core.rs` O(1) pop_front 替代 O(k) remove(0)
- **点云下采样**：每个目标最多 200 点，O(N²) 投票从 755ms 降到 9ms
- **矩阵求逆缓存**：`extract_points_in_box` 预计算一次逆矩阵，pcupd 从 75ms 降到 30ms

### Phase 8: 分类状态机
- `src/tracker/core.rs` — `apply_state_machine()` 双向状态机：
  ```
  Static ↔ Floating（滞后：上浮无门槛，沉淀需 N 帧）
  Floating ──→ Moving（点云投票 + 速度 + 连续帧）
  Moving ←──→ Movable（同层往返，速度决定）
  confirmed_moving=true → 永不回到 Static/Floating
  ```

### Phase 9D: KF 距离自适应噪声
- `src/tracker/kalman.rs` — `adjust_noise_for_distance()` 按目标距离动态调整测量噪声

---

## 待办

### A: 速度检测稳定性
**现状**：associate 错误导致速度尖峰（id=201 从 4.85→13.49 m/s）
**已做**：Kalman `clamp_velocity(10.0)` 限幅
**待优化**：
- associate 中加入速度一致性门控（不仅用位置马氏距离，还需检查速度预测 vs 观测）
- 或降低 `measurement_noise_vel` 信任度让 KF 更平滑

### B: 行人的 moving 确认
**现状**：YOLO 检测到 person 时 class_type 设为 "person"，但状态机仍按速度判断
**已做**：`confirmed_moving=true` 保证不再退回 floating/static
**待优化**：输出端 person 直接标记为 "moving" 还是交由速度阈值决定 moving↔movable？需确认语义

### C: 管线计时修复
**现状**：两级流水下 `frame_start` 设在 join lidar|cam(i) 之后，测量的是 fuse+tracker(i) 时间而不是完整帧时间
**问题**：总耗时 / n_frames 度量的是吞吐量（throughput），不是每帧延迟（latency）。流水线稳态下：
- 帧延迟（latency）= lidar|cam + fuse+tracker ≈ 80ms
- 帧吞吐量（throughput）= max(lidar|cam, fuse+tracker) ≈ 45ms
两种度量都有意义，需在输出中明确区分

### D: 流数据竞态条件
**现状**：fuse 用 `peek_latest()` 读 `clr_objs` 流（避免偷走 classify 的数据），但 `peek_latest()` 返回最新写入，不是帧 i 对应的 YOLO 结果
**风险**：lidar|cam(i+1) 提前写入 clr_objs 后，fuse(i) 读到的是下一帧的 YOLO 数据
**待改**：需要帧对齐机制——或 fuse 用 `read()` + classify 不做 YOLO 处理

### E: 点云投票参数调优
**现状**：`skip_frames=5`，`vote_threshold=0.8`，`max_points=200`
**待验证**：
- 200 点下采样是否丢失关键运动特征
- `skip_frames=5` 在 25 FPS 下 = 200ms 间隔，是否太长
- `vote_threshold=0.8` 是否太严格导致漏检

### F: 长期跟踪稳定性
**现状**：目标消失后 `max_disappeared=8` 帧移除，重新出现时分配新 ID
**问题**：
- 遮挡后 ID 切换频繁
- 速度历史在消失期间未更新，恢复后 KF 预测位置偏移大
- associate 用位置马氏距离，消失再出现时容易匹配到错误目标

---

## 依赖关系

```
A (速度稳定性)       ← 独立
B (行人确认)         ← 依赖 A 的效果
C (管线计时修复)      ← 独立，可先做
D (流数据竞态)       ← 与 A/B 互相独立
E (投票参数调优)      ← 依赖 B 完成后
F (长期跟踪)          ← 依赖 A+D
```

## 测试验证

```bash
# 性能测试
cargo run --example pipeline_test -- --frames 200
# 预期：平均 ≤ 50ms/帧 (≥ 20 FPS)

# CSV 日志分析
cargo run --example pipeline_test -- --frames 685 --csv log/result.csv
# 检查 CSV 中 ground 目标的 is_dynamic 始终为 false
# 检查 person 目标的 classification 是否为 moving/movable
```
