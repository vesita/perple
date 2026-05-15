# 性能优化记录

## 基线 (优化前)

**帧耗时: ~356ms | ~2.8 FPS**

| 阶段 | 耗时 | 说明 |
|------|------|------|
| 图像预处理 (Camera) | ~259ms | resize(CatmullRom) + to_rgb8() + NCHW |
| YOLO 推理 | ~31ms | 串行于图像预处理之后 |
| Lidar 处理 | ~28ms | 地面提取 + 聚类 |
| Tracker | ~38ms | 关联 + KF 更新 + 分类 |
| **帧总耗时** | **~356ms** | 各阶段串行累加 |

## 优化记录

### Opt 1: DataLoader 内存预加载

- **改动**: DataLoader `load()` 一次性读入全部 685 帧到内存，帧循环 `load_next()` 仅做内存→流拷贝
- **效果**: 帧循环中零磁盘 I/O
- **风险**: 启动延迟 53s（685 帧加载），内存占用增加 ~1.5GB

### Opt 2: 图像插值 CatmullRom → Triangle

- **改动**: `src/color/image.rs` resize filter CatmullRom → Triangle
- **效果**: 图像预处理 ~259ms → ~182ms（-77ms）
- **说明**: CatmullRom(4x4 卷积核) 对 YOLO 输入过于奢侈，Triangle(bilinear, 2x2) 质量足够

### Opt 3: 去除 to_rgb8() 中间拷贝

- **改动**: `fill_input_image` 直接用 `DynamicImage::as_bytes()` 读取原始像素，跳过 `to_rgb8()` 转换
- **效果**: 图像预处理 ~182ms → ~176ms（-6ms）
- **说明**: 通过检测 bytes_per_pixel (3 或 4) 兼容 RGB/RGBA 格式

### Opt 4: Triangle → Nearest + 一步到位 NCHW 采样

- **改动**: 移除 `image::imageops::resize` 调用，直接计算 Nearest 采样映射填充 NCHW 张量
- **效果**: 图像预处理 ~176ms → ~1ms（-175ms），帧总耗时 356ms → 136ms
- **说明**: 640×480 → 640×640 Nearest 映射 = 逐像素坐标映射，无需中间 buffer 分配

### Opt 5: 帧循环计时清理

- **改动**: 将预加载步骤从帧循环移至初始化阶段
- **效果**: 帧耗时 136ms → 106ms（纯净帧时间，不包含预加载）

### Opt 6: tokio::join! → tokio::spawn 真并行

**发现背景**：
`tokio::join!(lidar.act(), camera.act())` 在同一个 tokio 任务内协作轮询两个 future。
Camera 做 30ms 同步 ONNX 推理时，lidar 在该任务内得不到执行——结果是**顺序执行**而非并行。

**改动**:
- `examples/pipeline_test.rs`: `tokio::join!` → `tokio::spawn` + handle join
- `src/color/core.rs`: 推理时不持有输出流锁（`local_bounds` 缓冲区 + `mem::swap`），
  防止 lidar 的 YOLO refinement 被 camera 的输出锁阻塞

**效果**: 帧总耗时 106ms → 80ms（-25%），FPS 9.4 → 12.5

**原理**:
- `tokio::join!` 是单任务协作调度，CPU 密集型任务会阻塞同伴
- `tokio::spawn` 创建独立 tokio 任务，由工作线程池调度到不同核心上真并行
- ONNX Runtime 4 线程 + lidar 独立线程 = 12 核机器上充分并行

### Opt 7: 推理锁释放

**改动**: Camera 推理时不再持有 `clr_objs` 输出锁，使用本地 `local_bounds: Vec<ClrBud>` 缓冲推理结果，推理完成后简短获取锁写入

**效果**: 消除了 camera 与 lidar YOLO refinement 之间的锁竞争（配合 Opt 6 的 `tokio::spawn` 后效果更明显）

---

## 最终指标

**帧耗时: ~80ms | ~12.5 FPS** (Improvement: 78% from baseline)

### 当前帧时间分布 (2026-05-01)

| 阶段 | 耗时 | 说明 |
|------|------|------|
| Lidar 处理 | ~29ms | 与 Camera 并行 |
| YOLO 推理 | ~30ms | 与 Lidar 并行 (独立 tokio 任务) |
| **并行段墙钟** | **~40ms** | 含 spawn 调度开销 |
| Fuse | ~0ms | 串行 |
| Tracker | ~39ms | 串行，**当前瓶颈** |
| 帧统计 I/O | ~1ms | 串行 |
| **帧总耗时** | **~80ms** | |

### 时间线（Opt 6 后）

```text
Frame N:
  ┌─ tokio::spawn: lidar(~29ms) ───────────────┐
  ├─ tokio::spawn: camera(~30ms) ──────────────┼── wait max(~30ms)
  │                                              │
  └────────── 并行段墙钟 ~40ms ──────────────────┘
                          ↓
                    fuse(~0ms) → tracker(~39ms) → stats(~1ms)
```

### 当前瓶颈：Tracker ~39ms

现阶段的三个 O(n²)/O(n³) 串行算法：

| 子阶段 | 复杂度 | 说明 |
|--------|--------|------|
| Hungarian 关联 | O(K³) | 每帧重新分配 `vec![vec![f64; M]; N]` 矩阵 |
| 速度 DBSCAN | O(N²) | 无空间索引 |
| 点云投票 | O(T·P²) | 线性扫描 NN，每帧逐目标 |

### 待优化方向

1. **匈牙利矩阵预分配** — 避免每帧堆分配，预计省 ~5ms
2. **Vec::remove(0) → VecDeque** — 点云历史 O(k) → O(1)，预计省 ~3ms
3. **Tracker 内 DBSCAN + 投票用 spawn 并行** — 省 ~10ms（需拆 &mut 冲突）
4. **YOLO ONNX int8 量化** — 预计省 ~15ms（需验证精度损失）
5. **两级流水** — 下一帧 lidar+camera 与当前帧 fuse+tracker 重叠，可隐藏 ~40ms，但需要流架构改动

## 关键文件

| 文件 | 改动 |
|------|------|
| `src/color/image.rs` | resize filter + NCHW 一步到位采样 |
| `src/color/core.rs` | 推理锁释放 + local_bounds 缓冲区 |
| `src/cloud/core.rs` | lidar 计时 |
| `src/optional/data_loader.rs` | 内存预加载 + on-demand streaming |
| `src/tracker/kalman.rs` | measurement_noise_vel 0.8→0.1 |
| `src/tracker/core.rs` | person→movable 覆写 + cluster_N→obstacle |
| `examples/pipeline_test.rs` | 预加载提前 + spawn 真并行 + 阶段计时 |
