# 策略级联 (Strategy Cascade)

## 核心思想

流水线各阶段通过 trait 解耦。上游策略由配置驱动，下游对上游实现无感知。Benchmark 模式下固定上游，遍历下游候选策略。

## 架构

```
DataLoader
    │
    ▼
GroundStrategy          ← 配置驱动 (create_ground_strategy)
    │ ground_mask, non_ground_points
    ▼
ClusteringStrategy      ← 配置驱动 (create_strategy)
    │ clusters, noise
    ▼
Tracker
    │ targets
    ▼
Output
```

每层只依赖上层的 trait 接口，不关心具体实现。切换算法只需改配置或替换 strategy 实例。

## Bench Framework

`src/bench/` 提供策略测试框架，消除 benchmark example 中的重复代码。

### 模块结构

```
src/bench/
├── bench.rs        — 模块导出
├── recorder.rs     — BenchRecorder（数据输出模块）
├── strategy.rs     — BenchStrategy trait + Preprocessor trait + Preprocessed 枚举
└── harness.rs      — BenchHarness（测试执行器）
```

### BenchStrategy trait

```rust
pub trait BenchStrategy {
    fn name(&self) -> &str;
    fn run(&mut self, frame: &FrameData) -> Duration;
    fn write_frame(&mut self, recorder: &mut BenchRecorder, frame: &FrameData);
    fn summarize(&self);
}
```

- `run` — 执行候选策略，返回耗时
- `write_frame` — 将检测结果写入 recorder（策略自行决定可视化方式）
- `summarize` — 输出汇总统计表

### Preprocessor trait（级联预处理）

```rust
pub trait Preprocessor {
    fn name(&self) -> &str;
    fn preprocess(&mut self, cloud: &[[f32; 3]]) -> Preprocessed;
}
```

预处理步骤，每帧执行一次，结果共享给所有候选策略。级联规则：
- ground_bench → `PassthroughPreprocessor`（无前序处理）
- cluster_bench → `GroundPreprocessor`（地面提取作为前序）
- 未来 tracker_bench → `ClusterPreprocessor`（地面+聚类作为前序）

`Preprocessed` 枚举承载每层产出，新增下游策略时扩展枚举即可。

### BenchRecorder（数据输出模块）

封装 `RdraWriter`，提供点云和检测框的写入辅助方法：

```rust
recorder.begin_frame(frame_idx);           // destroy_all + 设置 base_id
recorder.write_point_cloud(points, mat, n); // 写入点云（自动下采样）
recorder.write_boxes(&boxes, mat);          // 写入检测框（带 tag）
recorder.end_frame();                       // 结束帧
recorder.save("output.rdra");              // 保存文件
```

### BenchHarness（测试执行器）

自动完成：DataLoader 加载 → 逐帧迭代 → 预处理（每帧一次） → 策略串行执行 → 保存 .rdra

```rust
// ground_bench：无前序处理
let mut preprocessor = PassthroughPreprocessor;
BenchHarness::new("./data/test", 5, "output/ground_bench")
    .run(&mut preprocessor, &mut strategies)
    .await?;

// cluster_bench：地面提取作为前序
let mut preprocessor = GroundPreprocessor::default();
BenchHarness::new("./data/test", 64, "output/cluster_bench")
    .run(&mut preprocessor, &mut strategies)
    .await?;
```

### Example 用法

```rust
struct GroundBench { label: String, strategy: Box<dyn GroundPickStrategy>, /* stats */ }

impl BenchStrategy for GroundBench {
    fn name(&self) -> &str { &self.label }
    fn run(&mut self, frame: &FrameData) -> Duration {
        let mut cloud = frame.cloud.to_vec();
        let start = Instant::now();
        self.strategy.pick(&mut cloud);
        start.elapsed()
    }
    fn write_frame(&mut self, rec: &mut BenchRecorder, frame: &FrameData) { /* ... */ }
    fn summarize(&self) { /* 打印统计表 */ }
}

let mut strategies: Vec<Box<dyn BenchStrategy>> = vec![
    Box::new(GroundBench::new("histogram", Box::new(HistogramExpand::new()))),
    Box::new(GroundBench::new("ransac",    Box::new(RansacGround::new()))),
];
let mut preprocessor = PassthroughPreprocessor;
BenchHarness::new("./data/test", 64, "output/ground_bench")
    .run(&mut preprocessor, &mut strategies).await?;
```

## 生产环境：配置驱动

```rust
let mut ground = create_ground_strategy();
let result = ground.extract(&mut cloud);

let mut cluster = create_strategy();
let (processed, objects) = cluster.run(&non_ground);
```

## 关键设计

1. **trait 解耦** — `GroundStrategy`、`ClusteringStrategy`、`BenchStrategy` 各自定义接口
2. **工厂函数** — `create_ground_strategy()` / `create_strategy()` 封装配置读取
3. **with_params 构造** — 每个策略提供 `with_params()` 用于 benchmark 显式传参
4. **数据输出模块** — `BenchRecorder` 统一点云 + 检测框写入逻辑
5. **测试执行器** — `BenchHarness` 消除 DataLoader + swapl + 帧循环的重复代码

## 适用场景

- 多算法对比测试（ground_bench, cluster_bench）
- 可配置流水线（生产环境只改 config 切换算法）
- 新算法接入（实现 trait 即可，不改动下游）
