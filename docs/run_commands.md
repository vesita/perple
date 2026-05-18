# 运行命令速查

## 构建

```bash
cargo build                        # Debug 构建
cargo build --release              # Release 构建
```

## 完整管线

| 命令 | 说明 |
|------|------|
| `cargo run --release` | 完整检测→融合→跟踪管线 + .rdra/JSONL 输出 |
| `cargo run --release -- --frames 100` | 仅处理前 100 帧 |
| `cargo run --release -- --skip 10` | 跳过前 10 帧 |
| `cargo run --release -- --output ./my_output` | 指定输出目录 |

**默认配置**：`peak_scan` + `bev_edlines` + `prune_qt`（见 `config/default.toml`）

> .rdra 文件：`output/pipeline_<timestamp>/` 下生成 `ground.db`、`wall.db`、`cluster.db`、`tracker.db`

## 精度评估

### 标注评估（eval_labeled）

```bash
cargo run --release --example eval_labeled                         # 默认全部帧
cargo run --release --example eval_labeled -- --frames 408         # 指定帧数
cargo run --release --example eval_labeled -- --center-dist 0.5    # 中心距离匹配
cargo run --release --example eval_labeled -- --iou 0.15           # IoU 匹配
```

### 消融评估（eval_ablation）

```bash
cargo run --release --example eval_ablation -- --cluster-toml 'strategy="prune_qt"' --center-dist 0.5 --frames 408
cargo run --release --example eval_ablation -- --cluster-toml 'strategy="dbscan_qt"' --center-dist 0.5 --frames 408
cargo run --release --example eval_ablation -- --cluster-toml 'strategy="cc"' --center-dist 0.5 --frames 408
```

### EDLines 对比评估

```bash
cargo run --release --example edlines_labeled_bench                # BevEdLines vs EdLinesRef
cargo run --release --example edlines_labeled_bench -- --frames 50
```

## 单模块 Benchmark

### 地面提取

```bash
cargo run --example ground_bench -- --mode=quick
```

### 墙体提取

```bash
cargo run --example wall_bench -- --mode=quick
cargo run --example wall_bench -- --mode=full
```

### 后聚类

```bash
cargo run --example cluster_bench -- --mode=quick
cargo run --example cluster_bench -- --strategy=prune_qt --denoise-radius=0.20 --denoise-min-pts=3
cargo run --example denoise_bench -- --mode=quick
```

## 管线对比

```bash
cargo run --example pipeline_evolution_bench                       # 管线演化对比（论文用）
cargo run --example pipeline_evolution_bench -- --frames 50
cargo run --example wall_pipeline_bench -- --mode=quick            # 墙体策略对聚类影响
cargo run --example wall_pipeline_bench -- --mode=full
```

## 客户端测试

```bash
cargo run --example redra_test --package redra_client
cargo run --example label_test --package redra_client
```

## 性能延迟测试

```bash
cargo run --release --example latency_bench
```

## ROS 模式

```bash
cargo run --features ros1
```

## Python 分析

```bash
# 轨迹可视化
.venv/Scripts/python.exe scripts/viz_trajectory.py

# PR 曲线
.venv/Scripts/python.exe scripts/viz_pr_curve.py

# 汇总图表
.venv/Scripts/python.exe scripts/viz_summary.py
.venv/Scripts/python.exe scripts/thesis_viz.py

# EDLines 对比可视化
.venv/Scripts/python.exe scripts/edlines_compare_viz.py
.venv/Scripts/python.exe scripts/edlines_labeled_viz.py
.venv/Scripts/python.exe scripts/edlines_speed_stability.py
.venv/Scripts/python.exe scripts/ablation_aggregate.py
.venv/Scripts/python.exe scripts/ablation_charts.py

# 分析管线（从已存数据重绘图）
.venv/Scripts/python.exe scripts/bench_pipeline.py --analysis-only
.venv/Scripts/python.exe scripts/run_wall_pipeline.py
.venv/Scripts/python.exe scripts/run_wall_pipeline.py --mode=full
```

## 配置覆盖

```bash
# 使用指定配置文件
PERPLE_CONFIG_PATH=config/edlines_ref.toml cargo run --release

# 覆盖墙体策略
PERPLE_CONFIG_PATH=config/edlines_ref.toml cargo run --release --example eval_ablation -- --frames 408
```
