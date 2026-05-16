# Bench 级联流水线

## 核心理念

Bench 系统采用 **TOML 驱动 + 三级级联** 架构：

```
TOML 策略定义 → Rust 执行引擎 → Python 编排与分析
```

每层职责明确，通过文件和标准输出通信。

---

## 目录约定

```
output/bench/
  {task}/                    # 任务（ground / cluster / wall）
    {strategy}/              # 策略（peak_scan / ransac / …）
      info.json              # Rust 写入的运行结果（含所有参数组合）
      *.rdra                 # 可视化数据文件，每个参数组合一个
  analysis/
    quick/{task}/            # 快速测试图表（Python 生成）
    full/{task}/             # 全量测试图表（Python 生成）
    cross/                   # 快速 vs 全量交叉对比

config/bench/
  {task}/                    # TOML 策略定义
    {strategy}.toml          # 每个策略一个文件
```

### 文件管理规则

| 阶段 | 写入目录 | 可删除 | 可读取 |
|------|---------|--------|--------|
| Rust bench 执行 | `output/bench/{task}/` | 自己目录 | 无限制 |
| Python 快速分析 | `analysis/quick/` | 自己目录 | 所有 output/bench/ |
| Python 全量分析 | `analysis/full/` | 自己目录 | 所有 output/bench/ |
| Python 交叉分析 | `analysis/cross/` | 自己目录 | 所有 analysis/ |

TOML 策略文件由 Rust 读写 `[stats]` 段，Python 只读不改。

---

## TOML 策略格式

每个策略一个 TOML 文件，放置在 `config/bench/{task}/{strategy}.toml`。

```toml
name = "峰值扫描地面检测"        # 中文说明
type = "peak_scan"               # 内部类型标识，与 Rust 工厂匹配
preprocessor = "passthrough"     # 所需预处理器

[quick]
frames = 1                       # 快速测试帧数
params = [                       # 快速测试参数子集
  { threshold = 0.10, expand = 0.10 },
  { threshold = 0.10, expand = 0.15 },
]

[full]
frames = 50                      # 全量测试帧数
params = [                       # 全量测试完整参数网格
  { threshold = 0.03, expand = 0.05 },
  { threshold = 0.05, expand = 0.10 },
  # ...
]

[stats]                          # 持久化的速度统计（Rust 自动更新）
fastest_ms = 12.3
slowest_ms = 45.6
avg_ms = 28.9
median_ms = 27.1
last_run = "1746758400"          # Unix 时间戳
```

已有 17 个策略文件，分布在 `ground/`（5）、`cluster/`（5）、`wall/`（7）。

### 参数命名规范

- 下划线命名（`cell_size`、`min_pts`），与 Rust 侧和 `param_dirname()` 匹配
- 所有数值使用浮点数（即使逻辑上是整数），避免 TOML 类型歧义

---

## BenchMode 运行模式

### Single（传统模式）

```bash
cargo run --example ground_bench -- --frames 5
cargo run --example ground_bench -- --strategy=ransac --distance=0.2
```

- 单个策略 + 单组参数，从 CLI 读取
- 输出到 `output/ground_bench/`（遗留路径）
- 不读写 TOML 文件

### Quick（快速测试）

```bash
cargo run --example ground_bench -- --mode quick
```

- 从 `config/bench/ground/*.toml` 读取所有策略
- 使用 `[quick].params` 子集 + `[quick].frames` 帧数
- 所有策略全部测试（不跳过慢策略）
- 输出到 `output/bench/ground/{strategy}/`
- 更新 TOML `[stats]` 段

### Full（全量测试）

```bash
cargo run --example ground_bench -- --mode full
```

- 从 TOML 读取所有策略
- 使用 `[full].params` 完整网格 + `[full].frames` 帧数
- **跳过**上次 `[stats]` 中 `slowest_ms > 100ms` 且已测过的策略
- 输出到 `output/bench/ground/{strategy}/`
- 更新 TOML `[stats]` 段

---

## 速度筛选机制

全量测试自动跳过慢策略：

```rust
if stats.slowest_ms > 100.0 && stats.last_run.is_some() {
    println!("跳过 {} (上次最慢 {:.0}ms > 100ms)", name, slowest_ms);
    // 不加入到策略列表
}
```

跳过条件：
1. `slowest_ms > 100.0` — 最慢一帧超过 100ms
2. `last_run.is_some()` — 已经跑过至少一次

首次测试（`last_run = None`）或 `slowest_ms <= 100` 的策略不跳过。

---

## 级联流程

Python 脚本 `scripts/bench_pipeline.py` 编排三级流程：

```
第 1 步：快速测试
  ├── cargo run task --mode quick（对所有任务）
  ├── 收集 output/bench/{task}/*/info.json
  └── 生成 analysis/quick/{task}/ 图表

第 2 步：全量测试（除非 --quick-only）
  ├── cargo run task --mode full（对所有任务）
  ├── 收集 info.json
  └── 生成 analysis/full/{task}/ 图表

第 3 步：交叉分析
  ├── 对比 quick vs full 的 avg_ms
  ├── 计算差异百分比
  └── 生成 analysis/cross/{task}_comparison.png
```

---

## Rust 执行引擎

### 架构

```
BenchHarness::run()
  → 加载数据（DataLoader）
  → 逐帧迭代
    → 预处理（Preprocessor）
    → 对所有策略执行 strategy.run()
    → 对所有策略执行 strategy.write_frame()
  → 返回 (Vec<BenchStats>, Vec<BenchRecorder>)

run_toml_bench()
  → 加载 TOML 策略（load_task_strategies）
  → 过滤慢策略（Full 模式）
  → 调用 StrategyBuilder 构建策略
  → 调用 BenchHarness::run()
  → 保存 .rdra + info.json
  → 更新 TOML [stats]
```

### 关键接口

| 接口 | 位置 | 作用 |
|------|------|------|
| `StrategyFamily` | `src/bench/config.rs` | TOML 策略定义 |
| `StrategyBuilder` | `src/bench/harness.rs` | 从 TOML 构建策略 |
| `run_toml_bench()` | `src/bench/harness.rs` | TOML 驱动的完整流水线 |
| `BenchHarness::run()` | `src/bench/harness.rs` | 核心执行引擎 |
| `BenchStrategy` trait | `src/bench/strategy.rs` | 策略测试接口 |
| `param_dirname()` | `src/bench/config.rs` | 参数→文件名 |
| `update_strategy_stats()` | `src/bench/config.rs` | 更新 TOML stats |

---

## 输出格式

### info.json

```json
{
  "strategy": "peak_scan",
  "mode": "quick",
  "stats": {
    "fastest_ms": 5.2,
    "slowest_ms": 18.7,
    "avg_ms": 8.3,
    "median_ms": 7.1
  },
  "results": [
    {
      "params": { "threshold": 0.10, "expand": 0.10 },
      "frame_count": 1,
      "total_ms": 12.3,
      "avg_ms": 12.3,
      "extra": { "avg_ground": 12345, "ground_ratio": 24.69 }
    }
  ]
}
```

### TOML [stats] 段

```toml
[stats]
fastest_ms = 5.2
slowest_ms = 18.7
avg_ms = 8.3
median_ms = 7.1
last_run = "1746758400"
```

---

## 对比旧系统

| 方面 | 旧系统 | 新系统 |
|------|--------|--------|
| 策略定义 | Rust 代码硬编码 `build_sweep_strategies()` | TOML 文件，不改代码 |
| 参数修改 | 改 Rust + 重编译 | 改 TOML |
| 速度记录 | 无 | TOML `[stats]` 持久化 |
| 测试模式 | `--sweep` 模糊 | `--mode quick|full|single` 明确 |
| 慢策略处理 | 所有策略都测 | Full 模式自动跳过 >100ms |
| 输出层级 | `output/ground_bench/*.rdra` 扁平 | `output/bench/ground/peak_scan/*.rdra` 按策略 |
| 分析生成 | `param_sweep.py` 从 stdout JSON 解析 | `bench_pipeline.py` 从 info.json 读取 |
