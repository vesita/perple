# Perple 相机 + 雷达联合检测工具

## 项目介绍

基于 Rust 的相机 + LiDAR 多模态行人检测与跟踪系统，使用三级点云处理管线（地面 → 墙体 → 聚类），配合 YOLO 检测 + 匈牙利关联 + 卡尔曼滤波跟踪。

### 当前性能（中心距 0.5m 匹配，408帧）

| 指标 | Person 过滤 | 全部类别 |
|------|:---:|:---:|
| Precision | **85.1%** | 58.8% |
| Recall | **69.0%** | 78.8% |
| F1 | **0.762** | 0.673 |
| FP | 148 | 675 |

### 致谢

墙体检测模块的 EDLines 算法参考了 [opencv_idz](https://github.com/DemonFromRussia/opencv_idz) 项目中的 C++ 参考实现（Akinlar & Topal, 2011），在此表示感谢。

### 关键特性

- **三级点云处理**: 地面检测（PeakScan）→ 墙体检测（BevEdLines）→ 聚类（剪叶聚类 PruneQt）
- **墙体检测**: EDLines 锚点检测 + 链式追踪，二进制方向（无三角函数），~14ms/帧
- **跟踪**: 匈牙利关联 + 9D CA 卡尔曼 + 点云投票 + 航迹评分 + 几何 fallback
- **融合**: 2D YOLO 检测通过标定矩阵映射到 3D 点云，YOLO 帧间标签平滑

## 目录结构

```
.
├── config/                         # TOML 配置文件
├── docs/                           # 设计文档（见 docs/README.md）
├── crates/adskalman-rs             # 自适应卡尔曼滤波库
├── examples/                       # 基准测试 & 功能示例
├── scripts/                        # Python 分析 & 训练脚本
├── prompt/                         # AI 提示文件
├── src/
│   ├── cloud/                      # 点云处理
│   │   ├── classify/               #   聚类策略族 + 分类管线
│   │   │   ├── cluster.rs          #     聚类器 + YOLO 辅助分裂
│   │   │   ├── core.rs             #     三级管线（地面→墙体→聚类）
│   │   │   └── strategy/           #     9 种聚类策略
│   │   ├── ground/                 #   5 种地面检测策略
│   │   ├── wall/                   #   3 种墙体检测策略
│   │   ├── core.rs                 #   LiDAR 主处理
│   │   ├── output.rs               #   CldBud 输出类型
│   │   └── ego_motion.rs           #   自车速度估计
│   ├── color/                      # YOLO ONNX 图像检测
│   │   ├── core.rs                 #   检测 + 去畸变
│   │   ├── detect.rs               #   YOLO 检测器
│   │   ├── model.rs                #   ONNX 推理引擎
│   │   └── look.rs                 #   2D→3D 视线投影
│   ├── tracker/                    # 多目标跟踪
│   │   ├── core.rs                 #   匈牙利关联 + 航迹管理
│   │   ├── kalman.rs               #   模块根（9D CA + 6D CV 卡尔曼）
│   │   ├── kalman/                 #   子模块
│   │   │   ├── ca.rs               #     9D 恒加速度模型（主推）
│   │   │   └── cv.rs               #     6D 常速度模型（备选）
│   │   ├── association.rs          #   数据关联
│   │   ├── object.rs               #   跟踪目标（状态机）
│   │   ├── lifecycle.rs            #   航迹分级管理
│   │   ├── trick.rs                #   几何 fallback
│   │   ├── output.rs               #   Target 输出
│   │   ├── features.rs             #   特征提取
│   │   ├── hungarian.rs            #   匈牙利算法
│   │   └── analysis.rs             #   跟踪分析
│   ├── fuse.rs                     # 2D-3D 融合
│   ├── config.rs                   # TOML 配置加载
│   ├── swapl.rs                    # 全局数据总线
│   ├── yolo_smooth.rs              # YOLO 标签平滑
│   ├── perple.rs                   # MultiLoop 编排
│   ├── main.rs                     # 生产管线入口
│   └── bench/                      # Benchmark 框架
├── Cargo.toml
├── CLAUDE.md
└── README.md
```

## 文档

项目详细设计文档位于 [`docs/`](docs/) 目录，以 [`docs/README.md`](docs/README.md) 为索引：

| 类别 | 文档 |
|------|------|
| 架构设计 | [bench_design.md](docs/bench_design.md), [color.md](docs/color.md), [kalman_guide.md](docs/kalman_guide.md), [wall_strategy_design.md](docs/wall_strategy_design.md), [ground_detection_conclusion.md](docs/ground_detection_conclusion.md) |
| 精度评估 | [baseline_accuracy.md](docs/baseline_accuracy.md), [evaluation_workflow.md](docs/evaluation_workflow.md) |
| 管线演化 | [pipeline_evolution.md](docs/pipeline_evolution.md) |
| 流程图 | [flowcharts/frame.svg](docs/flowcharts/frame.svg) |

## 核心模块说明

### 1. Color 模块 (`src/color/`)

基于 YOLO ONNX 的图像目标检测模块：

- **model.rs**: ONNX 模型加载和推理引擎封装（ORT 会话管理）
- **image.rs**: 图像预处理（Letterbox 缩放、归一化、填充）
- **detect.rs**: YOLO 检测器（置信度阈值 + NMS 后处理）
- **output.rs**: 检测结果输出容器 `ClrBud`
- **core.rs**: 核心检测逻辑（去畸变 → Letterbox → 推理 → 结果写入 DualBuf）
- **look.rs**: 2D 检测框 → 3D 视线向量（`Sight`）投影
- **utils.rs**: 可视化工具（边界框绘制、坐标转换）

### 2. Cloud 模块 (`src/cloud/`)

LiDAR 点云处理模块，核心为三级管线（地面 → 墙体 → 聚类）：

- **classify/core.rs**: 管线编排 — `GroundPickStrategy::pick()` → `WallPickStrategy::pick()` → `ClusteringStrategy::run()` + YOLO refine
- **classify/cluster.rs**: 聚类器 `Cluster`，策略 trait + 工厂模式，支持 `prune_qt` / `dbscan_qt` / `cc` / `lvdot` / `ransac` / `seq` / `xy_dbscan` 等 9 种策略
- **classify/strategy/**: 各聚类策略实现（`prune_qt.rs`, `dbscan.rs`, `cc_cluster.rs`, `lvdot_cluster.rs`, `range_image.rs`, `ransac_cluster.rs`, `seq_cluster.rs`, `xy_grid_dbscan.rs`）
- **ground/**: 地面检测策略族 — `PeakScan`（默认）、`HistogramExpand`、`RansacGround`、`HistoseedPlane`、`GpfGround`
- **wall/**: 墙体检测策略族 — `BevEdLines`（默认，~14ms）、`BevLsd`（~17ms）、`BevHough`（备选）
- **core.rs**: LiDAR 主处理（读取 Stream 输入，调用 Classify）
- **output.rs**: `CldBud` 输出类型（3D 边界框 + 质心）
- **ego_motion.rs**: 基于地面平面方程帧间变化的自车速度估计

### 3. Tracker 模块 (`src/tracker/`)

多目标跟踪模块，基于卡尔曼滤波的数据关联和轨迹管理：

- **core.rs**: 跟踪器核心
  - 数据关联（匈牙利算法，IoU + 马氏距离 + 卡方门控）
  - 轨迹管理（创建、更新、删除、航迹评分 N≥3）
  - 点云投票（voxel occupancy → KDE 投票 → 置信度提升/衰减）
  - 几何 fallback：盲区行人补充检测（recall 26.3% → 61.4%）
- **kalman**: 9D CA 卡尔曼 + 6D CV 备选（`kalman.rs` 模块根 + `kalman/` 子模块）
  - 恒加速度模型 `[x,y,z,vx,vy,vz,ax,ay,az,l,w,h]`
  - 距离自适应 / 置信度自适应测量噪声
  - Z 轴独立 EMA 跟踪（不纳入卡尔曼状态）
- **hungarian.rs**: 匈牙利算法矩阵求解
- **output.rs**: 跟踪结果 `Target`（轨迹 ID、速度、动态/静态分类）

### 4. Utils 模块 (`src/utils/`)

通用工具函数和数据结构：

- **boxes**: `Box2D` / `Box3D` 包围盒定义（`boxes.rs` 模块根，子模块 `boxes/`）
  - `bev_iou()` — BEV 2D 多边形交并比（行人检测推荐）
  - `obb_iou()` — 真 3D OBB 交并比（三角形网格裁剪算法）
  - `cloud2box()` — 从点云计算 AABB
- **stream.rs**: 数据流管理
  - `Stream<T>`: 固定容量循环缓冲区（写满覆盖旧数据）
  - `Eap<T>` = `Arc<Mutex<T>>`: 线程安全共享指针
  - `Cream<I,O>`: 双向流适配器
  - `DualBuffer<T>`: 双缓冲（检测阶段写 producer → swap → 后融合阶段读 consumer）
- **muloop.rs**: MultiLoop 异步循环运行器（任务编排）
- **sort.rs**: 排序算法（quick_sort, group_sort）

### 5. Fuse 模块 (`src/fuse.rs`)

2D-3D 融合模块：

- 读取 `cld_buds_raw`（聚类结果）+ `clr_objs`（YOLO 检测）
- 将 3D 簇投影到 2D，计算投影框与 YOLO 框的 IoU 匹配
- 无 YOLO 时透传原始聚类结果
- 有 YOLO 时执行投影匹配 + 合并 + 标签更新

### 6. Optional 模块 (`src/optional/`)

可选功能组件：

- **data_loader.rs**: 数据加载器（帧数限制、下采样、异步预加载）
- **visual.rs**: rerun 3D 可视化集成

### 7. Config 模块 (`src/config.rs`)

TOML 配置管理系统：

- **全量加载**: `config/default.toml` → `OnceLock<Config>` → `fixif()`
- **增量更新**: 运行时 `update_from_toml()` 覆盖部分字段
- **全局单例**: `fixif()` 返回 `&'static Config`（懒加载、线程安全）
- **配置分组**: camera（内外参）、cluster（策略/体素/eps）、ground（策略/expand）、wall（策略/距离）、tracker（失踪帧数/速度阈值）
- **Option 字段处理**: `min_points_per_cluster` 等使用 `Some()` 显式包裹

## 环境搭建与部署

### 必需工具

- **Rust toolchain**: 版本 1.92.0+, Edition 2024, stable 通道
- **Python**: 3.8+
- **uv**: Python 环境管理工具
- **ONNX Runtime**: 模型推理引擎
- **PyTorch**: 模型训练框架
- **Git**: 版本控制

### 安装步骤

1. 安装 Rust
    推荐先配置 rust 镜像源，然后安装 Rust

    1.1 配置 rustup 镜像源

        rustup 源：<https://mirrors.tuna.tsinghua.edu.cn/help/rustup/>

    1.2 安装 rust 编译器
        ```bash
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
        ```

    1.3 配置 cargo 源

        cargo 源：<https://mirrors.tuna.tsinghua.edu.cn/help/crates.io-index/>

2. 安装 Python

   本项目使用 uv 管理虚拟环境

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

### 构建项目

```
# 开发构建（调试模式）
cargo build

# 发布构建（优化模式）
cargo build --release

# 运行测试
cargo test --all

# 代码格式化检查
cargo fmt --all

# 代码 lint 检查
cargo clippy -- -D warnings
```

### 配置 Python 环境

```
# 同步依赖（创建虚拟环境并安装依赖）
uv sync

# 运行 Python 脚本
uv run python scripts/dev/train.py
```

### 运行示例

```
# 运行 Rust 示例程序
cargo run --example lidar_reader
cargo run --example muloop_example
cargo run --example visualize

# 运行 Python 训练脚本
uv run python scripts/dev/train.py
uv run python scripts/dev/continuous_train.py
uv run python scripts/dev/eval.py

# 模型导出为 ONNX
uv run python scripts/dev/to_onnx.py
```

## 需要预先安装的依赖包

1. libssl-dev
2. pkg-config
3. libfontconfig1-dev
4. libwayland-dev
5. libasound2-dev
6. libudev-dev
7. libopenblas-dev
