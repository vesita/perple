# Perple 相机 + 雷达联合检测工具

## 项目介绍

本项目是一个基于 Rust 实现的相机与雷达多模态障碍物检测系统，采用常速模型 (Constant Velocity Model) 卡尔曼滤波进行多目标跟踪。

开发于 Linux 环境

### 项目特点

- **高性能数据处理**: 使用 Rust 实现图像和点云数据的高效处理
- **卡尔曼滤波跟踪**: 基于 `adskalman` 库实现常速模型的多目标跟踪
- **灵活训练控制**: Python 脚本提供灵活的训练流程控制
- **持续训练机制**: 自动化的训练 - 评估 - 归档闭环，解决数据增强导致的初期性能波动问题
- **多模态融合**: 同时支持图像和 LiDAR 点云数据处理及融合
- **模型导出**: 支持将训练好的模型导出为 ONNX 格式便于部署
- **可视化工具**: 使用 rerun 进行 3D 数据可视化
- **配置管理**: 支持 TOML 格式的全量和增量配置更新

## 目录结构

```
.
├── config
│   ├── default.toml
│   └── update_example.toml
├── crates
│   └── adskalman-rs
│       ├── src
│       │   ├── lib.rs
│       │   └── state_and_covariance.rs
│       └── tests
│           └── integration.rs
├── doc
│   ├── color.md
│   ├── config.md
│   ├── ground_detection_conclusion.md
│   ├── kalman_guide.md
│   └── lidar.md
├── examples
│   ├── check_axis.rs
│   ├── ground_bench.rs
│   ├── lidar_reader.rs
│   ├── muloop_example.rs
│   └── visualize.rs
├── scripts
│   ├── ana
│   │   └── pcd_analyzer.py
│   ├── configs
│   │   └── model.yaml
│   ├── dev
│   │   ├── archive.py
│   │   ├── continuous_train.py
│   │   ├── eval.py
│   │   ├── to_onnx.py
│   │   └── train.py
│   ├── hyper
│   │   ├── dataset.yaml
│   │   ├── hyp.yaml
│   │   └── my.yaml
│   └── model
│       ├── original
│       │   └── yolo11.yaml
│       └── records
│           └── (训练记录)
├── src
│   ├── cloud/
│   │   ├── classify/
│   │   │   ├── cluster.rs
│   │   │   ├── core.rs
│   │   │   ├── environment.rs
│   │   │   ├── kdtree.rs
│   │   │   ├── quadtree.rs
│   │   │   └── somecode.rs
│   │   ├── classify.rs
│   │   ├── core.rs
│   │   └── output.rs
│   ├── color/
│   │   ├── core.rs
│   │   ├── detect.rs
│   │   ├── image.rs
│   │   ├── look.rs
│   │   ├── model.rs
│   │   ├── output.rs
│   │   └── utils.rs
│   ├── tracker/
│   │   ├── core.rs
│   │   ├── hungarian.rs
│   │   ├── kalman.rs
│   │   └── output.rs
│   ├── optional/
│   │   ├── visual/
│   │   │   ├── interface/
│   │   │   │   └── draw.rs
│   │   │   ├── utils/
│   │   │   │   ├── coordinate.rs
│   │   │   │   └── wirefra.rs
│   │   │   ├── core.rs
│   │   │   ├── interface.rs
│   │   │   ├── resource.rs
│   │   │   ├── scripts.rs
│   │   │   └── utils.rs
│   │   ├── data_loader.rs
│   │   └── visual.rs
│   ├── utils/
│   │   ├── boxes.rs
│   │   ├── combine.rs
│   │   ├── muloop.rs
│   │   ├── random.rs
│   │   ├── sight.rs
│   │   ├── sort.rs
│   │   └── stream.rs
│   ├── cloud.rs
│   ├── color.rs
│   ├── config.rs
│   ├── fuse.rs
│   ├── lib.rs
│   ├── main.rs
│   ├── optional.rs
│   ├── perple.rs
│   ├── swapl.rs
│   ├── tracker.rs
│   └── utils.rs
├── Cargo.toml
├── PLAN.md
├── README.md
├── TODO.md
└── pyproject.toml
```

## 核心模块说明

### 1. Color 模块 (`src/color/`)

基于 YOLO 模型的图像目标检测模块：

- **model.rs**: ONNX(Open Neural Network Exchange) 模型加载和推理引擎封装
- **image.rs**: 图像预处理和加载（包括缩放、归一化等）
- **detect.rs**: YOLO 检测器实现（包含置信度阈值和 NMS 后处理）
- **output.rs**: 检测结果输出容器 `ClrBud`（固定容量，避免动态内存分配）
- **utils.rs**: 可视化和后处理工具（边界框绘制、坐标转换）
- **core.rs**: 核心检测逻辑（整合各组件）
- **look.rs**: 视觉分析功能（场景理解、统计信息）

### 2. Cloud 模块 (`src/cloud/`)

LiDAR 点云数据处理模块：

- **core.rs**: 点云数据核心处理（PCD 格式读取、LiDAR 原生帧）
- **classify/**: 点云分类子模块
  - **cluster.rs**: DBSCAN 聚类算法实现（基于距离阈值和最小点数）
  - **core.rs**: 分类核心逻辑（地面检测 → 聚类）
  - **environment.rs**: 地面检测（histoseed 混合策略：直方图种子 + RANSAC 生长），支持倒装 LiDAR
  - **kdtree.rs**: KD-Tree 空间索引（快速近邻搜索）
  - **quadtree.rs**: 四叉树空间索引（平面区域分割）
  - **somecode.rs**: 辅助代码和工具函数
- **output.rs**: 点云检测结果输出容器 `CldBud`（存储 3D 边界框）

### 3. Tracker 模块 (`src/tracker/`)

多目标跟踪模块，基于卡尔曼滤波的数据关联和轨迹管理：

- **core.rs**: 跟踪器核心逻辑
  - 数据关联（匈牙利算法，马氏距离 + 卡方门控）
  - 轨迹管理（创建、更新、删除）
  - 失踪目标处理（最大失踪帧数控制）
  - 速度聚类动态/静态分类（DBSCAN 聚类 Kalman 速度向量）
- **kalman.rs**: 卡尔曼滤波器实现
  - 常速运动模型 (Constant Velocity Model)
  - 6 维状态向量 `[x, y, z, vx, vy, vz]ᵀ`
  - 位置观测模型，动态 dt 帧间隔计算
  - 协方差管理和数值稳定性优化
- **hungarian.rs**: 匈牙利算法求解最优匹配
- **output.rs**: 跟踪结果输出容器 `Target`（包含轨迹 ID、速度、动态/静态分类）

### 4. Utils 模块 (`src/utils/`)

通用工具函数和数据结构：

- **boxes.rs**: 2D/3D 边界框定义
  - `Box2D`: 二维边界框（图像检测）
  - `Box3D`: 三维边界框（点云检测），支持体积、面积、IoU 计算
- **stream.rs**: 数据流管理
  - `Stream<T>`: 固定容量的循环缓冲区
  - `Eap`: 早期访问协议 trait
- **muloop.rs**: 多循环模式支持（多传感器同步）
- **random.rs**: 随机数生成工具
- **sight.rs**: 视线检测（可见性判断）
- **sort.rs**: 排序算法（按置信度、距离等）
- **combine.rs**: 多模态融合工具
- **world.rs**: 世界坐标系管理

### 5. Fuse 模块 (`src/fuse.rs`)

2D-3D 融合模块：

- 计算 `cam_from_lidar` 标定矩阵（`inv(camera.extrinsic) * lidar.extrinsic`）
- 将 LiDAR 点云投影到相机图像平面
- 为检测框提供深度信息

### 6. Optional 模块 (`src/optional/`)

可选功能组件：

- **data_loader.rs**: 数据加载器，支持帧数限制、下采样、独立路径模式
- **visual.rs**: rerun 3D 可视化集成

### 7. Config 模块 (`src/config.rs`)

配置管理系统：

- 支持 TOML 格式配置文件
- **全量配置加载**: 从 `config/default.toml` 加载完整配置
- **增量配置更新**: 运行时通过增量文件更新部分配置项
- **全局配置单例**: 通过 `fixif()` 函数访问（懒加载、线程安全）
- **支持的配置项**:
  - 基础配置（流容量、分辨率、类别标签）
  - 目标检测参数（置信度阈值、NMS 阈值、模型路径）
  - DBSCAN 聚类参数（最小点数、距离阈值）
  - 地面检测参数（histoseed: expand, ransac_distance, upside_down）
  - 相机内参和外参矩阵
  - 雷达外参矩阵
  - 点云聚类参数（合并耐心值、体素大小）

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
