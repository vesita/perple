# Perple 相机 + 雷达联合检测工具

## 项目介绍

本项目是一个基于 Rust 实现的相机与雷达多模态障碍物检测系统。

开发于 Linux 环境

### 项目特点

- **高性能数据处理**: 使用 Rust 实现图像和点云数据的高效处理
- **卡尔曼滤波跟踪**: 基于 adskalman 库实现常速模型的多目标跟踪
- **灵活训练控制**: Python 脚本提供灵活的训练流程控制
- **持续训练机制**: 自动化的训练 - 评估 - 归档闭环，解决数据增强导致的初期性能波动问题
- **多模态融合**: 同时支持图像和 LiDAR 点云数据处理及融合
- **模型导出**: 支持将训练好的模型导出为 ONNX 格式便于部署
- **可视化工具**: 使用 rerun 进行 3D 数据可视化

## 目录结构

```file
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
│   ├── kalman_guide.md
│   └── lidar.md
├── examples
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
│   │   │   ├── claster.rs
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
│   │   ├── stream.rs
│   │   └── world.rs
│   ├── cloud.rs
│   ├── color.rs
│   ├── config.rs
│   ├── lib.rs
│   ├── main.rs
│   ├── optional.rs
│   ├── perple.rs
│   ├── swapl.rs
│   ├── tracker.rs
│   └── utils.rs
├── Cargo.toml
├── README.md
├── TODO.md
└── pyproject.toml
```

## 核心模块说明

### 1. Color 模块 (`src/color/`)

基于 YOLO 模型的图像目标检测模块：

- **model.rs**: ONNX 模型加载
- **image.rs**: 图像预处理和加载
- **detect.rs**: YOLO 检测器实现
- **output.rs**: 检测结果输出容器
- **utils.rs**: 可视化和后处理工具
- **core.rs**: 核心检测逻辑
- **look.rs**: 视觉分析功能

### 2. Cloud 模块 (`src/cloud/`)

LiDAR 点云数据处理模块：

- **core.rs**: 点云数据核心处理
- **classify/**: 点云分类子模块（包含聚类、KD-Tree、四叉树等）
- **output.rs**: 点云检测结果输出

### 3. Tracker 模块 (`src/tracker/`)

多目标跟踪模块：

- **core.rs**: 跟踪器核心逻辑，数据关联和轨迹管理
- **kalman.rs**: 卡尔曼滤波器实现（常速模型）
- **output.rs**: 跟踪结果输出

### 4. Utils 模块 (`src/utils/`)

通用工具函数：

- **boxes.rs**: 2D/3D 边界框定义
- **stream.rs**: 数据流管理
- **muloop.rs**: 多循环模式支持
- **sight.rs**: 视线检测
- **sort.rs**: 排序算法
- 其他工具函数

### 5. Config 模块 (`src/config.rs`)

配置管理系统：

- 支持 TOML 格式配置
- 全量配置加载和增量更新
- 全局配置单例访问

## 环境搭建与部署

### 必需工具

- Rust toolchain (edition 2024)
- Python 3.8+
- pip / pip-tools
- ONNX Runtime
- PyTorch

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

```bash
# 开发构建
cargo build
```

### 配置 Python 环境

```bash
uv sync
```

### 运行示例

```bash
# 运行示例程序
cargo run --example lidar_reader
cargo run --example muloop_example

# 运行 Python 训练脚本
python scripts/dev/train.py
python scripts/dev/continuous_train.py
```

## 需要预先安装的依赖包

1. libssl-dev
2. pkg-config
3. libfontconfig1-dev
4. libwayland-dev
5. libasound2-dev
6. libudev-dev
7. libblas-dev
