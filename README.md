# Perple 相机+雷达联合检测工具

## 项目介绍

本项目是一个基于相机与雷达的障碍物检测工具，基于 Rust 实现。

开发于linux环境

### 项目特点

- **高性能数据处理**: 使用 Rust 实现图像和点云数据的高效处理
- **灵活训练控制**: Python 脚本提供灵活的训练流程控制
- **持续训练机制**: 自动化的训练-评估-归档闭环，解决数据增强导致的初期性能波动问题
- **多模态支持**: 同时支持图像和 LiDAR 点云数据处理
- **模型导出**: 支持将训练好的模型导出为 ONNX 格式便于部署
- **可视化工具**: 使用 rerun 进行 3D 数据可视化

## 目录结构

```file
.
├── examples
│   ├── image_test.rs
│   ├── lidar_reader.rs
│   ├── loop_modes.rs
│   └── muloop_example.rs
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
│           ├── 25_11_03_00
│           │   └── args.yaml
│           ├── 25_11_03_01
│           │   └── args.yaml
│           └── ...
├── src
│   ├── color
│   │   ├── array.rs
│   │   ├── bounds.rs
│   │   ├── core.rs
│   │   ├── detect.rs
│   │   ├── image.rs
│   │   ├── model.rs
│   │   └── utils.rs
│   ├── lidar
│   │   ├── bounds.rs
│   │   ├── claster.rs
│   │   ├── core.rs
│   │   └── lifra.rs
│   ├── utils
│   │   ├── muloop.rs
│   │   ├── sort.rs
│   │   └── stream.rs
│   ├── color.rs
│   ├── config.rs
│   ├── lib.rs
│   ├── lidar.rs
│   ├── main.rs
│   ├── perple.rs
│   ├── swapl.rs
│   ├── utils.rs
│   └── world.rs
├── Cargo.toml
├── README.md
└── pyproject.toml
```

## 环境搭建与部署

### 必需工具

- Rust toolchain (edition 2024)
- Python 3.8+
- pip / pip-tools
- ONNX Runtime
- PyTorch

### 安装步骤

1. 安装Rust
    推荐先配置rust镜像源，然后安装Rust
    1.1 配置rustup镜像源
        rustup源: <https://mirrors.tuna.tsinghua.edu.cn/help/rustup/>
    1.2 安装rust编译器
        ```bash
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
        ```
    1.3 配置cargo源
        cargo源: <https://mirrors.tuna.tsinghua.edu.cn/help/crates.io-index/>
2. 安装Python
    本项目使用uv管理虚拟环境

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

### 构建项目

```bash
# 开发构建
cargo build
```

### 配置python环境

```bash
uv sync
```

### 运行示例

```bash
# 运行示例程序
cargo run --example lidar_reader
cargo run --example image_test

# 运行Python训练脚本
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
7. libopenblas-dev
