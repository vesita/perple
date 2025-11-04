# Perple - YOLO 模型训练项目

## 项目介绍

本项目是一个基于 YOLO 的目标检测模型训练系统，使用 Rust 作为主框架，Python 脚本进行模型训练和评估。

## 目录结构

```
.
├── examples
│   └── counter.rs
├── py-scripts
│   ├── configs
│   │   └── model.yaml
│   ├── dev
│   │   ├── eval.py
│   │   └── train.py
│   ├── hyper
│   │   ├── amt.yaml
│   │   ├── dataset.yaml
│   │   ├── optimizer.yaml
│   │   └── train.yaml
│   ├── model
│   │   └── records
│   ├── utils
│   │   ├── __init__.py
│   │   └── archive.py
│   └── __init__.py
├── src
│   ├── lib.rs
│   ├── main.rs
│   └── utils.rs
├── Cargo.toml
├── README.md
└── pyproject.toml
```

## 训练策略优化

### 传统训练方式

使用 `py-scripts/dev/train.py` 进行标准训练。

### 持续训练策略

为了克服数据增强带来的初始性能波动问题，我们引入了持续训练策略：

1. 使用 `py-scripts/dev/continuous_train.py` 脚本
2. 该脚本会自动评估每轮训练的结果
3. 如果模型未达到预期性能，则继续下一轮训练
4. 每轮训练的结果都会被自动归档

### 配置优化

1. 增加了训练轮数 (epochs) 从 32 到 100
2. 增加了批次大小 (batch) 从 8 到 16
3. 增加了早停耐心值 (patience) 从 20 到 50
4. 优化了数据增强参数，特别是 copy_paste 从 0.2 提高到 0.3

### 使用方法

```bash
# 标准训练
python py-scripts/dev/train.py

# 持续训练（推荐）
python py-scripts/dev/continuous_train.py
```

### 训练建议

1. **首次训练**：使用标准训练方式直到模型收敛
2. **持续优化**：使用持续训练策略进一步提升模型性能
3. **监控指标**：关注 mAP50 和 mAP50-95 指标，确保模型性能提升
4. **资源管理**：持续训练会自动归档每轮结果，便于回溯和管理
