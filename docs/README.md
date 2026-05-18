# 文档索引

## 架构设计

| 文档 | 说明 |
|------|------|
| [bench_design.md](bench_design.md) | Benchmark 管线设计规范（三级串联：地面→墙体→聚类） |
| [color.md](color.md) | Color 模块文档（YOLO ONNX 图像检测） |
| [kalman_guide.md](kalman_guide.md) | 卡尔曼滤波模块使用指南（9D CA 模型） |
| [wall_strategy_design.md](wall_strategy_design.md) | 墙体提取策略设计（BEV + 图像边缘检测） |
| [ground_detection_conclusion.md](ground_detection_conclusion.md) | 地面检测基准测试结论报告 |

## 精度与评估

| 文档 | 说明 |
|------|------|
| [baseline_accuracy.md](baseline_accuracy.md) | 基线精度评估（408 帧，3 轮平均 F1=0.745） |
| [evaluation_workflow.md](evaluation_workflow.md) | 验证流程文档（管线运行→精度评估→可视化） |
| [pipeline_evolution.md](pipeline_evolution.md) | 点云处理管线演化记录（从 DBSCAN 到三级级联） |
| [run_commands.md](run_commands.md) | 常用运行命令速查（构建/管线/eval/bench/Python） |

## 流程图

| 文件 | 说明 |
|------|------|
| [flowcharts/frame.svg](flowcharts/frame.svg) | 系统框架图 |
| [flowcharts/bev_edlines.drawio](flowcharts/bev_edlines.drawio) | EDLines 墙体检测流程图（Draw.io） |

## 归档文档

移至 `archive/` 的历史文档，仅供参考：

| 文档 | 说明 |
|------|------|
| `archive/design.md` | 旧架构设计（三级管线前的版本） |
| `archive/config.md` | 旧配置参考（字段名与实际不匹配） |
| `archive/lidar.md` | 旧 LiDAR 模块文档（DBSCAN 中心时代） |
| `archive/ground_strategy_analysis.md` | 地面策略分析（已被结论报告替代） |
| `archive/wall_strategy_analysis.md` | 墙体策略分析（已被设计文档替代） |
| `archive/nalgebra_matrix_layout_bug.md` | nalgebra 矩阵布局 Bug 记录 |
| `archive/perf-optimize.md` | 性能优化记录 |
| `archive/cluster_center_plan.md` | 聚类中心偏移修复方案 |
| `archive/dev_note_strategy_cascade.md` | 策略级联开发笔记 |
