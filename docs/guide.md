# Perple 文档导览

## 核心文档

- **[pipeline_evolution.md](pipeline_evolution.md)** — 点云处理管线技术演进：从 DBSCAN → 三层级联（地面→墙体→后聚类），解释了每个阶段的设计决策和工程权衡。
- **[pruneqt.md](pruneqt.md)** — PruneQt 自适应四叉树聚类算法完整描述：背景、难点、SplitPolicy、自适应深度公式、与 LV-DOT 的关联。含原始基线参数（depth=10, occ=4）作为永久参考。
- **[lvdot_vs_pruneqt.md](lvdot_vs_pruneqt.md)** — 聚类前降采样/过滤策略对比：LV-DOT 均匀体素滤波 vs PruneQt 自适应密度聚类。
- **[baseline_accuracy.md](baseline_accuracy.md)** — 管线基线精度评估结果：Person F1=0.745（P=82.3%, R=68.1%），MOTA=55.3%，IDF1=76.0%。不同策略对比数据。
- **[kalman_guide.md](kalman_guide.md)** — 卡尔曼滤波模块使用指南：9D 恒加速度模型（KalmanFilterCA）和 6D 常速备选。含初始化参数、IOU 门控、状态转移方程。
- **[ground_detection_conclusion.md](ground_detection_conclusion.md)** — 地面检测五种策略基准测试结论报告。倒装 LiDAR 场景下的策略选择建议。
- **[color.md](color.md)** — Color 模块（YOLO 图像目标检测）：模型结构、置信度过滤、输入预处理参数。

## 验证与评估

- **[evaluation_workflow.md](evaluation_workflow.md)** — 三层验证体系：单元测试 + 模块级 bench + 标注数据评估。含 eval_labeled / eval_tracking 的命令与输出格式说明。
- **[bench_design.md](bench_design.md)** — Benchmark 管线设计规范：三级串联架构（Ground→Wall→Cluster），可组合策略测试框架。

## 开发参考

- **[run_commands.md](run_commands.md)** — 运行命令速查：构建、测试、bench 示例、Python 分析管线、protobuf 生成。
- **[chart_style_guide.md](chart_style_guide.md)** — 论文图表风格指南：matplotlib 全局样式、颜色板、字体、图例规范的统一配置。

## 流程图

- **[flowcharts/frame.png](flowcharts/frame.png)** — 单帧数据处理流程（渲染视图）
- **[flowcharts/frame.svg](flowcharts/frame.svg)** — 单帧数据处理流程（可编辑 SVG）
- **[flowcharts/target_state.png](flowcharts/target_state.png)** — 跟踪目标状态机（Static→Floating→Moving↔Movable）
- **[flowcharts/frame_dif.svg](flowcharts/frame_dif.svg)** — 管线差异对比图

## 归档（历史记录）

- **[archive/design.md](archive/design.md)** — 初始设计文档
- **[archive/config.md](archive/config.md)** — 配置系统详细说明
- **[archive/lidar.md](archive/lidar.md)** — LiDAR 模块旧版文档
- **[archive/cluster_center_plan.md](archive/cluster_center_plan.md)** — 聚类中心选取方案设计
- **[archive/ground_strategy_analysis.md](archive/ground_strategy_analysis.md)** — 地面检测策略分析
- **[archive/wall_strategy_analysis.md](archive/wall_strategy_analysis.md)** — 墙体检测策略分析
- **[archive/dev_note_strategy_cascade.md](archive/dev_note_strategy_cascade.md)** — 策略级联开发笔记
- **[archive/nalgebra_matrix_layout_bug.md](archive/nalgebra_matrix_layout_bug.md)** — nalgebra 矩阵布局 row-major vs column-major 踩坑记录
- **[archive/perf-optimize.md](archive/perf-optimize.md)** — 性能优化记录
