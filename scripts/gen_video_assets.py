"""
视频素材生成器 — "挑战与优化之路" 可视化素材

生成图表:
  1. complexity_curve.png   — O(n²) vs O(n log n) vs O(k log k) 复杂度曲线
  2. pipeline_flow.png       — 点云逐级压缩漏斗图
  3. timing_breakdown.png    — Rust 管线子步骤耗时分解
  4. speed_comparison.png    — 各算法端到端速度对比
  5. pruning_effect.png      — 四叉树剪叶阈值分析
  6. storyboard.txt          — 视频分镜脚本（中文）
"""

import sys
import json
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.chart_style import (
    C_BLUE, C_RED, C_GREEN, C_YELLOW, C_ORANGE, C_GRAY, C_DARK, C_CYAN,
    COLORS_10, savefig, style_ax, SIZES
)

OUT = Path("output/bench_video_v2")
OUT.mkdir(parents=True, exist_ok=True)

# Rust 参考数据
RUST = {
    'points_per_frame': 19500,
    'ground_pct': 25,
    'wall_pct': 32,
    'timing_ms': {
        '地面\n(PeakScan)': 0.6,
        '墙体\n(BevEDLines)': 19.5,
        '剪叶聚类': 16.8,
        'YOLO 精化': 2.2,
        '管线总计': 39.1,
    },
    'comparison': {
        '暴力\nDBSCAN': 8000.0,
        'KD-tree\nDBSCAN': 120.0,
        'LV-DOT\n(Rust)': 16.9,
        '剪叶聚类\n(Rust)': 16.8,
    }
}


def plot_complexity():
    """理论复杂度曲线。"""
    n = np.logspace(2, 4.3, 300)
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=200)

    ax.plot(n, n**2, color=C_RED, lw=2.5, label='O(n^2) 暴力法')
    ax.plot(n, n * np.log2(n), color=C_ORANGE, lw=2.5, label='O(n log n) KD-tree')
    ax.plot(n, (n/30) * np.log2(n/30), color=C_GREEN, lw=2.5, label='O(k log k) 剪叶聚类（k=n/30）')

    ax.axvline(20000, color=COLORS_10[-2], ls='--', alpha=0.5)
    ax.text(20300, 10, '20K 点/帧', fontsize=10, color=COLORS_10[-2], rotation=90)

    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('点数量 (n)', fontsize=11)
    ax.set_ylabel('相对计算开销', fontsize=11)
    ax.set_title('算法复杂度对比', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    ax.set_xlim(100, 20000)
    style_ax(ax, grid_axis='both')

    savefig(fig, OUT / "complexity_curve.png")
    print(f"[OK] complexity_curve.png")


def plot_pipeline():
    """级联压缩漏斗图。"""
    stages = ['原始 LiDAR\n20000 点', '距离过滤\n(10m)\n19500 点', '地面过滤后\n15456 点', '墙体过滤后\n9274 点', '剪叶保留质心\n~200 个']
    counts = [20000, 19500, 15456, 9274, 200]
    colors = [C_GRAY, C_DARK, C_BLUE, C_ORANGE, C_GREEN]
    pcts = [100, 97, 77, 46, 1.0]

    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=200)
    bars = ax.bar(range(5), counts, width=0.55, color=colors, edgecolor='white', lw=2)

    for bar, c, p in zip(bars, counts, pcts):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+200,
                f'{c:,} 点（{p}%）', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_xticks(range(5))
    ax.set_xticklabels(stages, fontsize=10)
    ax.set_ylabel('点数量', fontsize=11)
    style_ax(ax, grid_axis='y')

    savefig(fig, OUT / "pipeline_flow.png")
    print(f"[OK] pipeline_flow.png")


def plot_timing():
    """Rust 管线耗时分解（水平条形图）。"""
    labels = list(RUST['timing_ms'].keys())
    times = list(RUST['timing_ms'].values())
    colors = [C_BLUE, C_ORANGE, C_GREEN, C_CYAN, C_RED]

    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=200)
    bars = ax.barh(labels, times, color=colors, edgecolor='white', height=0.5)

    for bar, v in zip(bars, times):
        ax.text(bar.get_width()+0.3, bar.get_y()+bar.get_height()/2,
                f'{v:.1f} ms', ha='left', va='center', fontsize=11, fontweight='bold')

    ax.axvline(50, color=C_RED, ls='--', lw=1.5, alpha=0.7, label='50 ms 时限（20 Hz）')
    ax.legend(fontsize=10, loc='lower right')
    ax.set_xlabel('耗时 (ms)', fontsize=11)
    ax.set_title('Rust 管线耗时分解', fontsize=13, fontweight='bold')
    style_ax(ax, grid_axis='x')

    savefig(fig, OUT / "timing_breakdown.png")
    print(f"[OK] timing_breakdown.png")


def plot_speed():
    """各算法速度对比柱状图。"""
    cats = list(RUST['comparison'].keys())
    vals = list(RUST['comparison'].values())
    colors = [C_RED, C_ORANGE, C_BLUE, C_GREEN]

    fig, ax = plt.subplots(figsize=(9, 5), dpi=200)
    bars = ax.bar(cats, vals, color=colors, edgecolor='white', width=0.5, lw=1.5)

    for bar, v in zip(bars, vals):
        label = f'{v:.0f} ms' if v >= 1 else f'{v:.1f} ms'
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.02,
                label, ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.axhline(50, color=C_RED, ls='--', lw=1.5, alpha=0.7, label='50 ms（20 Hz 阈值）')
    ax.legend(fontsize=10)
    ax.set_yscale('log')
    ax.set_ylabel('单帧耗时 (ms)', fontsize=11)
    ax.set_title('算法速度对比（Rust / 模拟）', fontsize=13, fontweight='bold')
    style_ax(ax, grid_axis='y')

    savefig(fig, OUT / "speed_comparison.png")
    print(f"[OK] speed_comparison.png")


def plot_pruning():
    """剪叶阈值分析（双轴图）。"""
    occ = np.array([1, 2, 3, 5, 10, 20, 30, 50])
    centroids = 20000 / occ

    fig, ax1 = plt.subplots(figsize=(9, 4.5), dpi=200)
    ax2 = ax1.twinx()

    l1 = ax1.plot(occ, centroids, 'o-', color=C_BLUE, lw=2, ms=8, label='剪叶后质心数')
    ax1.set_xlabel('叶节点最少点数 (min_occ)', fontsize=11)
    ax1.set_ylabel('质心数量', fontsize=11, color=C_BLUE)
    ax1.tick_params(axis='y', labelcolor=C_BLUE)

    l2 = ax2.plot(occ, centroids * np.log2(centroids) / 100, 's-', color=C_RED, lw=2, ms=8, label='估计聚类开销')
    ax2.set_ylabel('相对聚类开销', fontsize=11, color=C_RED)
    ax2.tick_params(axis='y', labelcolor=C_RED)

    ax1.axvline(3, color=C_GRAY, ls='--', alpha=0.5)
    ax1.text(3.2, ax1.get_ylim()[1]*0.9, '默认\nmin_occ=3', fontsize=10, color=C_GRAY)

    lines = l1 + l2
    ax1.legend(lines, [l.get_label() for l in lines], fontsize=10, loc='upper right')
    ax1.set_title('剪叶阈值 vs. 聚类复杂度', fontsize=13, fontweight='bold')
    ax1.set_xscale('log')
    style_ax(ax1, grid_axis='both')

    savefig(fig, OUT / "pruning_effect.png")
    print(f"[OK] pruning_effect.png")


def plot_voxel_comparison():
    """体素下采样 vs 管线压缩对比图。"""
    raw = 20000
    pipeline = 200
    voxel_pts = [raw, pipeline, 6384, 3552, 2310, 1721, 1037, 559]
    voxel_pcts = [100, 1.0, 32.0, 17.8, 11.6, 8.6, 5.2, 2.8]
    labels = ['原始点云\n20000 点', '级联处理流\n~200 质心',
              '体素 0.05m', '体素 0.10m', '体素 0.15m',
              '体素 0.20m', '体素 0.30m', '体素 0.50m']
    colors = [C_GRAY, C_GREEN] + [C_BLUE] * 6

    fig, ax = plt.subplots(figsize=(10, 5), dpi=200)
    bars = ax.bar(range(8), voxel_pts, width=0.6, color=colors, edgecolor='white', lw=1.5)

    for bar, c, p in zip(bars, voxel_pts, voxel_pcts):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.02,
                f'{c:,}\n({p}%)', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xticks(range(8))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('点数', fontsize=11)
    ax.set_yscale('log')
    style_ax(ax, grid_axis='y')

    savefig(fig, OUT / "voxel_comparison.png")
    print(f"[OK] voxel_comparison.png")


def load_cascade_metrics(batch_dir: Path, label_filtered: bool = True) -> tuple:
    """从 batch_XX/results.csv 读取级联处理流的均值指标。

    级联流经过 YOLO 标签精化，默认使用 person 过滤后指标。
    """
    import csv
    csv_path = batch_dir / "results.csv"
    if not csv_path.exists():
        print(f"  [WARN] {csv_path} 不存在，使用默认值")
        return 58.9, 79.2, 0.676
    prefix = 'person' if label_filtered else 'spatial'
    precisions, recalls, f1s = [], [], []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            precisions.append(float(row[f'{prefix}_precision']))
            recalls.append(float(row[f'{prefix}_recall']))
            f1s.append(float(row[f'{prefix}_f1']))
    return (np.mean(precisions), np.mean(recalls), np.mean(f1s))


def load_voxel_metrics(summary_path: Path) -> list:
    """从 experiment_summary.json 读取体素策略指标。"""
    if not summary_path.exists():
        print(f"  [WARN] {summary_path} 不存在，使用默认值")
        return [(72.3, 59.4, 0.652), (55.9, 56.1, 0.560), (27.6, 53.5, 0.364)]

    with open(summary_path) as f:
        data = json.load(f)

    # 从 list of dicts 中按 name 查找体素策略
    name_map = {
        'voxel_005': '体素 0.05m',
        'voxel_010': '体素 0.10m',
        'voxel_015': '体素 0.15m',
    }
    results = []
    entries = data if isinstance(data, list) else data.get('results', [])
    for key, label in name_map.items():
        match = next((x for x in entries if x.get('name') == key), None)
        if match is None:
            print(f"  [WARN] {key}({label}) 不在 summary 中")
            continue
        p = match.get('precision', 0)
        r = match.get('recall', 0)
        f = match.get('f1', 0)
        results.append((p, r, f))
    return results


def plot_voxel_vs_cascade():
    """体素下采样 vs 级联处理流检测质量对比。

    数据来源：
      - output/experiment_summary.json（体素类策略，408帧，空间匹配）
      - output/batch_40/results.csv（级联处理流 40次运行均值，经 YOLO 行人标签过滤）
    """
    batch_dir = Path("output/batch_40")
    summary_path = Path("output/experiment_summary.json")

    voxel_data = load_voxel_metrics(summary_path)
    cascade_p, cascade_r, cascade_f1 = load_cascade_metrics(batch_dir, label_filtered=True)

    # 补齐到 3 个体素（不足时补默认值）
    defaults = [(72.3, 59.4, 0.652), (55.9, 56.1, 0.560), (27.6, 53.5, 0.364)]
    for i in range(3):
        if i >= len(voxel_data):
            voxel_data.append(defaults[i])

    methods = ['体素 0.05m', '体素 0.10m', '体素 0.15m', '级联处理流\n(地面+墙体+剪叶聚类)']
    precision = [v[0] for v in voxel_data] + [cascade_p]
    recall    = [v[1] for v in voxel_data] + [cascade_r]
    f1_scores = [v[2] for v in voxel_data] + [cascade_f1]

    print(f"  级联: P={cascade_p:.1f}%, R={cascade_r:.1f}%, F1={cascade_f1:.3f}  "
          f"(来自 {batch_dir.name})")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=200,
                                    gridspec_kw={'width_ratios': [1, 0.7]})

    # ── 左图：Precision / Recall 分组柱状图 ──
    x = np.arange(len(methods))
    w = 0.30
    ax1.bar(x - w, precision, w, color=C_BLUE, edgecolor='white', label='Precision')
    ax1.bar(x,      recall,    w, color=C_ORANGE, edgecolor='white', label='Recall')
    for i in range(len(methods)):
        ax1.text(x[i] - w, precision[i] + 1, f'{precision[i]:.0f}%',
                 ha='center', va='bottom', fontsize=9, fontweight='bold', color=C_BLUE)
        ax1.text(x[i], recall[i] + 1, f'{recall[i]:.0f}%',
                 ha='center', va='bottom', fontsize=9, fontweight='bold', color=C_ORANGE)

    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, fontsize=9)
    ax1.set_ylabel('百分比 (%)', fontsize=11)
    ax1.set_title('Precision / Recall 对比', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10, loc='lower left')
    ax1.set_ylim(0, 95)
    style_ax(ax1, grid_axis='y')

    # ── 右图：F1 柱状图 ──
    colors_f1 = [C_BLUE, C_BLUE, C_BLUE, C_GREEN]
    bars = ax2.bar(x, f1_scores, 0.5, color=colors_f1, edgecolor='white', lw=1.5)
    for bar, v in zip(bars, f1_scores):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
                 f'{v:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, fontsize=9)
    ax2.set_ylabel('F1 Score', fontsize=11)
    ax2.set_title('F1 Score 对比', fontsize=13, fontweight='bold')
    ax2.set_ylim(0, 0.85)
    style_ax(ax2, grid_axis='y')

    fig.tight_layout()
    savefig(fig, OUT / "voxel_vs_cascade.png")
    print(f"[OK] voxel_vs_cascade.png")


def storyboard():
    """视频分镜脚本（中文）。"""
    sb = """
============================================
视频分镜方案: "挑战与优化之路"
============================================

[Scene 1: 问题展示] ~30s
  画面: 原始 LiDAR 点云 (20000 pts/frame)
  旁白: "室内移动机器人每秒产生20帧点云，
        每帧约20000点。直接聚类需要计算
        数亿次距离——在嵌入式平台上不可行。"
  素材: pipeline_flow.png Phase 1

[Scene 2: 复杂度困境] ~30s
  画面: 复杂度曲线图
  旁白: "O(n²)的暴力DBSCAN在20000点上需要
        约8秒/帧。使用KD-tree优化后仍需
        约120ms——仍高于20Hz的50ms上限。"
  素材: complexity_curve.png

[Scene 3: 级联处理流] ~40s
  画面: 逐级压缩漏斗图
  旁白: "我们设计了三阶段级联处理流：
        ① 地面过滤去除25%点
        ② BevEDLines墙体检测去除32%点
        ③ 剪叶聚类保留密集区域质心，仅约200个
        整体压缩约100倍。"
  素材: pipeline_flow.png

[Scene 4: BevEDLines] ~30s
  画面: BEV密度梯度图 + 线段检测示意
  旁白: "BevEDLines将点云投影为BEV密度图，
        用Sobel梯度+Edge Drawing链式追踪
        检测墙体线段，加上三维几何验证
        抑制误检。比LSD快约25%。"

[Scene 5: 剪叶聚类] ~30s
  画面: 四叉树自适应分区示意
  旁白: "剪叶聚类用四叉树替代均匀体素网格。
        近处行人→细密树叶保留精度；
        远处行人→大范围树叶保 recall。
        四个步骤：建树→剔除稀疏叶→密集叶质心→DBSCAN。"
  素材: pruning_effect.png

[Scene 6: 实测性能] ~30s
  画面: 速度对比柱状图 + 管线分解
  旁白: "Rust实现下，整个点云管线仅需39ms，
        远超20Hz实时要求(50ms)。剪叶聚类
        仅16.8ms，与LV-DOT相当但精度更高。"
  素材: timing_breakdown.png + speed_comparison.png

[总结] ~10s
  关键数据:
  - 原始点云: ~20000 pts → 质心: ~200 pts (≈100x压缩)
  - 剪叶聚类: ~17ms (Rust)
  - 全管线: ~39ms → 满足20Hz实时导航
  - 精度: P=82.7%, R=68.6%, F1=0.750
============================================
"""
    with open(OUT / "storyboard.txt", 'w', encoding='utf-8') as f:
        f.write(sb)
    print(f"[OK] storyboard.txt")


if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    print("生成视频素材中...")
    plot_complexity()
    plot_pipeline()
    plot_timing()
    plot_speed()
    plot_pruning()
    plot_voxel_comparison()
    plot_voxel_vs_cascade()
    storyboard()
    print(f"\n所有素材已保存至: {OUT.resolve()}")
