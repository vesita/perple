"""
Video assets generator — Benchmark visualization for the "difficulty & optimization" video.

Produces:
  1. complexity_curve.png   — O(n²) vs O(n log n) vs O(k log k)
  2. pipeline_flow.png       — Point cloud compression cascade
  3. timing_breakdown.png    — Rust pipeline sub-step timings
  4. speed_comparison.png    — End-to-end speed comparison
  5. pruning_effect.png      — Quadtree pruning threshold analysis
  6. storyboard.txt          — Video storyboard in Chinese
"""

import json
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Use English font for all plots
plt.rcParams['font.family'] = 'DejaVu Sans'

OUT = Path("output/bench_video_v2")
OUT.mkdir(parents=True, exist_ok=True)

# Rust reference data
RUST = {
    'points_per_frame': 15540,
    'ground_pct': 21,
    'wall_pct': 53,
    'timing_ms': {
        'Ground\n(PeakScan)': 0.6,
        'Wall\n(BevEDLines)': 19.5,
        'Clustering\n(PruneQt)': 16.8,
        'YOLO Refine': 2.2,
        'Pipeline Total': 39.1,
    },
    'comparison': {
        'Brute-force\nDBSCAN': 8000.0,     # O(n²) estimate
        'KD-tree\nDBSCAN': 120.0,           # sklearn estimate
        'LV-DOT\n(Rust)': 16.9,
        'PruneQt\n(Rust)': 16.8,
    }
}

# Color scheme
C_RAW = '#E74C3C'
C_LV = '#F39C12'
C_QT = '#27AE60'
C_BLUE = '#3498DB'
C_PURPLE = '#9B59B6'
C_GRAY = '#7F8C8D'


def plot_complexity():
    """Theoretical complexity curves."""
    n = np.logspace(2, 4.3, 300)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    ax.plot(n, n**2, color=C_RAW, lw=2.5, label='O(n²) Brute-force')
    ax.plot(n, n * np.log2(n), color=C_LV, lw=2.5, label='O(n log n) KD-tree')
    ax.plot(n, (n/30) * np.log2(n/30), color=C_QT, lw=2.5, label='O(k log k) PruneQt (k=n/30)')

    ax.axvline(14000, color='gray', ls='--', alpha=0.5)
    ax.text(14500, 10, '14K pts/frame', fontsize=10, color='gray', rotation=90)

    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('Point Count (n)', fontsize=12)
    ax.set_ylabel('Relative Computation Cost', fontsize=12)
    ax.set_title('Algorithmic Complexity Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
    ax.set_xlim(100, 20000)

    fig.savefig(OUT / "complexity_curve.png", dpi=150, bbox_inches='tight')
    print(f"[OK] complexity_curve.png")
    plt.close(fig)


def plot_pipeline():
    """Cascade compression funnel."""
    stages = ['Raw LiDAR\n15540 pts', 'After Ground\n12315 pts', 'After Wall\n4091 pts', 'After PruneQt\n~500 pts']
    counts = [15540, 12315, 4091, 500]
    colors = [C_GRAY, C_BLUE, '#E67E22', C_QT]
    pcts = [100, 79, 26, 3.2]

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    bars = ax.bar(range(4), counts, width=0.55, color=colors, edgecolor='white', lw=2)

    for bar, c, p in zip(bars, counts, pcts):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+200,
                f'{c:,} pts ({p}%)', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_xticks(range(4))
    ax.set_xticklabels(stages, fontsize=10)
    ax.set_ylabel('Point Count', fontsize=12)
    ax.set_title('Point Cloud Cascade Pipeline: 30x Compression', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    fig.savefig(OUT / "pipeline_flow.png", dpi=150, bbox_inches='tight')
    print(f"[OK] pipeline_flow.png")
    plt.close(fig)


def plot_timing():
    """Rust pipeline timing breakdown."""
    labels = list(RUST['timing_ms'].keys())
    times = list(RUST['timing_ms'].values())
    colors = [C_BLUE, '#E67E22', C_QT, C_PURPLE, C_RAW]

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    bars = ax.barh(labels, times, color=colors, edgecolor='white', height=0.5)

    for bar, v in zip(bars, times):
        ax.text(bar.get_width()+0.3, bar.get_y()+bar.get_height()/2,
                f'{v:.1f} ms', ha='left', va='center', fontsize=11, fontweight='bold')

    ax.axvline(50, color='red', ls='--', lw=1.5, alpha=0.7, label='50 ms deadline (20 Hz)')
    ax.legend(fontsize=11, loc='lower right')
    ax.set_xlabel('Time (ms)', fontsize=12)
    ax.set_title('Rust Pipeline Timing Breakdown', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    fig.savefig(OUT / "timing_breakdown.png", dpi=150, bbox_inches='tight')
    print(f"[OK] timing_breakdown.png")
    plt.close(fig)


def plot_speed():
    """Speed comparison bar chart."""
    cats = list(RUST['comparison'].keys())
    vals = list(RUST['comparison'].values())
    colors = [C_RAW, C_LV, C_BLUE, C_QT]

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    bars = ax.bar(cats, vals, color=colors, edgecolor='white', width=0.5, lw=1.5)

    for bar, v in zip(bars, vals):
        label = f'{v:.0f} ms' if v >= 1 else f'{v:.1f} ms'
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.02,
                label, ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.axhline(50, color='red', ls='--', lw=1.5, alpha=0.7, label='50 ms (20 Hz threshold)')
    ax.legend(fontsize=11)
    ax.set_yscale('log')
    ax.set_ylabel('Time per Frame (ms)', fontsize=12)
    ax.set_title('Algorithm Speed Comparison (Rust / Simulated)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    fig.savefig(OUT / "speed_comparison.png", dpi=150, bbox_inches='tight')
    print(f"[OK] speed_comparison.png")
    plt.close(fig)


def plot_pruning():
    """Pruning threshold analysis."""
    occ = np.array([1, 2, 3, 5, 10, 20, 30, 50])
    centroids = 14000 / occ

    fig, ax1 = plt.subplots(figsize=(10, 5), dpi=150)
    ax2 = ax1.twinx()

    l1 = ax1.plot(occ, centroids, 'o-', color=C_BLUE, lw=2, ms=8, label='Centroids after pruning')
    ax1.set_xlabel('Min points per leaf (min_occ)', fontsize=12)
    ax1.set_ylabel('Number of Centroids', fontsize=12, color=C_BLUE)

    l2 = ax2.plot(occ, centroids * np.log2(centroids) / 100, 's-', color=C_RAW, lw=2, ms=8, label='Estimated DBSCAN cost')
    ax2.set_ylabel('Relative Clustering Cost', fontsize=12, color=C_RAW)

    ax1.axvline(3, color='gray', ls='--', alpha=0.5)
    ax1.text(3.2, ax1.get_ylim()[1]*0.9, 'default\nmin_occ=3', fontsize=10, color='gray')

    lines = l1 + l2
    ax1.legend(lines, [l.get_label() for l in lines], fontsize=11, loc='upper right')
    ax1.set_title('Pruning Threshold vs. Clustering Complexity', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')

    fig.savefig(OUT / "pruning_effect.png", dpi=150, bbox_inches='tight')
    print(f"[OK] pruning_effect.png")
    plt.close(fig)


def storyboard():
    """Video storyboard in Chinese."""
    sb = """
============================================
视频分镜方案: "挑战与优化之路"
============================================

[Scene 1: 问题展示] ~30s
  画面: 原始 LiDAR 点云 (15540 pts/frame)
  旁白: "室内移动机器人每秒产生20帧点云，
        每帧约15000点。直接聚类需要计算
        数亿次距离——在嵌入式平台上不可行。"
  素材: pipeline_flow.png Phase 1

[Scene 2: 复杂度困境] ~30s
  画面: 复杂度曲线图
  旁白: "O(n²)的暴力DBSCAN在14000点上需要
        约8秒/帧。使用KD-tree优化后仍需
        约120ms——仍高于20Hz的50ms上限。"
  素材: complexity_curve.png

[Scene 3: 级联管线] ~40s
  画面: 逐级压缩漏斗图
  旁白: "我们设计了三阶段级联管线：
        ① 地面过滤去除21%点
        ② BevEDLines墙体检测去除53%点
        ③ 剪叶聚类将剩余点压缩30倍
        最终仅需处理约500个质心。"
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
        四个步骤：建树→剪叶→质心→DBSCAN。"
  素材: pruning_effect.png

[Scene 6: 实测性能] ~30s
  画面: 速度对比柱状图 + 管线分解
  旁白: "Rust实现下，整个点云管线仅需39ms，
        远超20Hz实时要求(50ms)。PruneQt
        仅16.8ms，与LV-DOT相当但精度更高。"
  素材: timing_breakdown.png + speed_comparison.png

[总结] ~10s
  关键数据:
  - 原始点云: 15540 pts → 质心: ~500 pts (30x压缩)
  - PruneQt: ~17ms (Rust)
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
    print("Generating video assets...")
    plot_complexity()
    plot_pipeline()
    plot_timing()
    plot_speed()
    plot_pruning()
    storyboard()
    print(f"\nAll assets saved to: {OUT.resolve()}")
