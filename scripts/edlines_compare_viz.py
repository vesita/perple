"""EDLines 对比可视化 — BevEdLines vs EdLinesRef 精度图 + 速度图

用法:
    .venv/Scripts/python.exe scripts/edlines_compare_viz.py
    .venv/Scripts/python.exe scripts/edlines_compare_viz.py <results.json>

输出: output/edlines_bench/ 下的 PNG 图片
"""
import json, os, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.chart_style import savefig, style_ax, C_BLUE, C_RED, C_GREEN, C_GRAY, C_DARK, SIZES

OUT_DIR = Path("output/edlines_bench")
FONT_LABEL = 10.5
FONT_TICK = 9
FONT_LEGEND = 9


def load_results(path=None):
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    default_path = OUT_DIR / "results.json"
    if default_path.exists():
        with open(default_path, "r", encoding="utf-8") as f:
            return json.load(f)
    raise FileNotFoundError(f"找不到结果文件: {path or default_path}")


def speed_per_frame_chart(data):
    """逐帧耗时对比折线图"""
    bev_ms = np.array([r["edlines_ms"] for r in data["bev_edlines"]])
    ref_ms = np.array([r["edlines_ms"] for r in data["edlines_ref"]])
    n = len(bev_ms)
    frames = np.arange(n)

    fig, ax = plt.subplots(figsize=(SIZES["dual_axis"]))
    ax.plot(frames, bev_ms, color=C_BLUE, linewidth=0.9, alpha=0.8, label="BevEdLines")
    ax.plot(frames, ref_ms, color=C_RED, linewidth=0.9, alpha=0.8, label="EdLinesRef")
    ax.fill_between(frames, bev_ms, ref_ms, alpha=0.06, color=C_GRAY)
    ax.set_xlabel("帧序号", fontsize=FONT_LABEL)
    ax.set_ylabel("耗时 (ms)", fontsize=FONT_LABEL)
    ax.legend(fontsize=FONT_LEGEND)
    style_ax(ax)

    savefig(fig, OUT_DIR / "speed_per_frame.png")
    print(f"  → speed_per_frame.png (BevEdLines μ={bev_ms.mean():.2f}ms, EdLinesRef μ={ref_ms.mean():.2f}ms)")


def speed_avg_chart(data):
    """平均耗时对比柱状图"""
    bev_ms = np.array([r["edlines_ms"] for r in data["bev_edlines"]])
    ref_ms = np.array([r["edlines_ms"] for r in data["edlines_ref"]])
    n = len(bev_ms)

    fig, ax = plt.subplots(figsize=(SIZES["single_bar"]))
    colors = [C_BLUE, C_RED]
    labels = ["BevEdLines", "EdLinesRef"]
    means = [bev_ms.mean(), ref_ms.mean()]
    stds = [bev_ms.std(), ref_ms.std()]

    bars = ax.bar(labels, means, yerr=stds, color=colors, width=0.45, capsize=4,
                   error_kw={"linewidth": 1.2, "alpha": 0.6})
    for bar, v in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                f"{v:.2f}ms", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax.set_ylabel("平均耗时 (ms)", fontsize=FONT_LABEL)
    style_ax(ax)

    ratio = means[0] / means[1] if means[1] > 0 else 1
    ax.text(0.5, 0.92, f"速度比: {ratio:.2f}x", transform=ax.transAxes,
            ha="center", fontsize=9, color=C_DARK,
            bbox=dict(facecolor="white", edgecolor=C_GRAY, alpha=0.8, boxstyle="round,pad=0.3"))

    savefig(fig, OUT_DIR / "speed_avg.png")
    print(f"  → speed_avg.png ({bev_ms.mean():.2f}ms vs {ref_ms.mean():.2f}ms, ratio={ratio:.2f}x)")


def wall_points_per_frame_chart(data):
    """逐帧墙壁点对比折线图"""
    bev_pts = np.array([r["n_wall_pts"] for r in data["bev_edlines"]])
    ref_pts = np.array([r["n_wall_pts"] for r in data["edlines_ref"]])
    n = len(bev_pts)
    frames = np.arange(n)

    fig, ax = plt.subplots(figsize=(SIZES["dual_axis"]))
    ax.plot(frames, bev_pts, color=C_BLUE, linewidth=0.9, alpha=0.8, label="BevEdLines")
    ax.plot(frames, ref_pts, color=C_RED, linewidth=0.9, alpha=0.8, label="EdLinesRef")
    ax.fill_between(frames, bev_pts, ref_pts, alpha=0.06, color=C_GRAY)
    ax.set_xlabel("帧序号", fontsize=FONT_LABEL)
    ax.set_ylabel("墙壁点数", fontsize=FONT_LABEL)
    ax.legend(fontsize=FONT_LEGEND)
    style_ax(ax)

    savefig(fig, OUT_DIR / "wall_points_per_frame.png")
    print(f"  → wall_points_per_frame.png")


def wall_points_diff_hist_chart(data):
    """墙壁点差异分布直方图"""
    bev_pts = np.array([r["n_wall_pts"] for r in data["bev_edlines"]])
    ref_pts = np.array([r["n_wall_pts"] for r in data["edlines_ref"]])
    n = len(bev_pts)
    diffs = bev_pts - ref_pts

    fig, ax = plt.subplots(figsize=(SIZES["single_bar"]))
    ax.hist(diffs, bins=min(20, max(5, n // 3)), color=C_GREEN, alpha=0.7, edgecolor="white", linewidth=0.5)
    ax.axvline(0, color=C_RED, linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axvline(diffs.mean(), color=C_DARK, linestyle=":", linewidth=1.0, alpha=0.8)
    ax.set_xlabel("墙壁点差异（BevEdLines − EdLinesRef）", fontsize=FONT_LABEL)
    ax.set_ylabel("帧数", fontsize=FONT_LABEL)
    style_ax(ax)

    savefig(fig, OUT_DIR / "wall_points_diff_hist.png")
    print(f"  → wall_points_diff_hist.png (μ={diffs.mean():.1f}, σ={diffs.std():.1f})")


def wall_points_avg_chart(data):
    """平均墙壁点对比柱状图"""
    bev_pts = np.array([r["n_wall_pts"] for r in data["bev_edlines"]])
    ref_pts = np.array([r["n_wall_pts"] for r in data["edlines_ref"]])

    fig, ax = plt.subplots(figsize=(SIZES["single_bar"]))
    labels = ["BevEdLines", "EdLinesRef"]
    means = [bev_pts.mean(), ref_pts.mean()]
    stds = [bev_pts.std(), ref_pts.std()]
    bars = ax.bar(labels, means, yerr=stds, color=[C_BLUE, C_RED], width=0.45, capsize=4,
                   error_kw={"linewidth": 1.2, "alpha": 0.6})
    for bar, v in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                f"{v:.0f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax.set_ylabel("平均墙壁点数", fontsize=FONT_LABEL)
    style_ax(ax)

    savefig(fig, OUT_DIR / "wall_points_avg.png")
    print(f"  → wall_points_avg.png")


def summary_table(data):
    """汇总表：打印统计摘要"""
    bev = data["bev_edlines"]
    ref = data["edlines_ref"]
    cfg = data.get("config", {})

    bev_ms = np.array([r["edlines_ms"] for r in bev])
    ref_ms = np.array([r["edlines_ms"] for r in ref])
    bev_pts = np.array([r["n_wall_pts"] for r in bev])
    ref_pts = np.array([r["n_wall_pts"] for r in ref])

    print(f"""
╔═══ EDLines 对比汇总 ═══╗
║ 配置:                    ║
║   帧数:          {data['frames_total']:>5}            ║
║   高斯 σ:       {cfg.get('gaussian_sigma', 0.0):>7.2f}               ║
║   锚点阈值:     {cfg.get('anchor_threshold', 0.0):>7.3f}              ║
║   墙壁距离:     {cfg.get('distance', 0.10):>7.2f}               ║
╠══════════════════════════╣
║ {'指标':<15} {'BevEdLines':>12} {'EdLinesRef':>12} ║
║{'─'*42}║
║ {'平均耗时(ms)':<15} {bev_ms.mean():>10.2f}ms {ref_ms.mean():>10.2f}ms ║
║ {'中位耗时(ms)':<15} {np.median(bev_ms):>10.2f}ms {np.median(ref_ms):>10.2f}ms ║
║ {'耗时标准差':<15} {bev_ms.std():>10.2f}ms {ref_ms.std():>10.2f}ms ║
║ {'平均墙壁点':<15} {bev_pts.mean():>10.0f}   {ref_pts.mean():>10.0f}   ║
║ {'墙壁点标准差':<15} {bev_pts.std():>10.0f}   {ref_pts.std():>10.0f}   ║
║{'─'*42}║
║ {'墙壁点差异(μ±σ)':<15} {f'{(bev_pts - ref_pts).mean():.1f} ± {(bev_pts - ref_pts).std():.1f}':>24} ║
║ {'速度比':<15} {f'{bev_ms.mean() / ref_ms.mean():.3f}x':>24} ║
╚══════════════════════════╝
""")


def accuracy_scatter_chart(data):
    """墙壁点一致性散点图"""
    bev = data["bev_edlines"]
    ref = data["edlines_ref"]
    bev_pts = np.array([r["n_wall_pts"] for r in bev])
    ref_pts = np.array([r["n_wall_pts"] for r in ref])

    fig, ax = plt.subplots(figsize=(SIZES["single_bar"]))
    ax.scatter(bev_pts, ref_pts, s=12, alpha=0.6, c=C_BLUE, edgecolors="none")
    min_val = min(bev_pts.min(), ref_pts.min())
    max_val = max(bev_pts.max(), ref_pts.max())
    ax.plot([min_val, max_val], [min_val, max_val], "k--", linewidth=0.7, alpha=0.4, label="y=x（完全一致）")
    ax.set_xlabel("BevEdLines 墙壁点", fontsize=FONT_LABEL)
    ax.set_ylabel("EdLinesRef 墙壁点", fontsize=FONT_LABEL)
    ax.legend(fontsize=FONT_LEGEND)
    style_ax(ax)

    savefig(fig, OUT_DIR / "accuracy_scatter.png")
    print(f"  → accuracy_scatter.png")


def accuracy_planes_chart(data):
    """平均墙壁平面数对比柱状图"""
    bev = data["bev_edlines"]
    ref = data["edlines_ref"]
    bev_planes = np.array([r["n_planes"] for r in bev])
    ref_planes = np.array([r["n_planes"] for r in ref])

    fig, ax = plt.subplots(figsize=(SIZES["single_bar"]))
    bar_width = 0.45
    x = np.arange(2)
    pm = [bev_planes.mean(), ref_planes.mean()]
    ps = [bev_planes.std(), ref_planes.std()]
    ax.bar(x, pm, yerr=ps, width=bar_width, color=[C_BLUE, C_RED], capsize=4,
           tick_label=["BevEdLines", "EdLinesRef"])
    ax.set_ylabel("平均墙壁平面数", fontsize=FONT_LABEL)
    ax.tick_params(labelsize=FONT_TICK)
    style_ax(ax)

    savefig(fig, OUT_DIR / "accuracy_planes.png")
    print(f"  → accuracy_planes.png")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else None
    data = load_results(path)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("生成 EDLines 对比图表...")
    speed_per_frame_chart(data)
    speed_avg_chart(data)
    wall_points_per_frame_chart(data)
    wall_points_diff_hist_chart(data)
    wall_points_avg_chart(data)
    accuracy_scatter_chart(data)
    accuracy_planes_chart(data)
    summary_table(data)
    print(f"\n所有图表已保存至: {OUT_DIR}/")


if __name__ == "__main__":
    main()
