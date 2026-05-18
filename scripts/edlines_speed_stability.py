"""EDLines 多次运行速度对比折线图 — BevEdLines vs EdLinesRef 10 轮"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.chart_style import savefig, style_ax, C_BLUE, C_RED, C_GRAY, SIZES

OUT_DIR = Path("output/edlines_bench")
FONT_LABEL = 10.5
FONT_LEGEND = 9

# 10 轮速度数据 (ms)
data = {
    "BevEdLines": [3.41, 3.42, 3.59, 3.34, 3.43, 3.37, 3.26, 3.37, 3.47, 3.29],
    "EdLinesRef": [4.40, 4.45, 4.61, 4.48, 4.55, 4.35, 4.39, 4.49, 4.48, 4.29],
}

fig, ax = plt.subplots(figsize=(SIZES["single_bar"]))

x = np.arange(1, 11)
ax.plot(x, data["BevEdLines"], color=C_BLUE, linewidth=1.2, marker="o", markersize=5, label="BevEdLines")
ax.plot(x, data["EdLinesRef"], color=C_RED, linewidth=1.2, marker="s", markersize=5, label="EdLinesRef")

# 均值线
bev_mean = np.mean(data["BevEdLines"])
ref_mean = np.mean(data["EdLinesRef"])
ax.axhline(bev_mean, color=C_BLUE, linestyle="--", linewidth=0.6, alpha=0.5)
ax.axhline(ref_mean, color=C_RED, linestyle="--", linewidth=0.6, alpha=0.5)

# 标注均值
ax.text(10.3, bev_mean, f"{bev_mean:.2f}ms", color=C_BLUE, fontsize=7.5, va="center")
ax.text(10.3, ref_mean, f"{ref_mean:.2f}ms", color=C_RED, fontsize=7.5, va="center")

ax.set_xlabel("运行轮次", fontsize=FONT_LABEL)
ax.set_ylabel("平均墙体提取耗时 (ms)", fontsize=FONT_LABEL)
# 无标题（遵循图表风格指南）
ax.set_xticks(x)
ax.legend(fontsize=FONT_LEGEND)
style_ax(ax)

# 增速比标注
ratio = bev_mean / ref_mean
ax.text(0.5, 0.08, f"BevEdLines / EdLinesRef = {ratio:.2f}x", transform=ax.transAxes,
        ha="center", fontsize=9, color=C_GRAY,
        bbox=dict(facecolor="white", edgecolor=C_GRAY, alpha=0.8, boxstyle="round,pad=0.3"))

fig.tight_layout()
savefig(fig, OUT_DIR / "speed_stability.png")
print(f"  → speed_stability.png")
print(f"  BevEdLines: μ={bev_mean:.3f}ms σ={np.std(data['BevEdLines']):.3f}ms")
print(f"  EdLinesRef: μ={ref_mean:.3f}ms σ={np.std(data['EdLinesRef']):.3f}ms")
print(f"  速度比: {ratio:.2f}x")
