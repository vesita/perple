"""PruneQt vs LV-DOT 消融实验对比图（统一风格）

用法:
    .venv/Scripts/python.exe scripts/ablation_charts.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from scripts.chart_style import savefig, style_ax, bar_labels, SIZES
from scripts.chart_style import C_BLUE, C_RED

BAR_WIDTH = 0.3
COLORS = [C_BLUE, C_RED]
LABELS = ['剪叶聚类（本文）', 'LV-DOT']
OUT_DIR = 'output/ablation_5run/charts'

# 数据（5 轮均值）
PERSON = dict(p=[84.6, 90.2], r=[68.5, 51.1], f=[75.7, 65.2])
ALL    = dict(p=[58.8, 62.5], r=[78.8, 58.7], f=[67.4, 60.6])
TP_FP_FN = dict(tp=[964, 719], fp=[675, 432], fn=[260, 505])


def draw_grouped_bar(title, metrics, y_max=100, suffixes=('%', '%', '%'), ylabel=''):
    """一组两策略三指标分组柱状图"""
    fig, ax = plt.subplots(figsize=SIZES["single_bar"])
    x = np.arange(3)
    for i in range(2):
        vals = [metrics[k][i] for k in ('p', 'r', 'f')]
        ax.bar(x + i * BAR_WIDTH, vals, BAR_WIDTH,
               color=COLORS[i], label=LABELS[i],
               edgecolor='white', linewidth=0.3)

    ax.set_xticks(x + BAR_WIDTH / 2)
    ax.set_xticklabels(title, fontsize=10)
    ax.set_ylim(0, y_max)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10)
    ax.legend(fontsize=9, framealpha=0.85, edgecolor='#cccccc')
    style_ax(ax)

    for i in range(2):
        for j, key in enumerate(('p', 'r', 'f')):
            v = metrics[key][i]
            suf = suffixes[j]
            ax.text(x[j] + i * BAR_WIDTH, v + y_max * 0.02,
                    f'{v:.1f}{suf}', ha='center', va='bottom',
                    fontsize=7.5, fontweight='bold', color=COLORS[i])

    return fig


# ── 图 1: Person 过滤 ──
fig = draw_grouped_bar(['精确率 P', '召回率 R', 'F1'], PERSON)
savefig(fig, f'{OUT_DIR}/person_metrics.png')
print('OK person_metrics.png')

# ── 图 2: 全部类别 ──
fig = draw_grouped_bar(['精确率 P', '召回率 R', 'F1'], ALL)
savefig(fig, f'{OUT_DIR}/all_metrics.png')
print('OK all_metrics.png')

# ── 图 3: TP / FP / FN ──
fig, ax = plt.subplots(figsize=SIZES["single_bar"])
x2 = np.arange(3)
labels = ['TP（正确）', 'FP（误检）', 'FN（漏检）']
for i in range(2):
    vals = [TP_FP_FN[k][i] for k in ('tp', 'fp', 'fn')]
    ax.bar(x2 + i * BAR_WIDTH, vals, BAR_WIDTH,
           color=COLORS[i], label=LABELS[i],
           edgecolor='white', linewidth=0.3)

ax.set_xticks(x2 + BAR_WIDTH / 2)
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel('数量', fontsize=10)
ax.legend(fontsize=9, framealpha=0.85, edgecolor='#cccccc')
style_ax(ax)

for i in range(2):
    for j, k in enumerate(('tp', 'fp', 'fn')):
        v = TP_FP_FN[k][i]
        ax.text(x2[j] + i * BAR_WIDTH, v + 15, f'{v}',
                ha='center', va='bottom', fontsize=7.5,
                fontweight='bold', color=COLORS[i])
savefig(fig, f'{OUT_DIR}/tp_fp_fn.png')
print('OK tp_fp_fn.png')

print(f'\n完成 - 3 张图表已保存至 {OUT_DIR}')
