# -*- coding: utf-8 -*-
"""精确率/召回率/F1 折线图（含标准差带 + 底部汇总表）

用法:
    .venv/Scripts/python.exe scripts/plot_accuracy_combined.py
"""
import csv, sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.chart_style import savefig

sys.stdout.reconfigure(encoding='utf-8')

CSV_PATH = Path("output/batch_40/results.csv")
OUTPUT_DIR = Path("output/batch_40")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with open(CSV_PATH) as f:
    rows = list(csv.DictReader(f))

runs = np.array([int(r['run_id']) for r in rows])
precisions = np.array([float(r['person_precision']) for r in rows])
recalls    = np.array([float(r['person_recall']) for r in rows])
f1_scores  = np.array([float(r['person_f1']) * 100 for r in rows])

fig = plt.figure(figsize=(12, 7))
gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.15)
ax = fig.add_subplot(gs[0])

series = [
    ('精确率', precisions, '#2ca02c', 'o'),
    ('召回率', recalls,    '#1f77b4', 's'),
    ('F1', f1_scores, '#d62728', '^'),
]

for name, vals, color, marker in series:
    mean = np.mean(vals)
    std  = np.std(vals)
    ax.plot(runs, vals, color=color, marker=marker, markersize=5,
            linewidth=1.2, label=f'{name}  μ={mean:.1f}%  σ={std:.2f}')
    ax.axhline(mean, color=color, linestyle='--', linewidth=0.8, alpha=0.5)
    ax.fill_between(runs, mean - std, mean + std, color=color, alpha=0.08)

ax.set_xlabel('运行次数', fontsize=14)
ax.set_ylabel('指标 (%)', fontsize=14)
ax.tick_params(labelsize=12)
ax.set_xlim(0.5, max(runs) + 0.5)
ax.set_ylim(55, 95)
ax.grid(True, alpha=0.25)
ax.legend(fontsize=12, loc='lower left')

ax_table = fig.add_subplot(gs[1])
ax_table.axis('off')

stats_data = [
    ('精确率 (%)', precisions),
    ('召回率 (%)', recalls),
    ('F1 (%)', f1_scores),
]
cell_data = []
for label, vals in stats_data:
    cell_data.append([label,
                      f'{np.mean(vals):.1f}', f'{np.std(vals):.2f}',
                      f'{np.min(vals):.1f}', f'{np.max(vals):.1f}',
                      f'{np.max(vals) - np.min(vals):.1f}'])

col_labels = ['指标', '均值', '标准差', '最小值', '最大值', '极差']
table = ax_table.table(cellText=cell_data, colLabels=col_labels,
                       loc='center', cellLoc='center',
                       colWidths=[0.18, 0.12, 0.12, 0.12, 0.12, 0.12])
table.auto_set_font_size(False)
table.set_fontsize(13)
table.scale(1, 2.2)
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_facecolor('#40466e')
        cell.set_text_props(color='white', fontweight='bold')
    elif row % 2 == 0:
        cell.set_facecolor('#f5f5f5')

savefig(fig, OUTPUT_DIR / 'fig_accuracy_combined.png')
print(f"  OK {OUTPUT_DIR / 'fig_accuracy_combined.png'}")
