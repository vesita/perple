"""评估结果可视化 — 生成适合论文使用的图表

用法:
    .venv/Scripts/python.exe scripts/eval_viz.py
    .venv/Scripts/python.exe scripts/eval_viz.py <stats.json>

输出: output/eval_viz/ 目录下的 PNG 图片
"""
import json, os, shutil, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.chart_style import savefig, style_ax, FIG_W, C_BLUE, C_RED, C_GREEN, C_ORANGE, C_GRAY, C_DARK

OUT_DIR = Path("output/eval_viz")

FONT_TITLE = 14
FONT_LABEL = 10.5
FONT_TICK = 9
FONT_LEGEND = 9
FIG_H = 3.8


def load_data(path=None):
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "note": "5 折评估结果（--start 200 --end 906 --fold-size 100）",
        "fold_size": 100,
        "n_folds": 5,
        "total_frames": 532,
        "discarded": 32,
        "avg_recall": 73.0,
        "folds": [
            {"id": 1, "file_range": "200-346", "frames": 100,
             "avg_clusters": 3.1, "avg_person": 1.3, "recall_pct": 84},
            {"id": 2, "file_range": "347-473", "frames": 100,
             "avg_clusters": 3.1, "avg_person": 0.7, "recall_pct": 45},
            {"id": 3, "file_range": "474-614", "frames": 100,
             "avg_clusters": 3.4, "avg_person": 1.4, "recall_pct": 82},
            {"id": 4, "file_range": "615-734", "frames": 100,
             "avg_clusters": 3.2, "avg_person": 1.0, "recall_pct": 75},
            {"id": 5, "file_range": "735-854", "frames": 100,
             "avg_clusters": 3.5, "avg_person": 1.3, "recall_pct": 79},
        ],
    }


def plot_recall_bar(data):
    folds = data["folds"]
    labels = [f"第 {f['id']} 折\n({f['file_range']})" for f in folds]
    recalls = [f["recall_pct"] for f in folds]
    avg_val = data["avg_recall"]

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    x = np.arange(len(labels))
    colors_bar = [C_GREEN if r >= 70 else C_RED for r in recalls]
    bars = ax.bar(x, recalls, width=0.55, color=colors_bar,
                  edgecolor="white", linewidth=0.5, zorder=3)
    for bar, r in zip(bars, recalls):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{r}%", ha="center", va="bottom", fontsize=FONT_TICK, fontweight="bold")
    ax.axhline(y=avg_val, color=C_BLUE, linestyle="--", linewidth=1.2,
               label=f"平均值 {avg_val}%", zorder=4)
    ax.legend(fontsize=FONT_LEGEND, loc="lower right")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=FONT_TICK)
    ax.set_ylabel("帧级召回率 (%)", fontsize=FONT_LABEL)
    ax.set_ylim(0, 100)
    ax.tick_params(axis="y", labelsize=FONT_TICK)
    style_ax(ax)
    savefig(fig, OUT_DIR / "recall_bar.png")
    print(f"  → {OUT_DIR / 'recall_bar.png'}")


def plot_recall_with_annotation(data):
    folds = data["folds"]
    labels = [f"第 {f['id']} 折" for f in folds]
    recalls = [f["recall_pct"] for f in folds]
    avg_person = [f["avg_person"] for f in folds]
    avg_val = data["avg_recall"]

    fig, ax1 = plt.subplots(figsize=(FIG_W, FIG_H))
    x = np.arange(len(labels))
    width = 0.35
    colors_bar = [C_GREEN if r >= 70 else C_RED for r in recalls]
    bars = ax1.bar(x - width/2, recalls, width, color=colors_bar,
                   edgecolor="white", linewidth=0.5, label="帧级召回率", zorder=3)
    for bar, r in zip(bars, recalls):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{r}%", ha="center", va="bottom", fontsize=FONT_TICK, fontweight="bold")
    ax1.axhline(y=avg_val, color=C_BLUE, linestyle="--", linewidth=1.0,
                label=f"均值 {avg_val}%", zorder=4)
    ax1.set_ylim(0, 100)
    ax1.set_ylabel("帧级召回率 (%)", fontsize=FONT_LABEL)

    ax2 = ax1.twinx()
    ax2.plot(x + width/2, avg_person, "D-", color=C_ORANGE, linewidth=1.5,
             markersize=5, label="帧均行人", zorder=5)
    ax2.set_ylabel("帧均行人数", fontsize=FONT_LABEL)
    ax2.tick_params(axis="y", labelsize=FONT_TICK)
    ax2.set_ylim(0, max(avg_person) * 1.6)

    l1, lb1 = ax1.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lb1 + lb2, fontsize=FONT_LEGEND, loc="upper right")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=FONT_TICK)
    ax1.grid(axis="y", alpha=0.3)
    ax1.spines["top"].set_visible(False)

    savefig(fig, OUT_DIR / "recall_with_person.png")
    print(f"  → {OUT_DIR / 'recall_with_person.png'}")


def plot_detailed_table(data):
    folds = data["folds"]
    col_labels = ["测试集", "文件范围", "帧数", "簇均值", "行人均", "有人帧"]
    rows = []
    for f in folds:
        rows.append([f"第 {f['id']} 折", f["file_range"], str(f["frames"]),
                     f"{f['avg_clusters']:.1f}", f"{f['avg_person']:.1f}", f"{f['recall_pct']}%"])
    rows.append(["平均",
                 f"{folds[0]['file_range'].split('-')[0]}-{folds[-1]['file_range'].split('-')[1]}",
                 str(data["fold_size"]),
                 f"{np.mean([f['avg_clusters'] for f in folds]):.1f}",
                 f"{np.mean([f['avg_person'] for f in folds]):.1f}",
                 f"{data['avg_recall']:.1f}%"])

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H * 0.65))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=col_labels, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(FONT_TICK)
    table.scale(1.0, 1.5)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor(C_DARK)
            cell.set_text_props(color="white", fontweight="bold", fontsize=FONT_TICK)
        elif row == len(rows):
            cell.set_facecolor("#ecf0f1")
            cell.set_text_props(fontweight="bold", fontsize=FONT_TICK)
        else:
            cell.set_facecolor("white" if row % 2 == 1 else "#f9f9f9")
            cell.set_text_props(fontsize=FONT_TICK)
        cell.set_edgecolor("#bdc3c7")
        cell.set_linewidth(0.5)
    savefig(fig, OUT_DIR / "summary_table.png")
    print(f"  → {OUT_DIR / 'summary_table.png'}")


def main():
    data_path = sys.argv[1] if len(sys.argv) > 1 else None
    data = load_data(data_path)

    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"评估数据：{data['n_folds']} 折，平均召回率 {data['avg_recall']}%")
    print(f"输出目录：{OUT_DIR}/\n")

    plot_recall_bar(data)
    plot_recall_with_annotation(data)
    plot_detailed_table(data)
    print(f"\n完成！3 张图")


if __name__ == "__main__":
    main()
