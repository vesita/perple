"""
评估结果可视化 — 生成适合论文使用的图表

用法:
  python scripts/eval_viz.py              # 使用内置数据
  python scripts/eval_viz.py <stats.json> # 从 JSON 读取

输出: output/eval_viz/ 目录下的 PNG 图片
"""

import json, os, shutil, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ─── 中文字体设置 ───────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["SimHei", "Microsoft YaHei", "SimSun"],
    "axes.unicode_minus": False,
})

OUT_DIR = "output/eval_viz"

# ─── 5 号字体系 ────────────────────────────────────────────────────────────
# 5号 = 10.5pt, 小五 = 9pt, 四号 = 14pt
FONT_TITLE = 14       # 图标题（四号）
FONT_LABEL = 10.5    # 轴标签（五号）
FONT_TICK = 9         # 刻度数字（小五）
FONT_LEGEND = 9       # 图例（小五）
FONT_CAPTION = 9      # 图注文字（小五）

# 图片尺寸：A4 文本区宽度 ≈ 15cm，按 300dpi ≈ 1772px
# 这里用英寸：15cm ≈ 5.9in
FIG_W = 5.9
FIG_H = 3.8


def load_data(path: str | None = None) -> dict:
    """加载评估数据，支持 JSON 或内置默认值"""
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    # 默认数据（来自一次 eval_pipeline 运行，trick 模块生效）
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


def plot_recall_bar(data: dict):
    """图1：五折帧级召回率柱状图"""
    folds = data["folds"]
    labels = [f"Fold {f['id']}\n({f['file_range']})" for f in folds]
    recalls = [f["recall_pct"] for f in folds]
    avg_val = data["avg_recall"]

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    # 柱状图
    x = np.arange(len(labels))
    colors = ["#2ecc71" if r >= 70 else "#e74c3c" for r in recalls]
    bars = ax.bar(x, recalls, width=0.55, color=colors, edgecolor="white", linewidth=0.5,
                  zorder=3)

    # 柱顶数值标签
    for bar, r in zip(bars, recalls):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{r}%", ha="center", va="bottom", fontsize=FONT_TICK,
                fontweight="bold")

    # 平均值虚线
    ax.axhline(y=avg_val, color="#2980b9", linestyle="--", linewidth=1.2,
               label=f"平均值 {avg_val}%", zorder=4)
    ax.legend(fontsize=FONT_LEGEND, loc="lower right")

    # 轴设置
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=FONT_TICK)
    ax.set_ylabel("帧级召回率 (%)", fontsize=FONT_LABEL)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(20))
    ax.tick_params(axis="y", labelsize=FONT_TICK)
    ax.grid(axis="y", alpha=0.3, zorder=0)

    # 移除上右边框
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "recall_bar.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}  (300dpi, {FIG_W}\"×{FIG_H}\")")


def plot_recall_with_annotation(data: dict):
    """图2：带标注的召回率图（含簇均值和行人均辅助信息）"""
    folds = data["folds"]
    labels = [f"Fold {f['id']}" for f in folds]
    recalls = [f["recall_pct"] for f in folds]
    avg_clusters = [f["avg_clusters"] for f in folds]
    avg_person = [f["avg_person"] for f in folds]
    avg_val = data["avg_recall"]

    fig, ax1 = plt.subplots(figsize=(FIG_W, FIG_H))

    x = np.arange(len(labels))
    width = 0.35

    # 柱状图：帧级召回率
    colors = ["#2ecc71" if r >= 70 else "#e74c3c" for r in recalls]
    bars = ax1.bar(x - width/2, recalls, width, color=colors, edgecolor="white",
                   linewidth=0.5, label="帧级召回率", zorder=3)

    # 柱顶数值
    for bar, r in zip(bars, recalls):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{r}%", ha="center", va="bottom", fontsize=FONT_TICK,
                fontweight="bold")

    ax1.axhline(y=avg_val, color="#2980b9", linestyle="--", linewidth=1.0,
                label=f"均值 {avg_val}%", zorder=4)
    ax1.set_ylim(0, 100)
    ax1.set_ylabel("帧级召回率 (%)", fontsize=FONT_LABEL)
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(20))

    # 第二轴：行人均
    ax2 = ax1.twinx()
    ax2.plot(x + width/2, avg_person, "D-", color="#e67e22", linewidth=1.5,
             markersize=5, label="帧均行人", zorder=5)
    ax2.set_ylabel("帧均行人数", fontsize=FONT_LABEL)
    ax2.tick_params(axis="y", labelsize=FONT_TICK)
    ax2.set_ylim(0, max(avg_person) * 1.6)

    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=FONT_LEGEND,
               loc="upper right")

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=FONT_TICK)
    ax1.tick_params(axis="x", labelsize=FONT_TICK)
    ax1.grid(axis="y", alpha=0.3, zorder=0)
    ax1.spines["top"].set_visible(False)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "recall_with_person.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")


def plot_detailed_table(data: dict):
    """图3：结果汇总表（以图片形式输出）"""
    folds = data["folds"]

    # 准备表格数据
    col_labels = ["测试集", "文件范围", "帧数", "簇均值", "行人均", "有人帧"]
    rows = []
    for f in folds:
        rows.append([
            f"Fold {f['id']}",
            f["file_range"],
            str(f["frames"]),
            f"{f['avg_clusters']:.1f}",
            f"{f['avg_person']:.1f}",
            f"{f['recall_pct']}%",
        ])
    # 平均值行
    rows.append([
        "平均",
        f"{folds[0]['file_range'].split('-')[0]}-{folds[-1]['file_range'].split('-')[1]}",
        str(data["fold_size"]),
        f"{np.mean([f['avg_clusters'] for f in folds]):.1f}",
        f"{np.mean([f['avg_person'] for f in folds]):.1f}",
        f"{data['avg_recall']:.1f}%",
    ])

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H * 0.65))
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )

    # 表格样式
    table.auto_set_font_size(False)
    table.set_fontsize(FONT_TICK)
    table.scale(1.0, 1.5)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold", fontsize=FONT_TICK)
        elif row == len(rows):  # 平均行
            cell.set_facecolor("#ecf0f1")
            cell.set_text_props(fontweight="bold", fontsize=FONT_TICK)
        else:
            cell.set_facecolor("white" if row % 2 == 1 else "#f9f9f9")
            cell.set_text_props(fontsize=FONT_TICK)
        cell.set_edgecolor("#bdc3c7")
        cell.set_linewidth(0.5)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "summary_table.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")


def main():
    data_path = sys.argv[1] if len(sys.argv) > 1 else None
    data = load_data(data_path)

    # 清理旧数据
    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"评估数据：{data['n_folds']} 折，平均召回率 {data['avg_recall']}%")
    print(f"输出目录：{OUT_DIR}/\n")

    print("正在生成图表...")
    plot_recall_bar(data)
    plot_recall_with_annotation(data)
    plot_detailed_table(data)

    print(f"\n完成！共 3 张图，300dpi，适合直接插入 Word。")
    print(f"图片说明（图注）：")
    print(f"  图1 五折帧级召回率柱状图 — 测试集 {data['total_frames']} 帧"
          f"×{data['n_folds']} 折，均值 {data['avg_recall']}%")
    print(f"  图2 召回率与帧均行人数 — 双轴对比，辅助分析召回率变化原因")
    print(f"  表1 五折评估结果汇总 — 完整统计指标")


if __name__ == "__main__":
    main()
