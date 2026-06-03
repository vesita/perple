"""EDLines 标注对比可视化 — P/R/F1 精度图 + 速度图

用法:
    .venv/Scripts/python.exe scripts/edlines_labeled_viz.py
    .venv/Scripts/python.exe scripts/edlines_labeled_viz.py <labeled_results.json>

输出: output/edlines_bench/ 下的 PNG 图片
"""
import json, os, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.chart_style import savefig, style_ax, C_BLUE, C_RED, C_GREEN, C_GRAY, C_DARK, C_CYAN, SIZES

OUT_DIR = Path("output/edlines_bench")
FONT_LABEL = 10.5
FONT_TICK = 9
FONT_LEGEND = 9


def load_data(path=None):
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    default_path = OUT_DIR / "labeled_results.json"
    if default_path.exists():
        with open(default_path, "r", encoding="utf-8") as f:
            return json.load(f)
    raise FileNotFoundError(f"找不到标注结果文件: {path or default_path}")


def _prf1(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp > 0 else 0.0
    r = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
    return p, r, f1

def prf1_chart(data):
    """P/R/F1 双策略对比柱状图"""
    bev = data["bev_total"]
    ref = data["ref_total"]

    bev_p, bev_r, bev_f1 = _prf1(bev["tp"], bev["fp"], bev["fn"])
    ref_p, ref_r, ref_f1 = _prf1(ref["tp"], ref["fp"], ref["fn"])

    metrics = ["精确率", "召回率", "F1"]
    bev_vals = [bev_p * 100, bev_r * 100, bev_f1 * 100]
    ref_vals = [ref_p * 100, ref_r * 100, ref_f1 * 100]

    x = np.arange(len(metrics))
    width = 0.32

    fig, ax = plt.subplots(figsize=(SIZES["single_bar"]))
    bars1 = ax.bar(x - width / 2, bev_vals, width, label="BevEdLines", color=C_BLUE, edgecolor="white")
    bars2 = ax.bar(x + width / 2, ref_vals, width, label="EdLinesRef", color=C_RED, edgecolor="white")

    # 标注数值
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1, f"{h:.1f}",
                ha="center", va="bottom", fontsize=7.5, fontweight="bold", color=C_BLUE)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1, f"{h:.1f}",
                ha="center", va="bottom", fontsize=7.5, fontweight="bold", color=C_RED)

    ax.set_xticks(x)
    ax.set_xticklabels(["精确率 P", "召回率 R", "F1"])
    ax.set_ylabel("百分比 (%)", fontsize=FONT_LABEL)
    ax.legend(fontsize=FONT_LEGEND)
    style_ax(ax)
    ax.set_ylim(0, max(max(bev_vals), max(ref_vals)) * 1.2)

    n = data["config"].get("n_frames", 0)
    fig.tight_layout()
    savefig(fig, OUT_DIR / "labeled_prf1.png")
    print(f"  → labeled_prf1.png (BevEdLines F1={bev_f1:.4f}, EdLinesRef F1={ref_f1:.4f}, n={int(n)}帧)")


def tp_fp_fn_chart(data):
    """TP/FP/FN 累计对比图"""
    bev = data["bev_total"]
    ref = data["ref_total"]

    x = np.arange(3)
    width = 0.32
    labels = ["TP", "FP", "FN"]
    bev_vals = [bev["tp"], bev["fp"], bev["fn"]]
    ref_vals = [ref["tp"], ref["fp"], ref["fn"]]

    fig, ax = plt.subplots(figsize=(SIZES["single_bar"]))
    bars1 = ax.bar(x - width / 2, bev_vals, width, label="BevEdLines", color=C_BLUE, edgecolor="white")
    bars2 = ax.bar(x + width / 2, ref_vals, width, label="EdLinesRef", color=C_RED, edgecolor="white")

    max_val = max(max(bev_vals), max(ref_vals))
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + max_val * 0.02, f"{int(h)}",
                ha="center", va="bottom", fontsize=8, fontweight="bold", color=C_BLUE)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + max_val * 0.02, f"{int(h)}",
                ha="center", va="bottom", fontsize=8, fontweight="bold", color=C_RED)

    ax.set_xticks(x)
    ax.set_xticklabels(["TP", "FP", "FN"])
    ax.set_ylabel("目标数", fontsize=FONT_LABEL)
    ax.legend(fontsize=FONT_LEGEND)
    style_ax(ax)

    n = data["config"].get("n_frames", 0)
    fig.tight_layout()
    savefig(fig, OUT_DIR / "labeled_tpfpfn.png")
    print(f"  → labeled_tpfpfn.png")


def labeled_speed_per_frame_chart(data):
    """标注帧逐帧耗时对比折线图"""
    frames = data["frames"]
    bev_ms = np.array([f["bev_time_ms"] for f in frames])
    ref_ms = np.array([f["ref_time_ms"] for f in frames])
    n = len(bev_ms)
    x = np.arange(n)

    fig, ax = plt.subplots(figsize=(SIZES["dual_axis"]))
    ax.plot(x, bev_ms, color=C_BLUE, linewidth=0.9, alpha=0.8, label="BevEdLines")
    ax.plot(x, ref_ms, color=C_RED, linewidth=0.9, alpha=0.8, label="EdLinesRef")
    ax.fill_between(x, bev_ms, ref_ms, alpha=0.06, color=C_GRAY)
    ax.set_xlabel("帧序号", fontsize=FONT_LABEL)
    ax.set_ylabel("耗时 (ms)", fontsize=FONT_LABEL)
    ax.legend(fontsize=FONT_LEGEND)
    style_ax(ax)

    savefig(fig, OUT_DIR / "labeled_speed_per_frame.png")
    print(f"  → labeled_speed_per_frame.png (BevEdLines μ={bev_ms.mean():.2f}ms, EdLinesRef μ={ref_ms.mean():.2f}ms)")


def labeled_speed_avg_chart(data):
    """标注帧平均耗时对比柱状图"""
    frames = data["frames"]
    bev_ms = np.array([f["bev_time_ms"] for f in frames])
    ref_ms = np.array([f["ref_time_ms"] for f in frames])

    fig, ax = plt.subplots(figsize=(SIZES["single_bar"]))
    means = [bev_ms.mean(), ref_ms.mean()]
    stds = [bev_ms.std(), ref_ms.std()]
    labels = ["BevEdLines", "EdLinesRef"]
    bars = ax.bar(labels, means, yerr=stds, color=[C_BLUE, C_RED], width=0.45, capsize=4)
    for bar, v in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                f"{v:.2f}ms", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax.set_ylabel("平均耗时 (ms)", fontsize=FONT_LABEL)
    style_ax(ax)

    ratio = means[0] / means[1] if means[1] > 0 else 1
    ax.text(0.5, 0.92, f"速度比: {ratio:.2f}x", transform=ax.transAxes,
            ha="center", fontsize=9, color=C_DARK,
            bbox=dict(facecolor="white", edgecolor=C_GRAY, alpha=0.8, boxstyle="round,pad=0.3"))

    savefig(fig, OUT_DIR / "labeled_speed_avg.png")
    print(f"  → labeled_speed_avg.png ({bev_ms.mean():.2f}ms vs {ref_ms.mean():.2f}ms)")


def labeled_per_frame_chart(data):
    """逐帧 TP 对比折线图"""
    frames = data["frames"]
    idx = np.array([f["frame_idx"] for f in frames])
    bev_tp = np.array([f["bev_tp"] for f in frames])
    ref_tp = np.array([f["ref_tp"] for f in frames])

    fig, ax = plt.subplots(figsize=(SIZES["dual_axis"]))
    ax.plot(idx, bev_tp, color=C_BLUE, linewidth=0.8, alpha=0.8, marker="o", markersize=3, label="BevEdLines TP")
    ax.plot(idx, ref_tp, color=C_RED, linewidth=0.8, alpha=0.8, marker="s", markersize=3, label="EdLinesRef TP")
    ax.set_xlabel("帧序号", fontsize=FONT_LABEL)
    ax.set_ylabel("TP", fontsize=FONT_LABEL)
    ax.legend(fontsize=FONT_LEGEND)
    style_ax(ax)

    fig.tight_layout()
    savefig(fig, OUT_DIR / "labeled_per_frame_tp.png")
    print(f"  → labeled_per_frame_tp.png")


def summary_table(data):
    """打印标注对比汇总表"""
    bev = data["bev_total"]
    ref = data["ref_total"]
    cfg = data.get("config", {})

    bev_p, bev_r, bev_f1 = _prf1(bev["tp"], bev["fp"], bev["fn"])
    ref_p, ref_r, ref_f1 = _prf1(ref["tp"], ref["fp"], ref["fn"])

    dp = (bev_p - ref_p) * 100
    dr = (bev_r - ref_r) * 100
    df = (bev_f1 - ref_f1) * 100

    print(f"""
╔═══ EDLines 标注对比汇总 ═══╗
║ 配置:                       ║
║   帧数:          {cfg.get('n_frames', 0):>6.0f}            ║
║   中心距阈值:   {cfg.get('center_dist', 0.5):>6.2f}m           ║
║   最大范围:     {cfg.get('max_range', 10):>6.0f}m           ║
╠═════════════════════════════╣
║ {'指标':<15} {'BevEdLines':>12} {'EdLinesRef':>12} ║
║{'─'*42}║
║ {'Precision':<15} {bev_p*100:>10.1f}% {ref_p*100:>10.1f}% ║
║ {'Recall':<15} {bev_r*100:>10.1f}% {ref_r*100:>10.1f}% ║
║ {'F1 Score':<15} {bev_f1:>10.4f}   {ref_f1:>10.4f}   ║
║{'─'*42}║
║ {'差值(pp)':<15}                           ║
║ {'  Precision':<15} {dp:>+10.2f}pp               ║
║ {'  Recall':<15} {dr:>+10.2f}pp               ║
║ {'  F1':<15} {df:>+10.2f}pp               ║
╚═════════════════════════════╝
""")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else None
    data = load_data(path)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("生成 EDLines 标注对比图表...")
    prf1_chart(data)
    tp_fp_fn_chart(data)
    labeled_speed_per_frame_chart(data)
    labeled_speed_avg_chart(data)
    labeled_per_frame_chart(data)
    summary_table(data)
    print(f"\n所有图表已保存至: {OUT_DIR}/")


if __name__ == "__main__":
    main()
