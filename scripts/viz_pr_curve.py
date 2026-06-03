"""绘制 PR 曲线 + F1 随阈值变化图（论文用）

用法:
    .venv/Scripts/python.exe scripts/viz_pr_curve.py output/pr_curve_xxx/pr_curve.json
"""
import json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.chart_style import savefig, style_ax, C_BLUE, C_RED, C_GREEN, C_ORANGE, C_DARK


def find_latest_json():
    output_dir = Path("output")
    dirs = sorted(output_dir.glob("pr_curve_*"), reverse=True)
    for d in dirs:
        j = d / "pr_curve.json"
        if j.exists():
            return j
    raise FileNotFoundError("未找到 pr_curve.json")


def load_data(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def plot_pr_curve(points_person, points_all, output_path):
    def prep(points):
        recalls = [p["recall"] for p in points]
        precisions = [p["precision"] for p in points]
        ap = 0.0
        si = np.argsort(recalls)
        sr = np.array([recalls[i] for i in si])
        sp = np.array([precisions[i] for i in si])
        for i in range(len(sr)):
            ap += sp[i] * (sr[i] - sr[i-1] if i > 0 else sr[i])
        return recalls, precisions, ap

    fig, ax = plt.subplots(figsize=(7, 5.5))

    r_p, p_p, ap_p = prep(points_person)
    ax.plot(r_p, p_p, "-o", color=C_BLUE, linewidth=1.8, markersize=5,
            markerfacecolor="white", markeredgecolor=C_BLUE,
            markeredgewidth=1.2, zorder=3, label=f"行人过滤 (AP={ap_p:.3f})")

    if points_all:
        r_a, p_a, ap_a = prep(points_all)
        ax.plot(r_a, p_a, "-s", color=C_RED, linewidth=1.8, markersize=5,
                markerfacecolor="white", markeredgecolor=C_RED,
                markeredgewidth=1.2, zorder=3, label=f"全部检测 (AP={ap_a:.3f})")

    label_indices = [0, 4, 8, 12, 16, 19]
    for idx in label_indices:
        if idx < len(points_person):
            p = points_person[idx]
            ax.annotate(f"{p['threshold']:.2f}m", (p["recall"], p["precision"]),
                        fontsize=7, color=C_DARK,
                        ha="left" if idx < 12 else "right", va="bottom",
                        xytext=(4, 4), textcoords="offset points")

    ax.plot([0, 1], [0, 1], "--", color="#ccc", linewidth=0.8, label="随机基准")
    ax.set_xlabel("Recall (召回率)")
    ax.set_ylabel("Precision (精确率)")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    savefig(fig, output_path)
    print(f"  [OK] PR 曲线 → {output_path}")


def plot_f1_curve(points_person, points_all, output_path):
    def prep(points):
        thr = [p["threshold"] for p in points]
        f1 = [p["f1"] for p in points]
        pr = [p["precision"] for p in points]
        re = [p["recall"] for p in points]
        return thr, f1, pr, re

    thr_p, f1_p, pr_p, re_p = prep(points_person)
    thr_a, f1_a, pr_a, re_a = prep(points_all) if points_all else (None, None, None, None)

    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.plot(thr_p, f1_p, "-s", color=C_BLUE, linewidth=2, markersize=4,
             markerfacecolor="white", markeredgecolor=C_BLUE,
             markeredgewidth=1.2, zorder=4, label="F1 (行人过滤)")
    ax1.fill_between(thr_p, f1_p, alpha=0.08, color=C_BLUE)
    if thr_a:
        ax1.plot(thr_a, f1_a, "-s", color=C_RED, linewidth=2, markersize=4,
                 markerfacecolor="white", markeredgecolor=C_RED,
                 markeredgewidth=1.2, zorder=4, label="F1 (全部检测)")
        ax1.fill_between(thr_a, f1_a, alpha=0.08, color=C_RED)

    bi = int(np.argmax(f1_p))
    ax1.axvline(x=thr_p[bi], color=C_BLUE, linestyle=":", alpha=0.4, linewidth=0.8)
    ax1.annotate(f"行人: 最佳 F1={f1_p[bi]:.4f} @ {thr_p[bi]:.2f}m",
                 xy=(thr_p[bi], f1_p[bi]), fontsize=8, color=C_BLUE, fontweight="bold",
                 xytext=(15, -20), textcoords="offset points",
                 arrowprops=dict(arrowstyle="->", color=C_BLUE, alpha=0.6))

    if thr_a:
        bi_a = int(np.argmax(f1_a))
        ax1.axvline(x=thr_a[bi_a], color=C_RED, linestyle=":", alpha=0.4, linewidth=0.8)
        ax1.annotate(f"全部: 最佳 F1={f1_a[bi_a]:.4f} @ {thr_a[bi_a]:.2f}m",
                     xy=(thr_a[bi_a], f1_a[bi_a]), fontsize=8, color=C_RED, fontweight="bold",
                     xytext=(15, 15), textcoords="offset points",
                     arrowprops=dict(arrowstyle="->", color=C_RED, alpha=0.6))

    ax1.set_xlabel("中心距离阈值 (m)")
    ax1.set_ylabel("F1 Score")
    ax1.set_xlim(0.05, 1.0)
    ax1.set_ylim(0, 1.0)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(thr_p, pr_p, "--^", color=C_BLUE, linewidth=1.0, markersize=3,
             alpha=0.5, label="精确率 (行人)")
    ax2.plot(thr_p, re_p, "--v", color=C_GREEN, linewidth=1.0, markersize=3,
             alpha=0.5, label="召回率 (行人)")
    if thr_a:
        ax2.plot(thr_a, pr_a, "--^", color=C_RED, linewidth=1.0, markersize=3,
                 alpha=0.4, label="精确率 (全部检测)")
        ax2.plot(thr_a, re_a, "--v", color="#E69D00", linewidth=1.0, markersize=3,
                 alpha=0.4, label="召回率 (全部检测)")
    ax2.set_ylabel("精确率 / 召回率")
    ax2.set_ylim(0, 1.0)

    l1, lb1 = ax1.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lb1 + lb2, loc="lower left", fontsize=8)

    savefig(fig, output_path)
    print(f"  [OK] F1 曲线 → {output_path}")


def main():
    if len(sys.argv) > 1:
        json_path = Path(sys.argv[1])
    else:
        json_path = find_latest_json()
    if not json_path.exists():
        print(f"[ERR] 文件不存在: {json_path}")
        sys.exit(1)

    out_dir = json_path.parent
    data = load_data(json_path)
    points = data["points"]
    points_all = data.get("points_all", None)

    print(f"读取数据: {json_path}")
    best = max(points, key=lambda p: p["f1"])
    print(f"[Person] 最佳 F1: {best['f1']:.4f} @ {best['threshold']:.2f}m")
    if points_all:
        ba = max(points_all, key=lambda p: p["f1"])
        print(f"[全部] 最佳 F1: {ba['f1']:.4f} @ {ba['threshold']:.2f}m")

    plot_pr_curve(points, points_all, out_dir / "fig_pr_curve.png")
    plot_f1_curve(points, points_all, out_dir / "fig_f1_vs_threshold.png")
    print(f"\n图表已保存至: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
