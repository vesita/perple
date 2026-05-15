"""绘制 PR 曲线 + F1 随阈值变化图（论文用）

用法:
    .venv/Scripts/python.exe scripts/viz_pr_curve.py output/pr_curve_xxx/pr_curve.json
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei"],
    "font.family": "sans-serif",
    "axes.unicode_minus": False,
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def find_latest_json() -> Path:
    output_dir = Path("output")
    dirs = sorted(output_dir.glob("pr_curve_*"), reverse=True)
    for d in dirs:
        j = d / "pr_curve.json"
        if j.exists():
            return j
    raise FileNotFoundError("未找到 pr_curve.json")


def load_data(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def plot_pr_curve(points_person, points_all, output_path: Path):
    """PR 曲线：对比 person 过滤 vs 全部检测"""
    def prep(points):
        recalls = [p["recall"] for p in points]
        precisions = [p["precision"] for p in points]
        # AP
        ap = 0.0
        sorted_idx = np.argsort(recalls)
        sorted_rec = np.array([recalls[i] for i in sorted_idx])
        sorted_prec = np.array([precisions[i] for i in sorted_idx])
        for i in range(len(sorted_rec)):
            if i == 0:
                ap += sorted_prec[i] * sorted_rec[i]
            else:
                ap += sorted_prec[i] * (sorted_rec[i] - sorted_rec[i - 1])
        return recalls, precisions, ap

    fig, ax = plt.subplots(figsize=(7, 5.5))

    # Person 过滤曲线
    recalls_p, precisions_p, ap_p = prep(points_person)
    line_p, = ax.plot(recalls_p, precisions_p, "-o", color="#457B9D", linewidth=1.8,
                      markersize=5, markerfacecolor="white", markeredgecolor="#457B9D",
                      markeredgewidth=1.2, zorder=3, label=f"Person 过滤 (AP={ap_p:.3f})")

    # 全部检测曲线
    if points_all:
        recalls_a, precisions_a, ap_a = prep(points_all)
        line_a, = ax.plot(recalls_a, precisions_a, "-s", color="#E63946", linewidth=1.8,
                          markersize=5, markerfacecolor="white", markeredgecolor="#E63946",
                          markeredgewidth=1.2, zorder=3, label=f"全部检测 (AP={ap_a:.3f})")

    # 标注阈值点（person 曲线）
    label_indices = [0, 4, 8, 12, 16, 19]
    for idx in label_indices:
        if idx < len(points_person):
            p = points_person[idx]
            ax.annotate(
                f"{p['threshold']:.2f}m",
                (p["recall"], p["precision"]),
                fontsize=7, color="#1D3557",
                ha="left" if idx < 12 else "right",
                va="bottom",
                xytext=(4, 4), textcoords="offset points",
            )

    ax.plot([0, 1], [0, 1], "--", color="#ccc", linewidth=0.8, label="随机基准")

    ax.set_xlabel("Recall (召回率)")
    ax.set_ylabel("Precision (精确率)")
    ax.set_title("Precision-Recall 曲线对比", fontweight="bold", pad=10)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  [OK] PR 曲线对比 → {output_path}")


def plot_f1_curve(points_person, points_all, output_path: Path):
    """F1 随中心距阈值变化（对比 person 过滤 vs 全部检测）"""
    def prep(points):
        thr = [p["threshold"] for p in points]
        f1 = [p["f1"] for p in points]
        pr = [p["precision"] for p in points]
        re = [p["recall"] for p in points]
        return thr, f1, pr, re

    thr_p, f1_p, pr_p, re_p = prep(points_person)
    thr_a, f1_a, pr_a, re_a = prep(points_all) if points_all else (None, None, None, None)

    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Person 过滤 F1
    ax1.plot(thr_p, f1_p, "-s", color="#457B9D", linewidth=2,
             markersize=4, markerfacecolor="white", markeredgecolor="#457B9D",
             markeredgewidth=1.2, zorder=4, label="F1 (Person 过滤)")
    ax1.fill_between(thr_p, f1_p, alpha=0.08, color="#457B9D")

    # 全部检测 F1
    if thr_a:
        ax1.plot(thr_a, f1_a, "-s", color="#E63946", linewidth=2,
                 markersize=4, markerfacecolor="white", markeredgecolor="#E63946",
                 markeredgewidth=1.2, zorder=4, label="F1 (全部检测)")
        ax1.fill_between(thr_a, f1_a, alpha=0.08, color="#E63946")

    # 最佳 F1 标记（person）
    best_idx = int(np.argmax(f1_p))
    best_thresh = thr_p[best_idx]
    best_f1 = f1_p[best_idx]
    ax1.axvline(x=best_thresh, color="#457B9D", linestyle=":", alpha=0.4, linewidth=0.8)
    ax1.annotate(
        f"Person 过滤: 最佳 F1={best_f1:.4f} @ {best_thresh:.2f}m",
        xy=(best_thresh, best_f1),
        fontsize=8, color="#457B9D", fontweight="bold",
        xytext=(15, -20), textcoords="offset points",
        arrowprops=dict(arrowstyle="->", color="#457B9D", alpha=0.6),
    )

    # 最佳 F1 标记（全部）
    if thr_a:
        best_idx_a = int(np.argmax(f1_a))
        best_thresh_a = thr_a[best_idx_a]
        best_f1_a = f1_a[best_idx_a]
        ax1.axvline(x=best_thresh_a, color="#E63946", linestyle=":", alpha=0.4, linewidth=0.8)
        ax1.annotate(
            f"全部检测: 最佳 F1={best_f1_a:.4f} @ {best_thresh_a:.2f}m",
            xy=(best_thresh_a, best_f1_a),
            fontsize=8, color="#E63946", fontweight="bold",
            xytext=(15, 15), textcoords="offset points",
            arrowprops=dict(arrowstyle="->", color="#E63946", alpha=0.6),
        )

    ax1.set_xlabel("中心距离阈值 (m)")
    ax1.set_ylabel("F1 Score")
    ax1.set_xlim(0.05, 1.0)
    ax1.set_ylim(0, 1.0)
    ax1.grid(True, alpha=0.3)

    # Precision / Recall 曲线（person 过滤）
    ax2 = ax1.twinx()
    ax2.plot(thr_p, pr_p, "--^", color="#457B9D", linewidth=1.0,
             markersize=3, alpha=0.5, label="Precision (Person 过滤)")
    ax2.plot(thr_p, re_p, "--v", color="#2A9D8F", linewidth=1.0,
             markersize=3, alpha=0.5, label="Recall (Person 过滤)")
    if thr_a:
        ax2.plot(thr_a, pr_a, "--^", color="#E63946", linewidth=1.0,
                 markersize=3, alpha=0.4, label="Precision (全部检测)")
        ax2.plot(thr_a, re_a, "--v", color="#E69D00", linewidth=1.0,
                 markersize=3, alpha=0.4, label="Recall (全部检测)")
    ax2.set_ylabel("Precision / Recall")
    ax2.set_ylim(0, 1.0)

    lines, labels = ax1.get_legend_handles_labels()
    l2, l2l = ax2.get_legend_handles_labels()
    ax1.legend(lines + l2, labels + l2l, loc="lower left", fontsize=8)

    ax1.set_title("F1 Score 随匹配阈值变化 (对比)", fontweight="bold", pad=10)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
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
    print(f"  {data['n_frames']} 帧, {len(points)} 个阈值")
    print(f"  管线耗时: {data['pipeline_elapsed_s']:.1f}s")

    # 最佳 F1
    best = max(points, key=lambda p: p["f1"])
    print(f"\n[Person 过滤] 最佳 F1: {best['f1']:.4f} @ 阈值 {best['threshold']:.2f}m")
    print(f"  P={best['precision']*100:.1f}%  R={best['recall']*100:.1f}%")
    if points_all:
        best_a = max(points_all, key=lambda p: p["f1"])
        print(f"\n[全部检测] 最佳 F1: {best_a['f1']:.4f} @ 阈值 {best_a['threshold']:.2f}m")
        print(f"  P={best_a['precision']*100:.1f}%  R={best_a['recall']*100:.1f}%")
    print()

    plot_pr_curve(points, points_all, out_dir / "fig_pr_curve.png")
    plot_f1_curve(points, points_all, out_dir / "fig_f1_vs_threshold.png")

    print(f"\n图表已保存至: {out_dir.resolve()}")
    print(f"  fig_pr_curve.png")
    print(f"  fig_f1_vs_threshold.png")


if __name__ == "__main__":
    main()
