"""论文图表生成脚本 — 室内移动机器人行人感知方法研究

用法:
    .venv/Scripts/python.exe scripts/thesis_viz.py

输出: output/thesis_figures/ 目录下的 PNG 图片（300dpi）
"""
import json, os, re, sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.chart_style import (
    savefig, style_ax, auto_find_latest,
    C_BLUE, C_RED, C_GREEN, C_YELLOW, C_ORANGE, C_GRAY, C_DARK, C_CYAN,
    COLORS_10,
)

OUT_DIR = Path("output/thesis_figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_W = 5.9


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def auto_find_eval_results():
    results = []
    dirs = sorted(Path("output").glob("eval_ablation_*"), reverse=True)
    ts_dirs = [d for d in dirs if re.match(r'^eval_ablation_\d+$', d.name)]
    for d in ts_dirs:
        j = d / "eval_result.json"
        if j.exists():
            data = load_json(j)
            if data.get("config", {}).get("cluster_strategy") == "prune_qt" \
               and data.get("n_frames", 0) >= 408:
                results.append(data)
        if len(results) >= 5:
            break
    return results


def auto_find_pr_curve():
    dirs = sorted(Path("output").glob("pr_curve_*"), reverse=True)
    for d in dirs:
        j = d / "pr_curve.json"
        if j.exists():
            return load_json(j)
    return None


def auto_find_pipeline_jsonl():
    for pattern in ["pipeline_*", "thesis_*"]:
        dirs = sorted(Path("output").glob(pattern), reverse=True)
        for d in dirs:
            j = d / "pipeline.jsonl"
            if j.exists():
                return j
    return None


# ════════════════════════════════════════════════════════════════════════════
#  图1: 主指标柱状图
# ════════════════════════════════════════════════════════════════════════════

def plot_main_metrics(results):
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8))
    for idx, (mode, key, label) in enumerate([
        ("Person 过滤", "person_only", "(a)"),
        ("全部类别", "all_classes", "(b)"),
    ]):
        ax = axes[idx]
        precisions = [r[key]["precision"] * 100 for r in results]
        recalls = [r[key]["recall"] * 100 for r in results]
        f1s = [r[key]["f1"] for r in results]

        means = [np.mean(precisions), np.mean(recalls), np.mean(f1s)]
        stds  = [np.std(precisions), np.std(recalls), np.std(f1s)]
        colors = [C_BLUE, C_GREEN, C_RED]

        bars = ax.bar(["精确率", "召回率", "F1"], means, yerr=stds,
                      capsize=4, color=colors, edgecolor="white", linewidth=0.5,
                      width=0.5, zorder=3,
                      error_kw={"linewidth": 1.2, "ecolor": C_DARK})

        for bar, m, s in zip(bars, means, stds):
            suf = "%" if m > 0 else ""
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                    f"{m:.1f}{suf}±{s:.1f}", ha="center", va="bottom",
                    fontsize=7, fontweight="bold", color=C_DARK)
        ax.set_ylim(0, 110)
        ax.set_ylabel("百分比 (%)")
        ax.text(0.02, 0.98, label, transform=ax.transAxes,
                fontsize=10, fontweight="bold", va="top")
        style_ax(ax)

    fig.tight_layout()
    savefig(fig, OUT_DIR / "fig01_main_metrics.png")
    print(f"  [OK] 图1 → {OUT_DIR / 'fig01_main_metrics.png'}")


# ════════════════════════════════════════════════════════════════════════════
#  图2: 策略对比
# ════════════════════════════════════════════════════════════════════════════

def plot_strategy_comparison():
    strategies = [
        ("剪叶聚类",   79.3, 58.6, 0.674),
        ("xy_grid_dbscan", 60.1, 59.6, 0.598),
        ("dbscan_qt",  61.4, 56.7, 0.589),
        ("lvdot",      74.9, 47.3, 0.580),
        ("cc",         51.7, 61.2, 0.560),
        ("dbscan_grid", 50.3, 60.6, 0.550),
        ("range_image", 67.7, 34.6, 0.458),
        ("seq",        70.3, 10.0, 0.176),
    ]
    names = [s[0] for s in strategies]
    precisions = [s[1] for s in strategies]
    recalls = [s[2] for s in strategies]
    f1s = [s[3] for s in strategies]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(names))
    width = 0.25

    bars_p = ax.bar(x - width, precisions, width, label="精确率",
                    color=C_BLUE, edgecolor="white", linewidth=0.3, zorder=3)
    bars_r = ax.bar(x, recalls, width, label="召回率",
                    color=C_GREEN, edgecolor="white", linewidth=0.3, zorder=3)
    bars_f = ax.bar(x + width, [f * 100 for f in f1s], width, label="F1×100",
                    color=C_RED, edgecolor="white", linewidth=0.3, zorder=3)

    for bars_list in [bars_p, bars_r, bars_f]:
        bars_list[0].set_edgecolor(C_DARK)
        bars_list[0].set_linewidth(1.5)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=7.5)
    ax.set_ylabel("百分比 (%)")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(0, 100)
    style_ax(ax)

    fig.tight_layout()
    savefig(fig, OUT_DIR / "fig02_strategy_comparison.png")
    print(f"  [OK] 图2 → {OUT_DIR / 'fig02_strategy_comparison.png'}")


# ════════════════════════════════════════════════════════════════════════════
#  图3: PR 曲线
# ════════════════════════════════════════════════════════════════════════════

def plot_pr_curve(pr_data):
    points = pr_data["points"]
    points_all = pr_data.get("points_all", None)

    fig, ax = plt.subplots(figsize=(6, 5))

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

    r_p, p_p, ap_p = prep(points)
    ax.plot(r_p, p_p, "-o", color=C_BLUE, linewidth=2, markersize=5,
            markerfacecolor="white", markeredgecolor=C_BLUE,
            markeredgewidth=1.2, zorder=3,
            label=f"Person 过滤 (AP={ap_p:.3f})")

    if points_all:
        r_a, p_a, ap_a = prep(points_all)
        ax.plot(r_a, p_a, "-s", color=C_RED, linewidth=2, markersize=5,
                markerfacecolor="white", markeredgecolor=C_RED,
                markeredgewidth=1.2, zorder=3,
                label=f"全部检测 (AP={ap_a:.3f})")

    step = max(1, len(points) // 6)
    for i in range(0, len(points), step):
        p = points[i]
        ax.annotate(f"{p['threshold']:.2f}m", (p["recall"], p["precision"]),
                    fontsize=6.5, color=C_DARK,
                    ha="left" if i < len(points)//2 else "right",
                    va="bottom", xytext=(3, 3), textcoords="offset points")

    ax.plot([0, 1], [0, 1], "--", color="#ccc", linewidth=0.8, label="随机基准")

    best = max(points, key=lambda p: p["f1"])
    ax.plot(best["recall"], best["precision"], "*", color=C_ORANGE,
            markersize=12, zorder=5, markeredgecolor="white", markeredgewidth=0.8)
    ax.annotate(f"最佳 F1={best['f1']:.3f}\n@ {best['threshold']:.2f}m",
                (best["recall"], best["precision"]), fontsize=7, color=C_ORANGE,
                fontweight="bold", xytext=(8, 8), textcoords="offset points",
                arrowprops=dict(arrowstyle="->", color=C_ORANGE, alpha=0.6))

    ax.set_xlabel("Recall (召回率)")
    ax.set_ylabel("Precision (精确率)")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower left", fontsize=8)

    fig.tight_layout()
    savefig(fig, OUT_DIR / "fig03_pr_curve.png")
    print(f"  [OK] 图3 → {OUT_DIR / 'fig03_pr_curve.png'}")


# ════════════════════════════════════════════════════════════════════════════
#  图4: F1 随阈值变化
# ════════════════════════════════════════════════════════════════════════════

def plot_f1_threshold(pr_data):
    points = pr_data["points"]
    points_all = pr_data.get("points_all", None)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    for idx, (pts, color) in enumerate([
        (points, C_BLUE),
        (points_all, C_RED) if points_all else (None, None),
    ]):
        if pts is None:
            continue
        ax = ax1 if idx == 0 else ax2
        thr = [p["threshold"] for p in pts]
        f1  = [p["f1"] for p in pts]
        pr  = [p["precision"] for p in pts]
        re  = [p["recall"] for p in pts]

        ax.plot(thr, f1, "-s", color=color, linewidth=2, markersize=4,
                label="F1", zorder=4)
        ax.plot(thr, pr, "--^", color=color, linewidth=1, markersize=3,
                alpha=0.6, label="精确率")
        ax.plot(thr, re, "--v", color=C_GREEN, linewidth=1, markersize=3,
                alpha=0.6, label="召回率")

        bi = int(np.argmax(f1))
        ax.axvline(x=thr[bi], color=color, linestyle=":", alpha=0.4)
        ax.annotate(f"F1={f1[bi]:.3f}\n@{thr[bi]:.2f}m",
                    xy=(thr[bi], f1[bi]), fontsize=7, color=color,
                    fontweight="bold", xytext=(10, -25),
                    textcoords="offset points",
                    arrowprops=dict(arrowstyle="->", color=color, alpha=0.5))

        ax.set_xlabel("中心距阈值 (m)")
        ax.set_xlim(0.05, 1.0)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7, loc="lower left")
        style_ax(ax)
        ax.text(0.02, 0.98, f"({'a' if idx == 0 else 'b'})",
                transform=ax.transAxes, fontsize=10, fontweight="bold", va="top")

    fig.tight_layout()
    savefig(fig, OUT_DIR / "fig04_f1_threshold.png")
    print(f"  [OK] 图4 → {OUT_DIR / 'fig04_f1_threshold.png'}")


# ════════════════════════════════════════════════════════════════════════════
#  图5: 分类分析
# ════════════════════════════════════════════════════════════════════════════

def plot_classification_analysis():
    categories = ["正确分类\n为 person", "误分类为\nobstacle", "完全漏检\n(FN)"]
    values_pct = [56.5, 13.9, 29.6]
    colors = [C_GREEN, C_ORANGE, C_RED]

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    bars = ax.bar(categories, values_pct, color=colors, edgecolor="white",
                  linewidth=0.8, width=0.5, zorder=3)
    for bar, v in zip(bars, values_pct):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{v:.1f}%", ha="center", va="bottom", fontsize=10,
                fontweight="bold", color=C_DARK)

    counts = [707, 170, 347]
    for bar, c in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 0.35,
                f"n={c}", ha="center", va="center", fontsize=7.5,
                color="white", fontweight="bold")

    ax.set_ylabel("占全部 GT 比例 (%)")
    ax.set_ylim(0, 80)
    style_ax(ax)

    fig.tight_layout()
    savefig(fig, OUT_DIR / "fig05_classification_analysis.png")
    print(f"  [OK] 图5 → {OUT_DIR / 'fig05_classification_analysis.png'}")


# ════════════════════════════════════════════════════════════════════════════
#  图6: 消融实验
# ════════════════════════════════════════════════════════════════════════════

def plot_ablation_comparison():
    configs = [
        "wd=0.08\nmin_occ=4\neps=0.20",
        "wd=0.08\nmin_occ=4\neps=0.30",
        "wd=0.05\nmin_occ=4\neps=0.20",
        "wd=0.08\nmin_occ=3\neps=0.30",
        "wd=0.05\nmin_occ=3\neps=0.30",
    ]
    f1s      = [0.674, 0.647, 0.646, 0.674, 0.659]
    precisions = [79.3, 72.5, 73.4, 78.4, 71.0]
    recalls   = [58.6, 58.5, 57.7, 59.1, 61.5]
    fps      = [187, 272, 256, 199, 307]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    x = np.arange(len(configs))
    width = 0.3

    bars = ax1.bar(x, f1s, width, color=C_BLUE, edgecolor="white", zorder=3)
    ax1.set_xticks(x)
    ax1.set_xticklabels(configs, fontsize=6.5, ha="center")
    ax1.set_ylabel("F1 Score")
    ax1.set_ylim(0.5, 0.75)
    style_ax(ax1)
    for bar, v in zip(bars, f1s):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=7,
                fontweight="bold", color=C_DARK)

    ax1b = ax1.twinx()
    ax1b.plot(x, fps, "D-", color=C_RED, linewidth=1.5, markersize=5, zorder=5)
    ax1b.set_ylabel("FP 数量", color=C_RED)
    ax1b.tick_params(axis="y", colors=C_RED)
    for i, fp in enumerate(fps):
        ax1b.annotate(str(fp), (x[i], fp), fontsize=6.5, color=C_RED,
                     ha="center", va="bottom")

    ax2.bar(x - width/2, precisions, width, color=C_BLUE, edgecolor="white",
            label="精确率", zorder=3)
    ax2.bar(x + width/2, recalls, width, color=C_GREEN, edgecolor="white",
            label="召回率", zorder=3)
    ax2.set_xticks(x)
    ax2.set_xticklabels(configs, fontsize=6.5, ha="center")
    ax2.set_ylabel("百分比 (%)")
    ax2.set_ylim(50, 85)
    ax2.legend(fontsize=7)
    style_ax(ax2)

    fig.tight_layout()
    savefig(fig, OUT_DIR / "fig06_ablation.png")
    print(f"  [OK] 图6 → {OUT_DIR / 'fig06_ablation.png'}")


# ════════════════════════════════════════════════════════════════════════════
#  图7: 管线延迟
# ════════════════════════════════════════════════════════════════════════════

def plot_latency_from_jsonl(jsonl_path):
    frames = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                frames.append(json.loads(line))
    if not frames:
        print("  [SKIP] pipeline.jsonl 无数据")
        return

    n_frames = max(f["frame"] for f in frames) + 1
    fi = [f["frame"] for f in frames]
    total = np.array([f["elapsed_ms"] for f in frames])
    stages = ["join", "fuse", "io", "tracker"]
    sl = ["点云处理", "特征融合", "I/O", "目标跟踪"]
    sc = [C_CYAN, C_BLUE, C_DARK, C_RED]
    sd = {s: np.array([f["stages_ms"][s] for f in frames]) for s in stages}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    ax1.stackplot(fi, [sd[s] for s in stages], labels=sl, colors=sc, alpha=0.85)
    ax1.plot(fi, total, "-", color="black", linewidth=0.8, alpha=0.5, label="总耗时")
    ax1.axhline(y=np.mean(total), color="gray", linestyle="--", alpha=0.5,
                label=f"均值 {np.mean(total):.0f}ms")
    ax1.set_xlabel("帧号")
    ax1.set_ylabel("耗时 (ms)")
    ax1.legend(fontsize=7, loc="upper left")
    ax1.set_xlim(0, n_frames)
    ax1.set_ylim(bottom=0)
    ax1.grid(axis="y", alpha=0.2)

    ax2.hist(total, bins=40, color=C_BLUE, edgecolor="white", linewidth=0.3,
             alpha=0.8, zorder=3)
    ax2.axvline(x=np.mean(total), color=C_RED, linestyle="--", linewidth=1.5,
                label=f"均值 {np.mean(total):.1f}ms")
    ax2.axvline(x=np.median(total), color=C_GREEN, linestyle=":", linewidth=1.5,
                label=f"中位数 {np.median(total):.1f}ms")
    ax2.set_xlabel("每帧总耗时 (ms)")
    ax2.set_ylabel("帧数")
    ax2.legend(fontsize=7)
    style_ax(ax2)

    fig.tight_layout()
    savefig(fig, OUT_DIR / "fig07_latency.png")
    print(f"  [OK] 图7 → {OUT_DIR / 'fig07_latency.png'}")
    print(f"    平均延迟: {np.mean(total):.1f}ms ± {np.std(total):.1f}")
    print(f"    等效 FPS: {1000/np.mean(total):.1f}")


# ════════════════════════════════════════════════════════════════════════════
#  图8: 检测统计
# ════════════════════════════════════════════════════════════════════════════

def plot_detection_stats_from_jsonl(jsonl_path):
    frames = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                frames.append(json.loads(line))
    if not frames:
        return

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    fi = [f["frame"] for f in frames]
    nt = [f["stats"]["n_targets"] for f in frames]
    np_ = [f["stats"]["n_person"] for f in frames]

    ax = axes[0]
    ax.plot(fi, nt, "-", color=C_BLUE, linewidth=0.7, label="总目标", alpha=0.8)
    ax.plot(fi, np_, "-", color=C_RED, linewidth=0.7, label="行人", alpha=0.8)
    ax.fill_between(fi, np_, alpha=0.12, color=C_RED)
    ax.set_xlabel("帧号")
    ax.set_ylabel("目标数")
    ax.legend(fontsize=7)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.2)

    ax = axes[1]
    nc = [f["stats"]["n_clusters"] for f in frames]
    ax.plot(fi, nc, "-", color=C_GREEN, linewidth=0.7, alpha=0.8)
    ax.fill_between(fi, nc, alpha=0.12, color=C_GREEN)
    ax.set_xlabel("帧号")
    ax.set_ylabel("聚类数")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.2)

    ax = axes[2]
    cats = ["moving", "static", "movable", "floating"]
    labels = ["运动中", "静止", "可移动", "漂浮"]
    colors = [C_RED, C_BLUE, C_GREEN, C_YELLOW]
    bottom = np.zeros(len(fi))
    for cat, color, label in zip(cats, colors, labels):
        values = np.array([f["stats"].get(f"n_{cat}", 0) for f in frames])
        if values.sum() > 0:
            ax.bar(fi, values, bottom=bottom, width=1.0,
                   color=color, label=label, alpha=0.8, edgecolor="none")
            bottom += values
    ax.set_xlabel("帧号")
    ax.set_ylabel("目标数")
    ax.legend(fontsize=6.5, loc="upper left")
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    savefig(fig, OUT_DIR / "fig08_detection_stats.png")
    print(f"  [OK] 图8 → {OUT_DIR / 'fig08_detection_stats.png'}")


# ════════════════════════════════════════════════════════════════════════════
#  图9: 管线演化
# ════════════════════════════════════════════════════════════════════════════

def plot_pipeline_evolution():
    eras = ["Era 1a\n原始全量", "Era 1b\n+降采样", "Era 2\n+去地面", "Era 3\n+去墙体"]
    points = [20006, 20006, 15779, 1484]
    latency = [10096, 41, 29, 29]
    persons = [12.1, 12.5, 12.9, 6.6]

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    ax = axes[0]
    colors = [C_RED if p == max(points) else C_BLUE for p in points]
    bars = ax.bar(eras, points, color=colors, edgecolor="white", width=0.5, zorder=3)
    ax.set_ylabel("输入点量")
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars, points):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.05,
                f"{v}", ha="center", va="bottom", fontsize=6.5, fontweight="bold")

    ax = axes[1]
    colors = [C_RED if l == max(latency) else C_BLUE for l in latency]
    bars = ax.bar(eras, latency, color=colors, edgecolor="white", width=0.5, zorder=3)
    ax.set_ylabel("每帧耗时 (ms)")
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars, latency):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.05,
                f"{v}ms", ha="center", va="bottom", fontsize=6.5, fontweight="bold")

    ax = axes[2]
    ax.plot(eras, persons, "-o", color=C_RED, linewidth=1.5, markersize=6, label="人检测数")
    ax.set_ylabel("人检测数（帧均）")
    ax.grid(alpha=0.3)
    for i, v in enumerate(persons):
        ax.annotate(f"{v:.1f}", (eras[i], v), fontsize=7,
                   ha="center", va="bottom", fontweight="bold")

    fig.tight_layout()
    savefig(fig, OUT_DIR / "fig09_pipeline_evolution.png")
    print(f"  [OK] 图9 → {OUT_DIR / 'fig09_pipeline_evolution.png'}")


# ════════════════════════════════════════════════════════════════════════════
#  图10: 多轮稳定性
# ════════════════════════════════════════════════════════════════════════════

def plot_stability_analysis(results):
    if len(results) < 2:
        return
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.5))
    runs = list(range(1, len(results) + 1))

    for idx, (key, color) in enumerate([
        ("person_only", C_BLUE), ("all_classes", C_RED),
    ]):
        ax = axes[idx]
        precisions = [r[key]["precision"] * 100 for r in results]
        recalls = [r[key]["recall"] * 100 for r in results]
        f1s = [r[key]["f1"] * 100 for r in results]

        ax.plot(runs, precisions, "-o", color=color, markersize=6, label="精确率")
        ax.plot(runs, recalls, "-s", color=C_GREEN, markersize=6, label="召回率")
        ax.plot(runs, f1s, "-^", color=C_ORANGE, markersize=6, label="F1")
        ax.set_xticks(runs)
        ax.set_xlabel("运行轮次")
        ax.set_ylabel("百分比 (%)")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        style_ax(ax)

    ax = axes[2]
    detections = [r[key]["n_detections"] for r in results]
    fps = [r["person_only"]["fp"] for r in results]
    tps = [r["person_only"]["tp"] for r in results]
    ax.plot(runs, detections, "-o", color=C_BLUE, markersize=6, label="总检测数")
    ax.plot(runs, fps, "-s", color=C_RED, markersize=6, label="FP")
    ax.plot(runs, tps, "-^", color=C_GREEN, markersize=6, label="TP")
    ax.set_xticks(runs)
    ax.set_xlabel("运行轮次")
    ax.set_ylabel("目标数")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    style_ax(ax)

    fig.tight_layout()
    savefig(fig, OUT_DIR / "fig10_stability.png")
    print(f"  [OK] 图10 → {OUT_DIR / 'fig10_stability.png'}")


# ════════════════════════════════════════════════════════════════════════════
#  主入口
# ════════════════════════════════════════════════════════════════════════════

def print_summary_table(results):
    print("\n" + "=" * 65)
    print("  论文实验数据汇总")
    print("=" * 65)
    for mode, key in [("Person 过滤", "person_only"), ("全部类别", "all_classes")]:
        prec = [r[key]["precision"] * 100 for r in results]
        rec  = [r[key]["recall"] * 100 for r in results]
        f1   = [r[key]["f1"] for r in results]
        print(f"\n  ── {mode} ──")
        print(f"    Precision: {np.mean(prec):.1f}% ± {np.std(prec):.1f}")
        print(f"    Recall:    {np.mean(rec):.1f}% ± {np.std(rec):.1f}")
        print(f"    F1:        {np.mean(f1):.3f} ± {np.std(f1):.4f}")
        print(f"    TP: {np.mean([r[key]['tp'] for r in results]):.0f} ± {np.std([r[key]['tp'] for r in results]):.0f}")
        print(f"    FP: {np.mean([r[key]['fp'] for r in results]):.0f} ± {np.std([r[key]['fp'] for r in results]):.0f}")
        print(f"    FN: {np.mean([r[key]['fn'] for r in results]):.0f} ± {np.std([r[key]['fn'] for r in results]):.0f}")


def main():
    results = auto_find_eval_results()
    if len(results) < 1:
        print("[ERR] 未找到 eval_ablation 结果")
        return

    print(f"找到 {len(results)} 轮验证结果")
    print_summary_table(results)

    pr_data = auto_find_pr_curve()
    if pr_data is None:
        print("[WARN] 未找到 pr_curve.json")

    jsonl_path = auto_find_pipeline_jsonl()

    print("\n生成论文图表...")
    plot_main_metrics(results)
    plot_strategy_comparison()
    plot_classification_analysis()
    plot_ablation_comparison()
    plot_stability_analysis(results)

    if pr_data:
        plot_pr_curve(pr_data)
        plot_f1_threshold(pr_data)
    else:
        print("  [SKIP] 图3/4 — 需要 pr_curve.json")

    if jsonl_path:
        print(f"\n  加载 pipeline.jsonl: {jsonl_path}")
        plot_latency_from_jsonl(jsonl_path)
        plot_detection_stats_from_jsonl(jsonl_path)
    else:
        print("  [SKIP] 图7/8 — 需要 pipeline.jsonl")

    plot_pipeline_evolution()

    print(f"\n完成！所有图表保存至: {OUT_DIR.resolve()}")
    for p in sorted(OUT_DIR.glob("fig*.png")):
        print(f"  {p.name}  ({p.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
