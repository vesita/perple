"""论文图表生成脚本 — 室内移动机器人行人感知方法研究

用法:
    .venv/Scripts/python.exe scripts/thesis_viz.py

输出: output/thesis_figures/ 目录下的 PNG 图片（300dpi，适合插入 Word）
"""

import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ─── 论文级全局样式 ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "SimSun"],
    "font.family": "sans-serif",
    "axes.unicode_minus": False,
    "font.size": 9,
    "axes.labelsize": 10.5,
    "axes.titlesize": 12,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

OUT_DIR = Path("output/thesis_figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── 调色板（学术风格，低饱和度） ──────────────────────────────────────────────
C_BLUE   = "#457B9D"
C_RED    = "#E63946"
C_GREEN  = "#2A9D8F"
C_YELLOW = "#E9C46A"
C_ORANGE = "#F4A261"
C_GRAY   = "#6C757D"
C_DARK   = "#1D3557"
C_CYAN   = "#A8DADC"

COLORS_10 = [
    "#457B9D", "#E63946", "#2A9D8F", "#E9C46A", "#F4A261",
    "#6D597A", "#B56576", "#219EBC", "#023047", "#8ECAE6",
]

# ─── 辅助函数 ────────────────────────────────────────────────────────────────

def find_latest(pattern: str) -> Path:
    """找最新的 output 子目录匹配 pattern 的文件"""
    output_dir = Path("output")
    dirs = sorted(output_dir.glob(pattern), reverse=True)
    for d in dirs:
        return d
    return None


def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def auto_find_eval_results() -> list[dict]:
    """自动查找最近的 prune_qt 408帧 eval_ablation 结果"""
    import re
    results = []
    all_dirs = sorted(Path("output").glob("eval_ablation_*"), reverse=True)
    # 只取纯时间戳目录（eval_ablation_数字）
    ts_dirs = [d for d in all_dirs if re.match(r'^eval_ablation_\d+$', d.name)]
    for d in ts_dirs:
        j = d / "eval_result.json"
        if j.exists():
            data = load_json(j)
            # 只保留 prune_qt + 408帧全量结果
            if data.get("config", {}).get("cluster_strategy") == "prune_qt" \
               and data.get("n_frames", 0) >= 408:
                results.append(data)
        if len(results) >= 5:
            break
    return results


def auto_find_pr_curve() -> dict | None:
    """自动查找最新的 pr_curve.json"""
    dirs = sorted(Path("output").glob("pr_curve_*"), reverse=True)
    for d in dirs:
        j = d / "pr_curve.json"
        if j.exists():
            return load_json(j)
    return None


def auto_find_pipeline_jsonl() -> Path | None:
    """自动查找最新的 pipeline.jsonl"""
    for pattern in ["pipeline_*", "thesis_*"]:
        dirs = sorted(Path("output").glob(pattern), reverse=True)
        for d in dirs:
            j = d / "pipeline.jsonl"
            if j.exists():
                return j
    return None

# ═══════════════════════════════════════════════════════════════════════════════
#  图1: 主指标柱状图 — Precision / Recall / F1（含多轮标准差）
# ═══════════════════════════════════════════════════════════════════════════════

def plot_main_metrics(results: list[dict]):
    """生成主指标对比图：Person 过滤 + 全部类别两组"""

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8))

    for idx, (mode, key, title) in enumerate([
        ("Person 过滤", "person_only", "(a) Person 过滤"),
        ("全部类别", "all_classes", "(b) 全部类别"),
    ]):
        ax = axes[idx]

        # 收集多轮数据
        precisions = [r[key]["precision"] * 100 for r in results]
        recalls = [r[key]["recall"] * 100 for r in results]
        f1s = [r[key]["f1"] for r in results]

        mean_p, std_p = np.mean(precisions), np.std(precisions)
        mean_r, std_r = np.mean(recalls), np.std(recalls)
        mean_f, std_f = np.mean(f1s), np.std(f1s)

        metrics = ["Precision", "Recall", "F1"]
        means = [mean_p, mean_r, mean_f]
        stds  = [std_p, std_r, std_f]
        colors = [C_BLUE, C_GREEN, C_RED]

        bars = ax.bar(metrics, means, yerr=stds, capsize=4, color=colors,
                      edgecolor="white", linewidth=0.5, width=0.5, zorder=3,
                      error_kw={"linewidth": 1.2, "ecolor": C_DARK})

        # 柱顶数值
        for bar, m, s in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                    f"{m:.1f}%±{s:.1f}" if m > 0 else f"{m:.1f}%",
                    ha="center", va="bottom", fontsize=7, fontweight="bold",
                    color=C_DARK)

        ax.set_ylim(0, 110)
        ax.set_title(title, fontweight="bold", pad=8)
        ax.set_ylabel("百分比 (%)")
        ax.grid(axis="y", alpha=0.3, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("检测精度指标（3 轮平均 ± 标准差）", fontweight="bold", y=1.02)
    fig.tight_layout()
    path = OUT_DIR / "fig01_main_metrics.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  [OK] 图1 主指标 → {path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  图2: 策略对比柱状图
# ═══════════════════════════════════════════════════════════════════════════════

def plot_strategy_comparison():
    """绘制聚类策略对比图（从文档数据）"""
    strategies = [
        ("prune_qt",   79.3, 58.6, 0.674),
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

    bars_p = ax.bar(x - width, precisions, width, label="Precision",
                    color=C_BLUE, edgecolor="white", linewidth=0.3, zorder=3)
    bars_r = ax.bar(x, recalls, width, label="Recall",
                    color=C_GREEN, edgecolor="white", linewidth=0.3, zorder=3)
    bars_f = ax.bar(x + width, [f * 100 for f in f1s], width, label="F1×100",
                    color=C_RED, edgecolor="white", linewidth=0.3, zorder=3)

    # 高亮 prune_qt
    for bars_list, val in [(bars_p, precisions[0]), (bars_r, recalls[0]),
                            (bars_f, f1s[0] * 100)]:
        bars_list[0].set_edgecolor("#1D3557")
        bars_list[0].set_linewidth(1.5)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=7.5)
    ax.set_ylabel("百分比 (%)")
    ax.set_title("八种聚类策略性能对比", fontweight="bold", pad=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    path = OUT_DIR / "fig02_strategy_comparison.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  [OK] 图2 策略对比 → {path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  图3: PR 曲线（使用现有 pr_curve.json）
# ═══════════════════════════════════════════════════════════════════════════════

def plot_pr_curve(pr_data: dict):
    """绘制 PR 曲线 — 论文主图"""
    points = pr_data["points"]
    points_all = pr_data.get("points_all", None)

    fig, ax = plt.subplots(figsize=(6, 5))

    def prep(points):
        recalls = [p["recall"] for p in points]
        precisions = [p["precision"] for p in points]
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

    # Person 过滤曲线
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

    # 阈值标注（每隔几个点）
    step = max(1, len(points) // 6)
    for i in range(0, len(points), step):
        p = points[i]
        ax.annotate(f"{p['threshold']:.2f}m",
                    (p["recall"], p["precision"]),
                    fontsize=6.5, color=C_DARK,
                    ha="left" if i < len(points)//2 else "right",
                    va="bottom", xytext=(3, 3), textcoords="offset points")

    ax.plot([0, 1], [0, 1], "--", color="#ccc", linewidth=0.8, label="随机基准")

    # 最佳 F1 点
    best = max(points, key=lambda p: p["f1"])
    ax.plot(best["recall"], best["precision"], "*", color=C_ORANGE, markersize=12,
            zorder=5, markeredgecolor="white", markeredgewidth=0.8)
    ax.annotate(f"最佳 F1={best['f1']:.3f}\n@ {best['threshold']:.2f}m",
                (best["recall"], best["precision"]),
                fontsize=7, color=C_ORANGE, fontweight="bold",
                xytext=(8, 8), textcoords="offset points",
                arrowprops=dict(arrowstyle="->", color=C_ORANGE, alpha=0.6))

    ax.set_xlabel("Recall (召回率)")
    ax.set_ylabel("Precision (精确率)")
    ax.set_title("Precision-Recall 曲线", fontweight="bold", pad=10)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower left", fontsize=8)

    fig.tight_layout()
    path = OUT_DIR / "fig03_pr_curve.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  [OK] 图3 PR 曲线 → {path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  图4: F1 / Precision / Recall 随阈值变化
# ═══════════════════════════════════════════════════════════════════════════════

def plot_f1_threshold(pr_data: dict):
    """F1 / P / R 随中心距阈值变化"""
    points = pr_data["points"]
    points_all = pr_data.get("points_all", None)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4), sharey=False)

    for idx, (pts, title, color) in enumerate([
        (points, "Person 过滤", C_BLUE),
        (points_all, "全部检测", C_RED) if points_all else (None, None, None),
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
                alpha=0.6, label="Precision")
        ax.plot(thr, re, "--v", color=C_GREEN, linewidth=1, markersize=3,
                alpha=0.6, label="Recall")

        # 最佳 F1 标记
        best_idx = int(np.argmax(f1))
        ax.axvline(x=thr[best_idx], color=color, linestyle=":", alpha=0.4)
        ax.annotate(f"F1={f1[best_idx]:.3f}\n@{thr[best_idx]:.2f}m",
                    xy=(thr[best_idx], f1[best_idx]),
                    fontsize=7, color=color, fontweight="bold",
                    xytext=(10, -25), textcoords="offset points",
                    arrowprops=dict(arrowstyle="->", color=color, alpha=0.5))

        ax.set_xlabel("中心距阈值 (m)")
        ax.set_xlim(0.05, 1.0)
        ax.set_ylim(0, 1.05)
        ax.set_title(title, fontweight="bold", pad=8)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7, loc="lower left")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("F1/Precision/Recall 随匹配阈值变化", fontweight="bold", y=1.02)
    fig.tight_layout()
    path = OUT_DIR / "fig04_f1_threshold.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  [OK] 图4 F1-阈值 → {path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  图5: 分类混淆分析（堆叠柱状图）
# ═══════════════════════════════════════════════════════════════════════════════

def plot_classification_analysis():
    """行人类别识别分析 — 堆叠柱状图"""
    # 数据来自 3 轮平均
    categories = ["正确分类\n为 person", "误分类为\nobstacle", "完全漏检\n(FN)"]
    values_pct = [56.5, 13.9, 29.6]  # 3 轮平均
    colors = [C_GREEN, C_ORANGE, C_RED]

    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    bars = ax.bar(categories, values_pct, color=colors, edgecolor="white",
                  linewidth=0.8, width=0.5, zorder=3)

    for bar, v in zip(bars, values_pct):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{v:.1f}%", ha="center", va="bottom", fontsize=10,
                fontweight="bold", color=C_DARK)

    # 在柱内加数量标注
    counts = [707, 170, 347]  # 3 轮平均近似值
    for bar, c in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 0.35,
                f"n={c}", ha="center", va="center", fontsize=7.5,
                color="white", fontweight="bold")

    ax.set_ylabel("占全部 GT 比例 (%)")
    ax.set_title("行人类别识别分析", fontweight="bold", pad=10)
    ax.set_ylim(0, 80)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    path = OUT_DIR / "fig05_classification_analysis.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  [OK] 图5 分类分析 → {path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  图6: 消融实验对比图
# ═══════════════════════════════════════════════════════════════════════════════

def plot_ablation_comparison():
    """参数消融对比（墙距离 / 剪叶 / eps 组合）"""
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

    # 左: F1 + FP
    x = np.arange(len(configs))
    width = 0.3
    bars = ax1.bar(x, f1s, width, color=C_BLUE, edgecolor="white", zorder=3)
    ax1.set_xticks(x)
    ax1.set_xticklabels(configs, fontsize=6.5, ha="center")
    ax1.set_ylabel("F1 Score")
    ax1.set_ylim(0.5, 0.75)
    ax1.set_title("F1 对比", fontweight="bold", pad=8)
    ax1.grid(axis="y", alpha=0.3, zorder=0)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    for bar, v in zip(bars, f1s):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=7,
                fontweight="bold", color=C_DARK)

    # FP 第二轴
    ax1b = ax1.twinx()
    ax1b.plot(x, fps, "D-", color=C_RED, linewidth=1.5, markersize=5, zorder=5)
    ax1b.set_ylabel("FP 数量", color=C_RED)
    ax1b.tick_params(axis="y", colors=C_RED)
    for i, fp in enumerate(fps):
        ax1b.annotate(str(fp), (x[i], fp), fontsize=6.5, color=C_RED,
                     ha="center", va="bottom")

    # 右: P/R
    ax2.bar(x - width/2, precisions, width, color=C_BLUE, edgecolor="white",
            label="Precision", zorder=3)
    ax2.bar(x + width/2, recalls, width, color=C_GREEN, edgecolor="white",
            label="Recall", zorder=3)
    ax2.set_xticks(x)
    ax2.set_xticklabels(configs, fontsize=6.5, ha="center")
    ax2.set_ylabel("百分比 (%)")
    ax2.set_ylim(50, 85)
    ax2.set_title("Precision / Recall 对比", fontweight="bold", pad=8)
    ax2.legend(fontsize=7)
    ax2.grid(axis="y", alpha=0.3, zorder=0)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.suptitle("prune_qt 参数消融实验", fontweight="bold", y=1.02)
    fig.tight_layout()
    path = OUT_DIR / "fig06_ablation.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  [OK] 图6 消融实验 → {path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  图7: 管线延迟分解 — 箱线图/堆叠
# ═══════════════════════════════════════════════════════════════════════════════

def plot_latency_from_jsonl(jsonl_path: Path):
    """从 pipeline.jsonl 读取并绘制延迟分解图"""
    frames = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                frames.append(json.loads(line))

    if not frames:
        print("  [SKIP] pipeline.jsonl 无数据")
        return

    # 提取各阶段延迟
    n_frames = max(f["frame"] for f in frames) + 1
    frame_indices = [f["frame"] for f in frames]
    total = np.array([f["elapsed_ms"] for f in frames])

    stages = ["join", "fuse", "io", "tracker"]
    stage_labels = ["点云处理", "特征融合", "I/O", "目标跟踪"]
    stage_colors = [C_CYAN, C_BLUE, C_DARK, C_RED]
    stage_data = {s: np.array([f["stages_ms"][s] for f in frames]) for s in stages}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))

    # 左: 堆叠面积图
    ax1.stackplot(frame_indices,
                  [stage_data[s] for s in stages],
                  labels=stage_labels, colors=stage_colors, alpha=0.85)
    ax1.plot(frame_indices, total, "-", color="black", linewidth=0.8, alpha=0.5, label="总耗时")
    ax1.axhline(y=np.mean(total), color="gray", linestyle="--", alpha=0.5,
                label=f"均值 {np.mean(total):.0f}ms")
    ax1.set_xlabel("帧号")
    ax1.set_ylabel("耗时 (ms)")
    ax1.set_title("管线各阶段延迟分解", fontweight="bold", pad=8)
    ax1.legend(fontsize=7, loc="upper left")
    ax1.set_xlim(0, n_frames)
    ax1.set_ylim(bottom=0)
    ax1.grid(axis="y", alpha=0.2)

    # 右: 总延迟直方图
    ax2.hist(total, bins=40, color=C_BLUE, edgecolor="white", linewidth=0.3,
             alpha=0.8, zorder=3)
    ax2.axvline(x=np.mean(total), color=C_RED, linestyle="--", linewidth=1.5,
                label=f"均值 {np.mean(total):.1f}ms")
    ax2.axvline(x=np.median(total), color=C_GREEN, linestyle=":", linewidth=1.5,
                label=f"中位数 {np.median(total):.1f}ms")
    ax2.set_xlabel("每帧总耗时 (ms)")
    ax2.set_ylabel("帧数")
    ax2.set_title("延迟分布直方图", fontweight="bold", pad=8)
    ax2.legend(fontsize=7)
    ax2.grid(axis="y", alpha=0.2)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.suptitle("管线实时性能分析", fontweight="bold", y=1.02)
    fig.tight_layout()
    path = OUT_DIR / "fig07_latency.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  [OK] 图7 延迟分析 → {path}")

    # 延迟统计摘要
    print(f"    总帧数: {len(frames)}")
    print(f"    平均延迟: {np.mean(total):.1f}ms ± {np.std(total):.1f}")
    print(f"    中位数: {np.median(total):.1f}ms | 最大: {np.max(total):.1f}ms | 最小: {np.min(total):.1f}ms")
    print(f"    等效 FPS: {1000/np.mean(total):.1f}")


# ═══════════════════════════════════════════════════════════════════════════════
#  图8: 检测数量统计图
# ═══════════════════════════════════════════════════════════════════════════════

def plot_detection_stats_from_jsonl(jsonl_path: Path):
    """从 pipeline.jsonl 读取检测统计"""
    frames = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                frames.append(json.loads(line))

    if not frames:
        return

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))

    frame_indices = [f["frame"] for f in frames]
    n_targets = [f["stats"]["n_targets"] for f in frames]
    n_person = [f["stats"]["n_person"] for f in frames]

    # 左: 目标数变化
    ax = axes[0]
    ax.plot(frame_indices, n_targets, "-", color=C_BLUE, linewidth=0.7, label="总目标", alpha=0.8)
    ax.plot(frame_indices, n_person, "-", color=C_RED, linewidth=0.7, label="行人", alpha=0.8)
    ax.fill_between(frame_indices, n_person, alpha=0.12, color=C_RED)
    ax.set_xlabel("帧号")
    ax.set_ylabel("目标数")
    ax.set_title("每帧检测目标数", fontweight="bold")
    ax.legend(fontsize=7)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.2)

    # 中: 聚类数
    ax = axes[1]
    n_clusters = [f["stats"]["n_clusters"] for f in frames]
    ax.plot(frame_indices, n_clusters, "-", color=C_GREEN, linewidth=0.7, alpha=0.8)
    ax.fill_between(frame_indices, n_clusters, alpha=0.12, color=C_GREEN)
    ax.set_xlabel("帧号")
    ax.set_ylabel("聚类数")
    ax.set_title("每帧聚类数", fontweight="bold")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.2)

    # 右: 分类分布堆叠
    ax = axes[2]
    cats = ["moving", "static", "movable", "floating"]
    labels = ["Moving", "Static", "Movable", "Floating"]
    colors = [C_RED, C_BLUE, C_GREEN, C_YELLOW]
    bottom = np.zeros(len(frame_indices))
    for cat, color, label in zip(cats, colors, labels):
        values = np.array([f["stats"].get(f"n_{cat}", 0) for f in frames])
        if values.sum() > 0:
            ax.bar(frame_indices, values, bottom=bottom, width=1.0,
                   color=color, label=label, alpha=0.8, edgecolor="none")
            bottom += values
    ax.set_xlabel("帧号")
    ax.set_ylabel("目标数")
    ax.set_title("目标分类分布", fontweight="bold")
    ax.legend(fontsize=6.5, loc="upper left")
    ax.set_ylim(bottom=0)

    fig.suptitle("检测统计与分析", fontweight="bold", y=1.02)
    fig.tight_layout()
    path = OUT_DIR / "fig08_detection_stats.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  [OK] 图8 检测统计 → {path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  图9: Pipeline 演化对比
# ═══════════════════════════════════════════════════════════════════════════════

def plot_pipeline_evolution():
    """管线演化对比图（基于文档数据）"""
    eras = ["Era 1a\n原始全量", "Era 1b\n+降采样", "Era 2\n+去地面", "Era 3\n+去墙体"]
    points = [20006, 20006, 15779, 1484]
    latency = [10096, 41, 29, 29]
    clusters = [41.4, 46.2, 36.3, 29.9]
    persons = [12.1, 12.5, 12.9, 6.6]

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))

    # 左: 输入点量
    ax = axes[0]
    colors = [C_RED if p == max(points) else C_BLUE for p in points]
    bars = ax.bar(eras, points, color=colors, edgecolor="white", width=0.5, zorder=3)
    ax.set_ylabel("输入点量")
    ax.set_title("输入点云量变化", fontweight="bold")
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars, points):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.05,
                f"{v}", ha="center", va="bottom", fontsize=6.5, fontweight="bold")

    # 中: 延迟
    ax = axes[1]
    colors = [C_RED if l == max(latency) else C_BLUE for l in latency]
    bars = ax.bar(eras, latency, color=colors, edgecolor="white", width=0.5, zorder=3)
    ax.set_ylabel("每帧耗时 (ms)")
    ax.set_title("处理延迟变化", fontweight="bold")
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars, latency):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.05,
                f"{v}ms", ha="center", va="bottom", fontsize=6.5, fontweight="bold")

    # 右: 检测性能
    ax = axes[2]
    ax.plot(eras, persons, "-o", color=C_RED, linewidth=1.5, markersize=6, label="人检测数")
    ax.set_ylabel("人检测数（帧均）")
    ax.set_title("检测性能变化", fontweight="bold")
    ax.grid(alpha=0.3)
    for i, v in enumerate(persons):
        ax.annotate(f"{v:.1f}", (eras[i], v), fontsize=7,
                   ha="center", va="bottom", fontweight="bold")

    fig.suptitle("点云处理管线演化对比", fontweight="bold", y=1.02)
    fig.tight_layout()
    path = OUT_DIR / "fig09_pipeline_evolution.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  [OK] 图9 管线演化 → {path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  图10: 多轮稳定性分析
# ═══════════════════════════════════════════════════════════════════════════════

def plot_stability_analysis(results: list[dict]):
    """多轮运行稳定性分析"""
    if len(results) < 2:
        return

    fig, axes = plt.subplots(1, 3, figsize=(9, 3.5))

    runs = list(range(1, len(results) + 1))

    for idx, (key, label, color) in enumerate([
        ("person_only", "Person 过滤", C_BLUE),
        ("all_classes", "全部类别", C_RED),
    ]):
        ax = axes[idx]
        precisions = [r[key]["precision"] * 100 for r in results]
        recalls = [r[key]["recall"] * 100 for r in results]
        f1s = [r[key]["f1"] * 100 for r in results]

        ax.plot(runs, precisions, "-o", color=color, markersize=6, label="Precision")
        ax.plot(runs, recalls, "-s", color=C_GREEN, markersize=6, label="Recall")
        ax.plot(runs, f1s, "-^", color=C_ORANGE, markersize=6, label="F1")

        ax.set_xticks(runs)
        ax.set_xlabel("运行轮次")
        ax.set_ylabel("百分比 (%)")
        ax.set_title(label, fontweight="bold", pad=8)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # 右: 检测数/FP 稳定性
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
    ax.set_title("检测数量稳定性", fontweight="bold", pad=8)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle("多轮运行稳定性分析（YOLO 非确定性）", fontweight="bold", y=1.02)
    fig.tight_layout()
    path = OUT_DIR / "fig10_stability.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  [OK] 图10 稳定性 → {path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  主入口
# ═══════════════════════════════════════════════════════════════════════════════

def print_summary_table(results: list[dict]):
    """终端输出最终结果汇总表"""
    print("\n" + "=" * 65)
    print("  论文实验数据汇总（3 轮平均）")
    print("=" * 65)

    # Person 过滤
    p_prec = [r["person_only"]["precision"] * 100 for r in results]
    p_rec  = [r["person_only"]["recall"] * 100 for r in results]
    p_f1   = [r["person_only"]["f1"] for r in results]
    p_fp   = [r["person_only"]["fp"] for r in results]
    p_tp   = [r["person_only"]["tp"] for r in results]
    p_fn   = [r["person_only"]["fn"] for r in results]

    # 全部类别
    a_prec = [r["all_classes"]["precision"] * 100 for r in results]
    a_rec  = [r["all_classes"]["recall"] * 100 for r in results]
    a_f1   = [r["all_classes"]["f1"] for r in results]
    a_tp   = [r["all_classes"]["tp"] for r in results]
    a_fp   = [r["all_classes"]["fp"] for r in results]
    a_fn   = [r["all_classes"]["fn"] for r in results]

    print(f"\n  ── Person 过滤 (仅 class_type == 'person') ──")
    print(f"    GT: {results[0]['person_only']['n_gt']}")
    print(f"    Precision: {np.mean(p_prec):.1f}% ± {np.std(p_prec):.1f}")
    print(f"    Recall:    {np.mean(p_rec):.1f}% ± {np.std(p_rec):.1f}")
    print(f"    F1:        {np.mean(p_f1):.3f} ± {np.std(p_f1):.4f}")
    print(f"    TP: {np.mean(p_tp):.0f} ± {np.std(p_tp):.0f}")
    print(f"    FP: {np.mean(p_fp):.0f} ± {np.std(p_fp):.0f}")
    print(f"    FN: {np.mean(p_fn):.0f} ± {np.std(p_fn):.0f}")

    print(f"\n  ── 全部类别 (All Classes) ──")
    print(f"    Precision: {np.mean(a_prec):.1f}% ± {np.std(a_prec):.1f}")
    print(f"    Recall:    {np.mean(a_rec):.1f}% ± {np.std(a_rec):.1f}")
    print(f"    F1:        {np.mean(a_f1):.3f} ± {np.std(a_f1):.4f}")
    print(f"    TP: {np.mean(a_tp):.0f} ± {np.std(a_tp):.0f}")
    print(f"    FP: {np.mean(a_fp):.0f} ± {np.std(a_fp):.0f}")
    print(f"    FN: {np.mean(a_fn):.0f} ± {np.std(a_fn):.0f}")

    # 分类质量
    # 分类质量
    tp_p = np.mean([r.get("tp_person", r["person_only"]["tp"]) for r in results])
    tp_n = np.mean([r.get("tp_nonperson", r["all_classes"]["tp"] - r["person_only"]["tp"]) for r in results])
    total_gt = results[0]["all_classes"]["n_gt"]
    print(f"\n  ── 分类质量分析 ──")
    print(f"    正确分类为 person: {tp_p:.0f} ({tp_p/total_gt*100:.1f}%)")
    print(f"    误分类为 obstacle: {tp_n:.0f} ({tp_n/total_gt*100:.1f}%)")
    print(f"    完全漏检 (FN):     {total_gt - tp_p - tp_n:.0f} ({(total_gt-tp_p-tp_n)/total_gt*100:.1f}%)")
    print(f"    分类正确率 (TP中person占比): {tp_p/(tp_p+tp_n)*100:.1f}%")
    print()


def main():
    # 1. 加载多轮验证结果
    results = auto_find_eval_results()
    if len(results) < 1:
        print("[ERR] 未找到 eval_ablation 结果，请先运行验证")
        return

    print(f"找到 {len(results)} 轮验证结果:")
    for i, r in enumerate(results):
        ts = r.get("config", {}).get("cluster_strategy", "?")
        print(f"  第{i+1}轮: {ts} | Person F1={r['person_only']['f1']:.4f} "
              f"P={r['person_only']['precision']*100:.1f}% "
              f"R={r['person_only']['recall']*100:.1f}%")

    # 2. 输出汇总表
    print_summary_table(results)

    # 3. 加载 PR 曲线数据
    pr_data = auto_find_pr_curve()
    if pr_data is None:
        print("[WARN] 未找到 pr_curve.json，跳过 PR 曲线相关图表")

    # 4. 加载 pipeline.jsonl
    jsonl_path = auto_find_pipeline_jsonl()

    # 5. 生成所有图表
    print("\n" + "=" * 65)
    print("  生成论文图表...")
    print("=" * 65)

    plot_main_metrics(results)
    plot_strategy_comparison()
    plot_classification_analysis()
    plot_ablation_comparison()
    plot_stability_analysis(results)

    if pr_data:
        plot_pr_curve(pr_data)
        plot_f1_threshold(pr_data)
    else:
        print("  [SKIP] 图3/4 PR曲线 — 需要 pr_curve.json")

    if jsonl_path:
        print(f"\n  加载 pipeline.jsonl: {jsonl_path}")
        plot_latency_from_jsonl(jsonl_path)
        plot_detection_stats_from_jsonl(jsonl_path)
    else:
        print("  [SKIP] 图7/8 管线性能 — 需要 pipeline.jsonl")

    plot_pipeline_evolution()

    print("\n" + "=" * 65)
    print(f"  完成！所有图表已保存至: {OUT_DIR.resolve()}")
    print("=" * 65)
    print(f"\n  生成文件列表:")
    for p in sorted(OUT_DIR.glob("fig*.png")):
        size_kb = p.stat().st_size / 1024
        print(f"    {p.name}  ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
