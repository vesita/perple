# -*- coding: utf-8 -*-
"""
墙体策略对比分析：BevLsd vs BevEdLine

输出：
  1. wall_projection.png     — BEV 投影对比（密度 + 墙点分类）
  2. accuracy_strict.png     — 严格精度对比（P/R/F1 + 标准差）
  3. accuracy_spatial.png    — 空间精度对比（P/R/F1 + 标准差）
  4. speed_comparison.png    — 速度对比（平均耗时 + 标准差）

用法：
  .venv/Scripts/python.exe scripts/wall_strategy_analysis.py
"""

import json
import os
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

# ── 字体配置 ──────────────────────────────────────────────────────────────
# 尝试多个中文字体路径，适配不同系统
_CANDIDATE_FONTS = [
    "C:/Windows/Fonts/msyh.ttc",         # 微软雅黑
    "C:/Windows/Fonts/simhei.ttf",        # 黑体
    "C:/Windows/Fonts/simsun.ttc",        # 宋体
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]

_FONT_SET = False
for fp in _CANDIDATE_FONTS:
    if os.path.exists(fp):
        font_manager.fontManager.addfont(fp)
        font_name = font_manager.FontProperties(fname=fp).get_name()
        plt.rcParams["font.family"] = font_name
        plt.rcParams["axes.unicode_minus"] = False
        _FONT_SET = True
        print(f"[字体] 使用: {font_name} ({fp})")
        break

if not _FONT_SET:
    print("[字体] 警告：未找到中文字体，中文可能显示为方框")

# ── 全局绘图参数（论文排版） ──────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

OUT_DIR = Path("output/wall_strategy_analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════
#  1. 墙线投影图
# ══════════════════════════════════════════════════════════════════════════

STRATEGY_LABELS = {
    "bev_lsd": "BevLSD",
    "bev_edlines": "BevEDLines",
    "bev_hough": "BevHough",
}
STRATEGY_COLORS = {
    "bev_lsd": "#E74C3C",
    "bev_edlines": "#2E86C1",
    "bev_hough": "#28B463",
}

def load_wall_viz(path="output/wall_compare_viz/wall_compare.json"):
    """加载墙体对比可视化 JSON 数据."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def load_non_ground(path="output/wall_compare_viz/non_ground.json"):
    """加载非地面点云."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)  # list of [x, y, z]

def plot_wall_projection(viz, non_ground):
    """生成 BEV 投影对比图：密度热力图 + 各策略墙点/非墙点分类."""
    bev = viz["bev"]
    size = bev["size"]
    max_range = bev["max_range"]
    density = np.array(bev["density"], dtype=np.float32).reshape(size, size)
    pts = np.array(non_ground, dtype=np.float32)  # (N, 3)

    extent = [-max_range, max_range, -max_range, max_range]
    strategies = viz["strategies"]

    strategies = [s for s in strategies if s["name"] in STRATEGY_LABELS]
    n_strat = len(strategies)
    fig, axes = plt.subplots(1, n_strat, figsize=(6 * n_strat, 5.5))
    if n_strat == 1:
        axes = [axes]

    for ax, strat in zip(axes, strategies):
        name = strat["name"]
        label = STRATEGY_LABELS.get(name, name)

        # 背景：BEV 密度
        ax.imshow(density, origin="lower", extent=extent, cmap="Greys", alpha=0.6)

        # 非墙点（灰色小点）
        nw_idx = strat["non_wall_indices"]
        if nw_idx:
            nw_pts = pts[nw_idx]
            ax.scatter(nw_pts[:, 0], nw_pts[:, 1], s=1.0, c="#AAAAAA",
                       alpha=0.4, linewidths=0, label="非墙面点")

        # 墙点（彩色）
        w_idx = strat["wall_indices"]
        if w_idx:
            w_pts = pts[w_idx]
            ax.scatter(w_pts[:, 0], w_pts[:, 1], s=1.5,
                       c=STRATEGY_COLORS.get(name, "#E74C3C"),
                       alpha=0.7, linewidths=0, label="墙面点")

        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-max_range, max_range)
        ax.set_aspect("equal")
        ax.set_title(f"{label}  ({strat['n_wall']} 墙面点)")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.legend(loc="upper right", markerscale=4, fontsize=9)
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    out_path = OUT_DIR / "wall_projection.png"
    fig.savefig(out_path)
    print(f"[墙线投影] → {out_path}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
#  2. 精度对比（5 次取平均）
# ══════════════════════════════════════════════════════════════════════════

def _parse_eval_json(path):
    """读取单次 eval_labeled 输出的 JSON."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def collect_accuracy(base_dir="output/eval_compare"):
    """
    从 eval_compare/bev_lsd/run{1..5}/eval_result.json 等路径加载数据，
    返回 { strategy_name: { metric: [values] } }.
    """
    strategies = ["bev_lsd", "bev_edlines"]
    metrics = {}

    for strat in strategies:
        values = {"precision": [], "recall": [], "f1": [],
                  "precision_spatial": [], "recall_spatial": [], "f1_spatial": [],
                  "tp": [], "fp": [], "fn": []}

        for run in range(1, 6):
            path = Path(base_dir) / strat / f"run{run}" / "eval_result.json"
            if not path.exists():
                print(f"[精度] 警告：{path} 不存在，跳过")
                continue
            try:
                d = _parse_eval_json(path)
                values["precision"].append(d["precision"])
                values["recall"].append(d["recall"])
                values["f1"].append(d["f1"])
                values["precision_spatial"].append(d["precision_spatial"])
                values["recall_spatial"].append(d["recall_spatial"])
                values["f1_spatial"].append(d["f1_spatial"])
                values["tp"].append(d["tp"])
                values["fp"].append(d["fp"])
                values["fn"].append(d["fn_"])
            except (KeyError, json.JSONDecodeError) as e:
                print(f"[精度] 警告：{path} 解析失败 ({e})")

        metrics[strat] = values

    return metrics


def plot_accuracy(metrics):
    """精度对比图：Strict + Spatial 分开两张图，各自独立输出."""
    strategies = ["bev_lsd", "bev_edlines"]
    labels = [STRATEGY_LABELS[s] for s in strategies]
    colors = [STRATEGY_COLORS[s] for s in strategies]
    x = np.arange(3)  # precision, recall, f1
    width = 0.30

    # ── Strict（仅 person 检测参与） ──
    fig, ax = plt.subplots(figsize=(6, 4.5))
    strict_metrics = ["precision", "recall", "f1"]
    for i, strat in enumerate(strategies):
        vals = [np.mean(metrics[strat][m]) * 100 for m in strict_metrics]
        errs = [np.std(metrics[strat][m]) * 100 for m in strict_metrics]
        ax.bar(x + i * width, vals, width, label=labels[i],
               color=colors[i], yerr=errs, capsize=3,
               error_kw={"linewidth": 1})

    ax.set_title("严格评估（仅 Person）")
    ax.set_ylabel("百分比 (%)")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(["Precision", "Recall", "F1"])
    ax.legend(fontsize=9)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out_path = OUT_DIR / "accuracy_strict.png"
    fig.savefig(out_path)
    print(f"[精度对比-严格] → {out_path}")
    plt.close(fig)

    # ── Spatial（全部检测参与） ──
    fig, ax = plt.subplots(figsize=(6, 4.5))
    spatial_metrics = ["precision_spatial", "recall_spatial", "f1_spatial"]
    for i, strat in enumerate(strategies):
        vals = [np.mean(metrics[strat][m]) * 100 for m in spatial_metrics]
        errs = [np.std(metrics[strat][m]) * 100 for m in spatial_metrics]
        ax.bar(x + i * width, vals, width, label=labels[i],
               color=colors[i], yerr=errs, capsize=3,
               error_kw={"linewidth": 1})

    ax.set_title("空间评估（全部检测）")
    ax.set_ylabel("百分比 (%)")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(["Precision", "Recall", "F1"])
    ax.legend(fontsize=9)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out_path = OUT_DIR / "accuracy_spatial.png"
    fig.savefig(out_path)
    print(f"[精度对比-空间] → {out_path}")
    plt.close(fig)


def print_accuracy_table(metrics):
    """打印精度表格到控制台."""
    print()
    print("═" * 70)
    print("  精度对比（5 次平均 ± 标准差）")
    print("═" * 70)
    print(f"  {'策略':<16} {'Precision':>12} {'Recall':>12} {'F1':>10}   {'S.Precision':>12} {'S.Recall':>12} {'S.F1':>10}")
    print("  " + "-" * 88)

    for strat in ["bev_lsd", "bev_edlines"]:
        m = metrics[strat]
        p_mean = np.mean(m["precision"]) * 100
        p_std = np.std(m["precision"]) * 100
        r_mean = np.mean(m["recall"]) * 100
        r_std = np.std(m["recall"]) * 100
        f_mean = np.mean(m["f1"])
        f_std = np.std(m["f1"])
        sp_mean = np.mean(m["precision_spatial"]) * 100
        sp_std = np.std(m["precision_spatial"]) * 100
        sr_mean = np.mean(m["recall_spatial"]) * 100
        sr_std = np.std(m["recall_spatial"]) * 100
        sf_mean = np.mean(m["f1_spatial"])
        sf_std = np.std(m["f1_spatial"])

        print(f"  {STRATEGY_LABELS[strat]:<16} "
              f"{p_mean:>5.1f}±{p_std:.1f}%  {r_mean:>5.1f}±{r_std:.1f}%  {f_mean:.4f}±{f_std:.4f}  "
              f"{sp_mean:>5.1f}±{sp_std:.1f}%  {sr_mean:>5.1f}±{sr_std:.1f}%  {sf_mean:.4f}±{sf_std:.4f}")

    print()


# ══════════════════════════════════════════════════════════════════════════
#  3. 速度对比（从 wall_compare_viz JSON 读取）
# ══════════════════════════════════════════════════════════════════════════

def plot_speed(viz):
    """速度对比图：各策略单帧耗时."""
    strategies = [s for s in viz["strategies"] if s["name"] in STRATEGY_LABELS]
    names = [STRATEGY_LABELS[s["name"]] for s in strategies]
    times = [s["elapsed_ms"] for s in strategies]
    colors = [STRATEGY_COLORS[s["name"]] for s in strategies]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(names, times, color=colors, width=0.5, edgecolor="black", linewidth=0.5)

    # 在柱子上标注数值
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{t:.1f} ms", ha="center", va="bottom", fontsize=10)

    ax.set_ylabel("耗时 (ms)")
    ax.set_title("墙体提取耗时对比（单帧）")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(times) * 1.25)

    plt.tight_layout()
    out_path = OUT_DIR / "speed_comparison.png"
    fig.savefig(out_path)
    print(f"[速度对比] → {out_path}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  墙体策略对比分析")
    print("=" * 60)

    # ── 1. 墙线投影图 ──
    print("\n[1/3] 生成墙线投影图...")
    viz_path = "output/wall_compare_viz/wall_compare.json"
    ng_path = "output/wall_compare_viz/non_ground.json"

    if os.path.exists(viz_path) and os.path.exists(ng_path):
        viz = load_wall_viz(viz_path)
        non_ground = load_non_ground(ng_path)
        plot_wall_projection(viz, non_ground)
    else:
        print(f"  [跳过] 缺少可视化数据 (需先运行 wall_compare_viz)")

    # ── 2. 精度对比 ──
    print("\n[2/3] 精度对比...")
    metrics = collect_accuracy()
    if any(len(v["precision"]) > 0 for v in metrics.values()):
        plot_accuracy(metrics)
        print_accuracy_table(metrics)
    else:
        print("  [跳过] 缺少精度数据 (需先运行 eval_labeled 5 次)")

    # ── 3. 速度对比 ──
    print("\n[3/3] 速度对比...")
    if os.path.exists(viz_path):
        viz = load_wall_viz(viz_path)
        plot_speed(viz)
    else:
        print("  [跳过] 缺少速度数据")

    print(f"\n所有图片已保存至: {OUT_DIR}/")
    print("完成。")


if __name__ == "__main__":
    main()
