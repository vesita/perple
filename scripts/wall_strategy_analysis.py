# -*- coding: utf-8 -*-
"""墙体策略对比分析 — BevLsd vs BevEdLines（统一风格）

用法:
    .venv/Scripts/python.exe scripts/wall_strategy_analysis.py
"""
import json, os, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.chart_style import savefig, style_ax, bar_labels
from scripts.chart_style import C_BLUE, C_RED, C_GREEN, C_GRAY, SIZES

OUT_DIR = Path("output/wall_strategy_analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FONT_LABEL = 10.5
FONT_LEGEND = 9

STRATEGY_LABELS = {"bev_lsd": "BevLSD", "bev_edlines": "BevEDLines"}
STRATEGY_COLORS = {"bev_lsd": C_RED, "bev_edlines": C_BLUE}


def load_wall_viz(path="output/wall_compare_viz/wall_compare.json"):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_non_ground(path="output/wall_compare_viz/non_ground.json"):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_bev_density(viz):
    bev = viz["bev"]
    size = bev["size"]
    return np.array(bev["density"], dtype=np.float32).reshape(size, size)


def plot_bev_density(viz):
    density = get_bev_density(viz)
    max_range = viz["bev"]["max_range"]
    extent = [-max_range, max_range, -max_range, max_range]

    fig, ax = plt.subplots(figsize=SIZES["single_bar"])
    ax.imshow(density, origin="lower", extent=extent, cmap="Greys")
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)", fontsize=FONT_LABEL)
    ax.set_ylabel("Y (m)", fontsize=FONT_LABEL)
    style_ax(ax, grid_axis="both")
    savefig(fig, OUT_DIR / "bev_density.png")
    print("  → bev_density.png")


def plot_wall_projection(viz, non_ground):
    density = get_bev_density(viz)
    max_range = viz["bev"]["max_range"]
    extent = [-max_range, max_range, -max_range, max_range]
    pts = np.array(non_ground, dtype=np.float32)

    strategies = [s for s in viz["strategies"] if s["name"] in STRATEGY_LABELS]

    for strat in strategies:
        name = strat["name"]
        label = STRATEGY_LABELS.get(name, name)

        fig, ax = plt.subplots(figsize=(SIZES["single_bar"]))
        ax.imshow(density, origin="lower", extent=extent, cmap="Greys", alpha=0.5)

        nw_idx = strat["non_wall_indices"]
        if len(nw_idx):
            nw_pts = pts[nw_idx]
            ax.scatter(nw_pts[:, 0], nw_pts[:, 1], s=3.0,
                       facecolors="none", edgecolors=C_GRAY,
                       alpha=0.5, linewidths=0.3, label="非墙面点")

        w_idx = strat["wall_indices"]
        if len(w_idx):
            w_pts = pts[w_idx]
            ax.scatter(w_pts[:, 0], w_pts[:, 1], s=5.0,
                       c=STRATEGY_COLORS.get(name, C_RED),
                       alpha=0.9, linewidths=0, label="墙面点")

        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-max_range, max_range)
        ax.set_aspect("equal")
        ax.set_xlabel("X (m)", fontsize=FONT_LABEL)
        ax.set_ylabel("Y (m)", fontsize=FONT_LABEL)
        ax.legend(loc="upper right", markerscale=10, fontsize=FONT_LEGEND)
        style_ax(ax, grid_axis="both")

        savefig(fig, OUT_DIR / f"wall_projection_{name}.png")
        print(f"  → wall_projection_{name}.png ({label}, {strat['n_wall']} 墙面点)")


def collect_accuracy(base_dir="output/eval_compare"):
    strategies = ["bev_lsd", "bev_edlines"]
    metrics = {}
    for strat in strategies:
        values = {"precision": [], "recall": [], "f1": [],
                  "precision_spatial": [], "recall_spatial": [], "f1_spatial": [],
                  "tp": [], "fp": [], "fn": []}
        for run in range(1, 6):
            path = Path(base_dir) / strat / f"run{run}" / "eval_result.json"
            if not path.exists():
                continue
            try:
                d = json.loads(path.read_text(encoding="utf-8"))
                for k in ["precision", "recall", "f1"]:
                    values[k].append(d[k])
                for k in ["precision_spatial", "recall_spatial", "f1_spatial"]:
                    values[k].append(d[k])
                values["tp"].append(d["tp"])
                values["fp"].append(d["fp"])
                values["fn"].append(d["fn_"])
            except (KeyError, json.JSONDecodeError):
                continue
        metrics[strat] = values
    return metrics


def plot_accuracy(metrics):
    strategies = ["bev_lsd", "bev_edlines"]
    labels = [STRATEGY_LABELS[s] for s in strategies]
    colors = [STRATEGY_COLORS[s] for s in strategies]
    x = np.arange(3)
    width = 0.30

    # ── 严格评估（Person 过滤）──
    fig, ax = plt.subplots(figsize=SIZES["single_bar"])
    strict_metrics = ["precision", "recall", "f1"]
    vals_list = []
    for i, strat in enumerate(strategies):
        vals = [np.mean(metrics[strat][m]) * 100 for m in strict_metrics]
        errs = [np.std(metrics[strat][m]) * 100 for m in strict_metrics]
        bars = ax.bar(x + i * width, vals, width, label=labels[i],
                      color=colors[i], yerr=errs, capsize=3,
                      error_kw={"linewidth": 1})
        vals_list.append(vals)
    ax.set_ylabel("百分比 (%)", fontsize=FONT_LABEL)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(["精确率 P", "召回率 R", "F1"], fontsize=10)
    ax.legend(fontsize=FONT_LEGEND)
    style_ax(ax)
    savefig(fig, OUT_DIR / "accuracy_strict.png")
    print("  → accuracy_strict.png")

    # ── 空间评估（全部检测）──
    fig, ax = plt.subplots(figsize=SIZES["single_bar"])
    spatial_metrics = ["precision_spatial", "recall_spatial", "f1_spatial"]
    for i, strat in enumerate(strategies):
        vals = [np.mean(metrics[strat][m]) * 100 for m in spatial_metrics]
        errs = [np.std(metrics[strat][m]) * 100 for m in spatial_metrics]
        bars = ax.bar(x + i * width, vals, width, label=labels[i],
                      color=colors[i], yerr=errs, capsize=3,
                      error_kw={"linewidth": 1})
    ax.set_ylabel("百分比 (%)", fontsize=FONT_LABEL)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(["精确率 P", "召回率 R", "F1"], fontsize=10)
    ax.legend(fontsize=FONT_LEGEND)
    style_ax(ax)
    savefig(fig, OUT_DIR / "accuracy_spatial.png")
    print("  → accuracy_spatial.png")


def print_accuracy_table(metrics):
    print()
    print("=" * 70)
    print("  精度对比（5 次平均 +/- 标准差）")
    print("=" * 70)
    hdr = (f"  {'策略':<16} {'精确率':>10} {'召回率':>10} {'F1':>10}"
           f"   {'空-精确率':>10} {'空-召回率':>10} {'空-F1':>10}")
    print(hdr)
    print("  " + "-" * 88)
    for strat in ["bev_lsd", "bev_edlines"]:
        m = metrics[strat]
        vals = []
        for k in ["precision", "recall", "f1", "precision_spatial", "recall_spatial", "f1_spatial"]:
            is_f1 = k == "f1" or k == "f1_spatial"
            vals.append((np.mean(m[k]) * (100 if not is_f1 else 1),
                         np.std(m[k]) * (100 if not is_f1 else 1)))
        row = f"  {STRATEGY_LABELS[strat]:<16}"
        for mean, std in vals:
            row += f"  {mean:>5.1f}±{std:.1f}" if mean > 2 else f"  {mean:.4f}±{std:.4f}"
        print(row)
    print()


def plot_speed(viz):
    strategies = [s for s in viz["strategies"] if s["name"] in STRATEGY_LABELS]
    names = [STRATEGY_LABELS[s["name"]] for s in strategies]
    times = [s["elapsed_ms"] for s in strategies]
    colors_list = [STRATEGY_COLORS[s["name"]] for s in strategies]

    fig, ax = plt.subplots(figsize=SIZES["single_bar"])
    bars = ax.bar(names, times, color=colors_list, width=0.45, edgecolor="white", linewidth=0.5)
    bar_labels(ax, bars, times, suffix="ms")
    ax.set_ylabel("耗时 (ms)", fontsize=FONT_LABEL)
    style_ax(ax)
    savefig(fig, OUT_DIR / "speed_comparison.png")
    print("  → speed_comparison.png")


def main():
    print("=" * 60)
    print("  墙体策略对比分析")
    print("=" * 60)

    # ── 1. 墙线投影图 ──
    print("\n[1/3] 墙线投影图...")
    viz_path = "output/wall_compare_viz/wall_compare.json"
    ng_path = "output/wall_compare_viz/non_ground.json"
    if os.path.exists(viz_path) and os.path.exists(ng_path):
        viz = load_wall_viz(viz_path)
        non_ground = load_non_ground(ng_path)
        plot_bev_density(viz)
        plot_wall_projection(viz, non_ground)
    else:
        print("  [跳过] 缺少可视化数据")

    # ── 2. 精度对比 ──
    print("\n[2/3] 精度对比...")
    metrics = collect_accuracy()
    if any(len(v["precision"]) > 0 for v in metrics.values()):
        plot_accuracy(metrics)
        print_accuracy_table(metrics)
    else:
        print("  [跳过] 缺少精度数据")

    # ── 3. 速度对比 ──
    print("\n[3/3] 速度对比...")
    if os.path.exists(viz_path):
        plot_speed(load_wall_viz(viz_path))
    else:
        print("  [跳过] 缺少速度数据")

    print(f"\n所有图片已保存至: {OUT_DIR}/")


if __name__ == "__main__":
    main()
