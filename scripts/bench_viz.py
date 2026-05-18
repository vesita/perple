"""
Bench 可视化 — 论文级图表

用法：
  .venv/Scripts/python.exe scripts/bench_viz.py

输出：output/bench_viz/{地面|墙体|聚类|降噪|综合}/ 下的 PNG 图片
"""
import csv, json, os, shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.chart_style import savefig, style_ax, FIG_W, C_BLUE, C_RED, C_GREEN, C_GRAY

OUT_DIR = Path("output/bench_viz")
BENCH_DIR = Path("output/bench")
ANALYSIS_DIR = Path("output/bench/analysis")

# 同族策略配对（网格 vs 四叉树，保留较快者）
_GRID_QT_PAIRS = [
    ("cc_pca_grid", "cc_pca_qt"),
    ("cc_l2_grid", "cc_l2_qt"),
    ("ransac_l2_grid", "ransac_l2_qt"),
    ("dbscan_l2_qt", "dbscan_l2_qt_dif"),
]


def load_bench_data(task):
    results = []
    task_dir = BENCH_DIR / task
    if not task_dir.is_dir():
        return results
    for strategy_dir in sorted(task_dir.iterdir()):
        info_path = strategy_dir / "info.json"
        if not info_path.exists():
            continue
        try:
            data = json.loads(info_path.read_text(encoding="utf-8"))
            for entry in data.get("results", []):
                entry["strategy"] = data.get("strategy", strategy_dir.name)
                entry["mode"] = data.get("mode", "")
                results.append(entry)
        except Exception:
            continue
    return results


def load_stata_csv(task):
    p = ANALYSIS_DIR / "full" / task / "stats.csv"
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def pick_median(data, key="avg_ms"):
    return sorted(data, key=lambda x: x.get(key, 0))[len(data) // 2]


def filter_faster_grid_qt(results):
    grp = {}
    for r in results:
        grp.setdefault(r.get("strategy", "?"), []).append(r)
    drop = set()
    for grid_name, qt_name in _GRID_QT_PAIRS:
        gd = grp.get(grid_name)
        qd = grp.get(qt_name)
        if gd and qd:
            gm = np.median([r.get("avg_ms", 0) for r in gd])
            qm = np.median([r.get("avg_ms", 0) for r in qd])
            drop.add(qt_name if gm <= qm else grid_name)
    return [r for r in results if r.get("strategy") not in drop]


def save_task_fig(fig, task_cn, name):
    path = OUT_DIR / task_cn / name
    path.parent.mkdir(parents=True, exist_ok=True)
    savefig(fig, path)


# ─── 1. 速度柱状图 ──────────────────────────────────────────────────────────

def plot_speed_bar(results, task_cn):
    if not results:
        return
    grp = {}
    for r in results:
        grp.setdefault(r.get("strategy", "?"), []).append(r.get("avg_ms", 0))
    names = sorted(grp, key=lambda k: np.median(grp[k]))
    means = [np.mean(grp[k]) for k in names]

    fig, ax = plt.subplots(figsize=(FIG_W, max(2.5, len(names) * 0.4)))
    colors = [C_GREEN if m <= 100 else C_RED for m in means]
    bars = ax.barh(range(len(names)), means, color=colors, height=0.55, zorder=3)
    for bar, v in zip(bars, means):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{v:.0f} ms", va="center", fontsize=9, color=C_GRAY)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("平均耗时 (ms)", fontsize=10.5)
    ax.invert_yaxis()
    ax.axvline(100, color=C_RED, linestyle="--", linewidth=0.8, alpha=0.5)
    style_ax(ax)
    fig.tight_layout()
    save_task_fig(fig, task_cn, "speed_bar.png")


# ─── 2. 双轴图 ──────────────────────────────────────────────────────────────

def plot_dual_axis(results, task_cn, y_key, y_label, y_fmt="{:.1f}"):
    if not results:
        return
    grp = {}
    for r in results:
        grp.setdefault(r.get("strategy", "?"), []).append(r)
    names = sorted(grp)
    picks = [pick_median(grp[n]) for n in names]
    speeds = [p.get("avg_ms", 0) for p in picks]
    metrics = [p.get("extra", {}).get(y_key, 0) for p in picks]

    fig, ax1 = plt.subplots(figsize=(FIG_W, 2.8))
    x = np.arange(len(names))
    bars = ax1.bar(x, speeds, 0.45, color=C_BLUE, alpha=0.8, zorder=3)
    for bar, v in zip(bars, speeds):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{v:.0f}", ha="center", va="bottom", fontsize=8, color=C_BLUE)
    ax1.set_ylabel("耗时 (ms)", fontsize=10.5, color=C_BLUE)
    ax1.tick_params(axis="y", labelcolor=C_BLUE)
    ax1.set_ylim(0, max(speeds) * 1.35)

    ax2 = ax1.twinx()
    ax2.plot(x, metrics, "o-", color="#e8883a", linewidth=1.5, markersize=5, zorder=5)
    for xi, yi in zip(x, metrics):
        ax2.annotate(y_fmt.format(yi), (xi, yi), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=8, color="#e8883a")
    ax2.set_ylabel(y_label, fontsize=10.5, color="#e8883a")
    ax2.tick_params(axis="y", labelcolor="#e8883a")

    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=8, rotation=20, ha="right")
    ax1.grid(axis="y", alpha=0.2)
    ax1.spines["top"].set_visible(False)
    fig.tight_layout()
    save_task_fig(fig, task_cn, "dual_axis.png")


# ─── 3. 散点图 ──────────────────────────────────────────────────────────────

def plot_scatter(results, task_cn, y_key, y_label, x_label="平均耗时 (ms)"):
    if not results:
        return
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_W * 0.6))
    strategies = sorted({r.get("strategy", "?") for r in results})
    cmap = plt.cm.Set2(np.linspace(0, 1, len(strategies)))
    for si, s_name in enumerate(strategies):
        pts = [r for r in results if r.get("strategy") == s_name]
        xs = [r.get("avg_ms", 0) for r in pts]
        ys = [r.get("extra", {}).get(y_key, 0) for r in pts]
        ax.scatter(xs, ys, c=[cmap[si]], label=s_name, s=45, alpha=0.8,
                   edgecolors="white", linewidth=0.3, zorder=3)
    ax.set_xlabel(x_label, fontsize=10.5)
    ax.set_ylabel(y_label, fontsize=10.5)
    ax.legend(fontsize=8, loc="best", markerscale=0.8)
    style_ax(ax)
    fig.tight_layout()
    save_task_fig(fig, task_cn, "scatter.png")


# ─── 4. 汇总表 ──────────────────────────────────────────────────────────────

def plot_summary_table(results, task_cn, extra_keys=None):
    if not results:
        return
    grp = {}
    for r in results:
        grp.setdefault(r.get("strategy", "?"), []).append(r)
    names = sorted(grp)
    extra_keys = extra_keys or []
    headers = ["策略", "参数", "耗时(ms)", "帧数"] + [k for k in extra_keys]
    rows = []
    for n in names:
        m = pick_median(grp[n])
        ev = [str(m.get("extra", {}).get(k, "")) for k in extra_keys]
        rows.append([n, str(len(grp[n])), f"{m.get('avg_ms',0):.1f}",
                     str(m.get("frame_count", 0))] + ev)

    fig, ax = plt.subplots(figsize=(FIG_W, 0.3 * len(rows) + 0.8))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=headers, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.3)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold", fontsize=8)
        elif row % 2 == 0:
            cell.set_facecolor("#f5f6fa")
        else:
            cell.set_facecolor("white")
        cell.set_edgecolor("#dcdde1")
        cell.set_linewidth(0.5)
    fig.tight_layout()
    save_task_fig(fig, task_cn, "summary_table.png")


# ─── 5. 快速 vs 全量 ────────────────────────────────────────────────────────

def plot_quick_vs_full():
    tasks_cn = {"ground": "地面", "cluster": "聚类", "wall": "墙体", "denoise": "降噪"}
    fig, axes = plt.subplots(2, 2, figsize=(FIG_W, FIG_W * 0.8))
    axes = axes.flatten()
    for ai, (task, cn) in enumerate(tasks_cn.items()):
        ax = axes[ai]
        quick_data, full_data = {}, {}
        qp = ANALYSIS_DIR / "quick" / task / "stats.csv"
        fp_ = ANALYSIS_DIR / "full" / task / "stats.csv"
        if qp.exists():
            with open(qp, encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    quick_data[row["策略"]] = float(row["平均(ms)"])
        if fp_.exists():
            with open(fp_, encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    full_data[row["策略"]] = float(row["平均(ms)"])
        common = sorted(set(quick_data) & set(full_data))
        if not common:
            ax.text(0.5, 0.5, "无数据", ha="center", va="center", fontsize=10)
            continue
        x = np.arange(len(common))
        w = 0.3
        ax.bar(x - w / 2, [quick_data[k] for k in common], w,
               color=C_BLUE, alpha=0.7, label="快速")
        ax.bar(x + w / 2, [full_data[k] for k in common], w,
               color="#e8883a", alpha=0.7, label="全量")
        ax.set_xticks(x)
        ax.set_xticklabels(common, fontsize=7, rotation=30, ha="right")
        style_ax(ax)
        if ai == 0:
            ax.legend(fontsize=8)
    fig.tight_layout()
    save_task_fig(fig, "综合", "quick_vs_full.png")


# ─── 墙体同族分组 ────────────────────────────────────────────────────────────

def plot_wall_family(results, family_name, strategies, family_cn):
    fr = [r for r in results if r.get("strategy") in strategies]
    if not fr:
        return

    def variant_label(r):
        p = r.get("params", {})
        parts = []
        for k in sorted(p.keys()):
            v = p[k]
            parts.append(f"{k[0]}{v:.2f}" if isinstance(v, float) else f"{k[0]}{v}")
        return "_".join(parts)

    grp = {}
    for r in fr:
        grp.setdefault(r["strategy"], []).append(r)
    ss = [s for s in strategies if s in grp]
    palette = {s: c for s, c in zip(strategies, [C_BLUE, "#e8883a", C_GREEN, C_RED])}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIG_W * 1.6, FIG_W * 0.55))
    for axi, (y_key, y_label) in enumerate([("avg_ms", "平均耗时 (ms)"), ("wall_ratio", "墙体占比 (%)")]):
        ax = [ax1, ax2][axi]
        x_pos = 0
        tick_pos, tick_labels = [], []
        all_x, all_y, all_c = [], [], []
        sep_positions = []
        for si, s_name in enumerate(ss):
            variants = sorted(grp[s_name], key=lambda r: r.get(y_key, 0))
            c = palette.get(s_name, C_GRAY)
            for v in variants:
                label = variant_label(v)
                y_val = v.get("extra", {}).get(y_key, v.get(y_key, 0)) if y_key == "wall_ratio" else v.get(y_key, 0)
                all_x.append(x_pos)
                all_y.append(y_val)
                all_c.append(c)
                tick_pos.append(x_pos)
                tick_labels.append(label)
                x_pos += 1
            sep_positions.append(x_pos)
            x_pos += 0.5

        ax.bar(all_x, all_y, color=all_c, width=0.6, zorder=3)
        for xi, yi in zip(all_x, all_y):
            fmt = f"{yi:.1f}" if y_key == "avg_ms" else f"{yi:.0f}%"
            ax.text(xi, yi + max(all_y) * 0.02, fmt, ha="center", va="bottom",
                    fontsize=6, color=C_GRAY)
        prev = 0
        for si, s_name in enumerate(ss):
            sep = sep_positions[si] if si < len(sep_positions) else len(all_x)
            mid = (prev + sep - 0.5) / 2 if sep > prev else prev
            ax.text(mid, ax.get_ylim()[1] * 1.08, s_name, ha="center", va="bottom",
                    fontsize=8, fontweight="bold", color=palette.get(s_name, C_GRAY))
            if si < len(ss) - 1:
                ax.axvline(x=sep - 0.25, color="#ccc", linewidth=0.5, linestyle=":")
            prev = sep - 0.5
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_labels, fontsize=6, rotation=60, ha="right")
        ax.set_ylabel(y_label, fontsize=9)
        style_ax(ax)
    fig.tight_layout()
    save_task_fig(fig, "墙体", f"family_{family_name}.png")


def plot_wall():
    results = load_bench_data("wall")
    if not results:
        return print("  [跳过]")
    filtered = filter_faster_grid_qt(results)
    if len(filtered) < len(results):
        print(f"    同族去重：剔除 {len(results) - len(filtered)}")
    plot_speed_bar(filtered, "墙体")
    plot_dual_axis(filtered, "墙体", "wall_ratio", "墙体占比 (%)")
    plot_scatter(filtered, "墙体", "wall_ratio", "墙体占比 (%)")
    plot_summary_table(filtered, "墙体", ["wall_ratio"])
    plot_wall_family(results, "pca_grid", ["cc_pca_grid", "adapt_pca_grid", "nrm_pca_grid", "seq_pca_grid"], "PCA 网格族")
    plot_wall_family(results, "pca_qt", ["cc_pca_qt", "dbscan_pca_qt", "nrm_pca_qt"], "PCA 四叉树族")
    plot_wall_family(results, "l2_grid", ["cc_l2_grid", "adapt_l2_grid", "ransac_l2_grid", "seq_l2_grid", "nrm_l2_grid"], "L2 网格族")
    plot_wall_family(results, "l2_qt", ["cc_l2_qt", "ransac_l2_qt", "dbscan_l2_qt", "dbscan_l2_qt_dif", "nrm_l2_qt"], "L2 四叉树族")


# ─── 各任务入口 ──────────────────────────────────────────────────────────────

def plot_ground():
    results = load_bench_data("ground")
    if not results:
        return print("  [跳过]")
    plot_speed_bar(results, "地面")
    plot_dual_axis(results, "地面", "ground_ratio", "地面占比 (%)")
    plot_scatter(results, "地面", "ground_ratio", "地面占比 (%)")
    plot_summary_table(results, "地面", ["ground_ratio"])


def plot_cluster():
    results = load_bench_data("cluster")
    if not results:
        return print("  [跳过]")
    plot_speed_bar(results, "聚类")
    plot_dual_axis(results, "聚类", "avg_clusters", "平均聚类数", "{:.1f}")
    plot_scatter(results, "聚类", "avg_clusters", "平均聚类数")
    plot_summary_table(results, "聚类", ["avg_clusters", "avg_noise"])


def plot_denoise():
    results = load_bench_data("denoise")
    if not results:
        return print("  [跳过]")
    plot_speed_bar(results, "降噪")
    plot_dual_axis(results, "降噪", "retention_pct", "点保留率 (%)")
    plot_scatter(results, "降噪", "retention_pct", "点保留率 (%)")
    plot_summary_table(results, "降噪", ["retention_pct"])


# ─── 主入口 ──────────────────────────────────────────────────────────────────

def main():
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True)
    print(f"输出目录: {OUT_DIR}/\n")

    for cn, fn in [("地面", plot_ground), ("墙体", plot_wall),
                   ("聚类", plot_cluster), ("降噪", plot_denoise)]:
        print(f"  [{cn}]")
        fn()
        print()

    print("  [综合对比]")
    plot_quick_vs_full()
    print()

    total = len(list(OUT_DIR.rglob("*.png")))
    print(f"完成！共 {total} 张图")


if __name__ == "__main__":
    main()
