"""
Bench 可视化 — 论文级图表（无参数标注，简洁美观）

读取 output/bench 下的已有数据，生成适合论文使用的对比图。
覆盖所有 bench 和 pipeline 分析图表类型，输出至独立目录。

用法：
  .venv/Scripts/python.exe scripts/bench_viz.py

输出：output/bench_viz/{地面|墙体|聚类|降噪|综合}/ 下的 PNG 图片
"""

import csv, json, os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ─── 论文级统一配置 ───────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["SimHei", "Microsoft YaHei", "SimSun"],
    "axes.unicode_minus": False,
    "font.size": 9,
    "axes.labelsize": 10.5,
    "axes.titlesize": 14,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
})
FIG_W = 5.9   # A4 文本区宽度 15cm
OUT_DIR = Path("output/bench_viz")
BENCH_DIR = Path("output/bench")
ANALYSIS_DIR = Path("output/bench/analysis")

# 学术配色
C_PRIMARY = "#3b6ba5"
C_SECONDARY = "#e8883a"
C_ACCENT = "#2a9d8f"
C_NEUTRAL = "#6c757d"
C_GREEN = "#27ae60"
C_RED = "#e74c3c"
C_BLUE = "#3498db"


# ─── 数据加载 ─────────────────────────────────────────────────────────────
def load_bench_data(task: str) -> list[dict]:
    """加载 output/bench/{task}/*/info.json"""
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


def load_stata_csv(task: str) -> list[dict] | None:
    """加载 analysis/full/{task}/stats.csv"""
    p = ANALYSIS_DIR / "full" / task / "stats.csv"
    if not p.exists():
        return None
    rows = []
    with open(p, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def pick_median(data, key="avg_ms"):
    """取中值参数的结果"""
    return sorted(data, key=lambda x: x.get(key, 0))[len(data) // 2]


# ─── 网格/四叉树去重：同族取更快 ──────────────────────────
# 策略配对：网格 vs 四叉树，保留较快者
_GRID_QT_PAIRS = [
    ("cc_pca_grid", "cc_pca_qt"),
    ("cc_l2_grid", "cc_l2_qt"),
    ("ransac_l2_grid", "ransac_l2_qt"),
    ("dbscan_l2_qt", "dbscan_l2_qt_dif"),
]


def filter_faster_grid_qt(results):
    """对 grid/qt 配对策略，只保留速度更快的那个。

    按 strategy 分组，对每组配对计算中位耗时，剔除较慢变体的所有条目。
    """
    grp = {}
    for r in results:
        grp.setdefault(r.get("strategy", "?"), []).append(r)

    # 找到每对中要剔除的慢变体
    drop_strategies = set()
    for grid_name, qt_name in _GRID_QT_PAIRS:
        grid_data = grp.get(grid_name)
        qt_data = grp.get(qt_name)
        if grid_data and qt_data:
            grid_med = np.median([r.get("avg_ms", 0) for r in grid_data])
            qt_med = np.median([r.get("avg_ms", 0) for r in qt_data])
            slow_name = qt_name if grid_med <= qt_med else grid_name
            drop_strategies.add(slow_name)

    return [r for r in results if r.get("strategy") not in drop_strategies]


# ─── 辅助绘图 ─────────────────────────────────────────────────────────────
def save_fig(fig, task_cn, name):
    path = OUT_DIR / task_cn / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")


# ─── 1. 速度柱状图（各任务通用） ─────────────────────────────────────────
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
                f"{v:.0f} ms", va="center", fontsize=9, color=C_NEUTRAL)

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("平均耗时 (ms)", fontsize=10.5)
    ax.set_title(f"{task_cn} 策略耗时对比", fontsize=14, fontweight="bold", pad=8)
    ax.invert_yaxis()
    ax.axvline(100, color=C_RED, linestyle="--", linewidth=0.8, alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    save_fig(fig, task_cn, "speed_bar.png")


# ─── 2. 双轴图（耗时 + 质量指标） ──────────────────────────────────────
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
    bars = ax1.bar(x, speeds, 0.45, color=C_PRIMARY, alpha=0.8, zorder=3)
    for bar, v in zip(bars, speeds):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{v:.0f}", ha="center", va="bottom", fontsize=8, color=C_PRIMARY)
    ax1.set_ylabel("耗时 (ms)", fontsize=10.5, color=C_PRIMARY)
    ax1.tick_params(axis="y", labelcolor=C_PRIMARY)
    ax1.set_ylim(0, max(speeds) * 1.35)

    ax2 = ax1.twinx()
    ax2.plot(x, metrics, "o-", color=C_SECONDARY, linewidth=1.5, markersize=5, zorder=5)
    for xi, yi in zip(x, metrics):
        ax2.annotate(y_fmt.format(yi), (xi, yi), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=8, color=C_SECONDARY)
    ax2.set_ylabel(y_label, fontsize=10.5, color=C_SECONDARY)
    ax2.tick_params(axis="y", labelcolor=C_SECONDARY)

    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=8, rotation=20, ha="right")
    ax1.set_title(f"{task_cn}：耗时与{y_label}", fontsize=14, fontweight="bold", pad=8)
    ax1.spines["top"].set_visible(False)
    ax1.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    save_fig(fig, task_cn, "dual_axis.png")


# ─── 3. 散点图（速度 vs 质量，无参数标注） ──────────────────────────────
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
    ax.set_title(f"{task_cn}：速度与{y_label}", fontsize=14, fontweight="bold", pad=8)
    ax.legend(fontsize=8, loc="best", markerscale=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    save_fig(fig, task_cn, "scatter.png")


# ─── 4. 横版汇总表 ──────────────────────────────────────────────────────
def plot_summary_table(results, task_cn, extra_keys=None):
    """表格形式的汇总图"""
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
        pts = grp[n]
        m = pick_median(pts)
        extra_vals = [str(m.get("extra", {}).get(k, "")) for k in extra_keys]
        rows.append([n, str(len(pts)), f"{m.get('avg_ms',0):.1f}",
                     str(m.get("frame_count", 0))] + extra_vals)

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
    ax.set_title(f"{task_cn} 策略汇总", fontsize=14, fontweight="bold", pad=8)
    plt.tight_layout()
    save_fig(fig, task_cn, "summary_table.png")


# ─── 5. 快速 vs 全量对比 ───────────────────────────────────────────────
def plot_quick_vs_full():
    """跨任务对比：从 analysis/ 读取 stats.csv"""
    tasks_cn = {"ground": "地面", "cluster": "聚类", "wall": "墙体", "denoise": "降噪"}
    fig, axes = plt.subplots(2, 2, figsize=(FIG_W, FIG_W * 0.8))
    axes = axes.flatten()

    for ai, (task, cn) in enumerate(tasks_cn.items()):
        ax = axes[ai]
        quick_csv = ANALYSIS_DIR / "quick" / task / "stats.csv"
        full_csv = ANALYSIS_DIR / "full" / task / "stats.csv"

        quick_data = {}
        if quick_csv.exists():
            with open(quick_csv, "r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    quick_data[row["策略"]] = float(row["平均(ms)"])
        full_data = {}
        if full_csv.exists():
            with open(full_csv, "r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    full_data[row["策略"]] = float(row["平均(ms)"])

        common = sorted(set(quick_data) & set(full_data))
        if not common:
            ax.text(0.5, 0.5, "无数据", ha="center", va="center", fontsize=10)
            continue

        x = np.arange(len(common))
        w = 0.3
        ax.bar(x - w / 2, [quick_data[k] for k in common], w,
               color=C_PRIMARY, alpha=0.7, label="快速")
        ax.bar(x + w / 2, [full_data[k] for k in common], w,
               color=C_SECONDARY, alpha=0.7, label="全量")
        ax.set_xticks(x)
        ax.set_xticklabels(common, fontsize=7, rotation=30, ha="right")
        ax.set_title(cn, fontsize=12, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.2)
        if ai == 0:
            ax.legend(fontsize=8)

    fig.suptitle("快速与全量测试耗时对比", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_fig(fig, "综合", "quick_vs_full.png")


# ─── 各任务绘图入口 ───────────────────────────────────────────────────────
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


def plot_wall():
    results = load_bench_data("wall")
    if not results:
        return print("  [跳过]")

    # 同族 grid/qt 取更快者（减少冗余）
    filtered = filter_faster_grid_qt(results)
    if len(filtered) < len(results):
        dropped = len(results) - len(filtered)
        print(f"    同族去重：剔除 {dropped} 个较慢变体（保留 {len(filtered)}）")

    plot_speed_bar(filtered, "墙体")
    plot_dual_axis(filtered, "墙体", "wall_ratio", "墙体占比 (%)")
    plot_scatter(filtered, "墙体", "wall_ratio", "墙体占比 (%)")
    plot_summary_table(filtered, "墙体", ["wall_ratio"])

    # 同族策略单独出图（按索引类型分组）
    print("    同族分组:")
    plot_wall_family(results, "pca_grid", ["cc_pca_grid", "adapt_pca_grid", "nrm_pca_grid", "seq_pca_grid"], "PCA 网格族")
    plot_wall_family(results, "pca_qt", ["cc_pca_qt", "dbscan_pca_qt", "nrm_pca_qt"], "PCA 四叉树族")
    plot_wall_family(results, "l2_grid", ["cc_l2_grid", "adapt_l2_grid", "ransac_l2_grid", "seq_l2_grid", "nrm_l2_grid"], "L2 网格族")
    plot_wall_family(results, "l2_qt", ["cc_l2_qt", "ransac_l2_qt", "dbscan_l2_qt", "dbscan_l2_qt_dif", "nrm_l2_qt"], "L2 四叉树族")


def plot_wall_family(results, family_name, strategies, family_cn):
    """为同族墙体策略生成双轴对比图（速度 + 墙体占比），按族分组"""
    family_results = [r for r in results if r.get("strategy") in strategies]
    if not family_results:
        return print(f"  [跳过 {family_cn}]")

    def variant_label(r):
        p = r.get("params", {})
        parts = []
        for k in sorted(p.keys()):
            v = p[k]
            if isinstance(v, float):
                parts.append(f"{k[0]}{v:.2f}")
            else:
                parts.append(f"{k[0]}{v}")
        return "_".join(parts)

    grp = {}
    for r in family_results:
        grp.setdefault(r["strategy"], []).append(r)

    sorted_strategies = [s for s in strategies if s in grp]
    palette = {s: c for s, c in zip(strategies, [C_PRIMARY, C_SECONDARY, C_ACCENT, C_GREEN])}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIG_W * 1.6, FIG_W * 0.55))

    for axi, (y_key, y_label) in enumerate([("avg_ms", "平均耗时 (ms)"), ("wall_ratio", "墙体占比 (%)")]):
        ax = [ax1, ax2][axi]
        x_pos = 0
        tick_pos = []
        tick_labels = []
        all_x = []
        all_y = []
        all_c = []
        sep_positions = []

        for si, s_name in enumerate(sorted_strategies):
            variants = sorted(grp[s_name], key=lambda r: r.get(y_key, 0))
            c = palette.get(s_name, C_NEUTRAL)
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
                    fontsize=6, color=C_NEUTRAL)

        # Add strategy separators and labels
        prev = 0
        for si, s_name in enumerate(sorted_strategies):
            sep = sep_positions[si] if si < len(sep_positions) else len(all_x)
            mid = (prev + sep - 0.5) / 2 if sep > prev else prev
            ax.text(mid, ax.get_ylim()[1] * 1.08, s_name, ha="center", va="bottom",
                    fontsize=8, fontweight="bold", color=palette.get(s_name, C_NEUTRAL))
            if si < len(sorted_strategies) - 1:
                ax.axvline(x=sep - 0.25, color="#ccc", linewidth=0.5, linestyle=":")
            prev = sep - 0.5  # next starts after gap

        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_labels, fontsize=6, rotation=60, ha="right")
        ax.set_ylabel(y_label, fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.2)

    fig.suptitle(f"墙体 {family_cn} 参数变体对比", fontsize=13, fontweight="bold", y=1.08)
    plt.tight_layout()
    save_fig(fig, "墙体", f"family_{family_name}.png")


def plot_denoise():
    results = load_bench_data("denoise")
    if not results:
        return print("  [跳过]")
    plot_speed_bar(results, "降噪")
    plot_dual_axis(results, "降噪", "retention_pct", "点保留率 (%)")
    plot_scatter(results, "降噪", "retention_pct", "点保留率 (%)")
    plot_summary_table(results, "降噪", ["retention_pct"])


# ─── 主入口 ───────────────────────────────────────────────────────────────
def main():
    # 清理旧数据
    if OUT_DIR.exists():
        import shutil
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True)
    print(f"输出目录: {OUT_DIR}/\n")
    tasks = [("地面", plot_ground), ("墙体", plot_wall),
             ("聚类", plot_cluster), ("降噪", plot_denoise)]

    for cn, fn in tasks:
        print(f"  [{cn}]")
        fn()
        print()

    print("  [综合对比]")
    plot_quick_vs_full()
    print()

    total = len(list(OUT_DIR.rglob("*.png")))
    print(f"完成！共 {total} 张图，300dpi，适合直接插入 Word。")


if __name__ == "__main__":
    main()
