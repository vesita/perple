"""
Bench 性能可视化脚本

用法：python scripts/bench_viz.py <stats.json> [--top N]

从 stats.json 读取 bench 统计数据，生成以下图表：
1. 条形图 + 误差棒（策略 vs 平均耗时 ± 标准差）
2. 箱线图（逐帧耗时分布）
3. 逐帧折线图（每帧耗时趋势）
4. CDF 曲线（累积分布）
5. 摘要表格（CSV）

输出到 stats.json 同级的 viz/ 目录。
"""

import json
import sys
import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 使用英文标签避免 CJK 字体问题
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial"],
})


def load_stats(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def short_name(name: str, max_len: int = 28) -> str:
    """截断过长的策略名"""
    return name if len(name) <= max_len else name[: max_len - 1] + "…"


def sorted_by_mean(stats: list[dict]) -> list[dict]:
    return sorted(stats, key=lambda s: np.mean(s["frame_times"]) if s["frame_times"] else 0)


def plot_bar(stats: list[dict], out_dir: str, top: int | None = None):
    """条形图：策略 vs 平均耗时 ± 标准差"""
    data = sorted_by_mean(stats)
    if top:
        data = data[:top]

    names = [short_name(s["name"]) for s in data]
    means = [np.mean(s["frame_times"]) for s in data]
    stds = [np.std(s["frame_times"]) for s in data]

    fig, ax = plt.subplots(figsize=(10, max(6, len(data) * 0.35)))
    y_pos = np.arange(len(data))

    colors = ["#e74c3c" if m > 100 else "#3498db" for m in means]
    bars = ax.barh(y_pos, means, xerr=stds, align="center", color=colors,
                   capsize=3, error_kw={"linewidth": 0.8})

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Avg Latency (ms)")
    ax.set_title("Strategy Latency Comparison (mean ± std)")

    # 100ms 阈值线
    ax.axvline(x=100, color="#e74c3c", linestyle="--", linewidth=1, alpha=0.7, label="100ms threshold")
    ax.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(out_dir, "bar_latency.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Bar chart → {path}")


def plot_box(stats: list[dict], out_dir: str, top: int | None = None):
    """箱线图：逐帧耗时分布"""
    data = sorted_by_mean(stats)
    if top:
        data = data[:top]

    names = [short_name(s["name"]) for s in data]
    times = [np.array(s["frame_times"]) for s in data]

    fig, ax = plt.subplots(figsize=(10, max(6, len(data) * 0.35)))
    bp = ax.boxplot(times, vert=False, patch_artist=True, widths=0.6,
                    boxprops=dict(facecolor="#3498db", alpha=0.7),
                    medianprops=dict(color="#e74c3c", linewidth=1.5),
                    flierprops=dict(marker="o", markersize=3, alpha=0.5))

    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Latency (ms)")
    ax.set_title("Per-Frame Latency Distribution (Box Plot)")
    ax.axvline(x=100, color="#e74c3c", linestyle="--", linewidth=1, alpha=0.7)

    plt.tight_layout()
    path = os.path.join(out_dir, "box_latency.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Box plot → {path}")


def plot_per_frame(stats: list[dict], out_dir: str, top: int | None = None):
    """逐帧折线图"""
    data = sorted_by_mean(stats)
    if top:
        data = data[:top]

    fig, ax = plt.subplots(figsize=(12, 6))

    # 色板
    cmap = plt.cm.tab20
    for i, s in enumerate(data):
        times = s["frame_times"]
        ax.plot(range(len(times)), times, label=short_name(s["name"], 20),
                color=cmap(i / max(len(data) - 1, 1)), linewidth=0.8, alpha=0.8)

    ax.set_xlabel("Frame Index")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Per-Frame Latency Trend")
    ax.axhline(y=100, color="#e74c3c", linestyle="--", linewidth=1, alpha=0.7, label="100ms threshold")

    if len(data) <= 12:
        ax.legend(fontsize=7, loc="upper right")

    plt.tight_layout()
    path = os.path.join(out_dir, "per_frame.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Per-frame chart → {path}")


def plot_cdf(stats: list[dict], out_dir: str, top: int | None = None):
    """CDF 曲线"""
    data = sorted_by_mean(stats)
    if top:
        data = data[:top]

    fig, ax = plt.subplots(figsize=(10, 6))

    cmap = plt.cm.tab20
    for i, s in enumerate(data):
        times = np.sort(s["frame_times"])
        cdf = np.arange(1, len(times) + 1) / len(times)
        ax.plot(times, cdf, label=short_name(s["name"], 20),
                color=cmap(i / max(len(data) - 1, 1)), linewidth=0.8, alpha=0.8)

    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title("Latency CDF")
    ax.axvline(x=100, color="#e74c3c", linestyle="--", linewidth=1, alpha=0.7, label="100ms threshold")
    ax.set_xlim(left=0)

    if len(data) <= 12:
        ax.legend(fontsize=7, loc="lower right")

    plt.tight_layout()
    path = os.path.join(out_dir, "cdf_latency.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  CDF curve → {path}")


def save_summary(stats: list[dict], out_dir: str):
    """摘要表格 CSV"""
    rows = []
    for s in stats:
        times = np.array(s["frame_times"]) if s["frame_times"] else np.array([0])
        rows.append({
            "name": s["name"],
            "frames": s["frame_count"],
            "mean_ms": round(np.mean(times), 2),
            "std_ms": round(np.std(times), 2),
            "p50_ms": round(np.percentile(times, 50), 2),
            "p95_ms": round(np.percentile(times, 95), 2),
            "p99_ms": round(np.percentile(times, 99), 2),
            "min_ms": round(np.min(times), 2),
            "max_ms": round(np.max(times), 2),
        })

    # 按 mean_ms 排序
    rows.sort(key=lambda r: r["mean_ms"])

    path = os.path.join(out_dir, "summary.csv")
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Summary table → {path}")


def main():
    if len(sys.argv) < 2:
        print("用法: python scripts/bench_viz.py <stats.json> [--top N]")
        sys.exit(1)

    stats_path = sys.argv[1]
    top = None
    if "--top" in sys.argv:
        idx = sys.argv.index("--top")
        top = int(sys.argv[idx + 1])

    stats = load_stats(stats_path)
    if not stats:
        print("stats.json 为空")
        sys.exit(1)

    # 输出目录：stats.json 同级的 viz/
    out_dir = os.path.join(os.path.dirname(stats_path), "viz")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Total {len(stats)} strategies, output to {out_dir}/\n")

    plot_bar(stats, out_dir, top)
    plot_box(stats, out_dir, top)
    plot_per_frame(stats, out_dir, top)
    plot_cdf(stats, out_dir, top)
    save_summary(stats, out_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
