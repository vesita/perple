"""BevEdLines 参数扫描分析图。

用法：
  uv run python scripts/bev_edlines_analysis.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── 路径 ──────────────────────────────────────────────────
JSON_PATH = Path("output/bench/bev_edlines_bench/results.json")
OUT_DIR   = Path("output/bench/bev_edlines_bench")

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"],
    "axes.unicode_minus": False,
})


def load() -> list[dict]:
    with open(JSON_PATH) as f:
        return json.load(f)


def plot_latency(data: list[dict]):
    """耗时分解堆叠图：EDlines + 网格聚类"""
    labels = [d["combo"] for d in data]
    ed_ms   = np.array([d["avg_edlines_ms"] for d in data])
    db_ms   = np.array([d["avg_cluster_ms"] for d in data])
    total_ms = ed_ms + db_ms

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    b1 = ax.bar(x, ed_ms, label="EDlines", color="#3498db")
    b2 = ax.bar(x, db_ms, bottom=ed_ms, label="网格聚类", color="#e74c3c")

    for i, t in enumerate(total_ms):
        ax.text(i, t + 8, f"{t:.0f}ms", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("耗时 (ms)")
    ax.set_title("BevEdLines 耗时分解")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "latency.png", dpi=150)
    plt.close(fig)
    print(f"  → latency.png")


def plot_wall_pts(data: list[dict]):
    """墙面点捕获数 + 剩余点数"""
    labels = [d["combo"] for d in data]
    wall_pts = np.array([d["avg_wall_pts"] for d in data])
    remain   = np.array([d["avg_cluster_pts"] + d["avg_noise"] for d in data])

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    b1 = ax.bar(x, wall_pts, label="墙面点", color="#e74c3c")
    b2 = ax.bar(x, remain, bottom=wall_pts, label="剩余(聚类+噪声)", color="#95a5a6")

    for i in range(len(labels)):
        total = wall_pts[i] + remain[i]
        pct = wall_pts[i] / total * 100
        ax.text(i, total + 150, f"{pct:.0f}%", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("点数")
    ax.set_title("墙面捕获 vs 剩余点")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "wall_capture.png", dpi=150)
    plt.close(fig)
    print(f"  → wall_capture.png")


def plot_clusters(data: list[dict]):
    """障碍簇数 + 簇点数 + 噪声点"""
    labels = [d["combo"] for d in data]
    n_clusters = np.array([d["avg_clusters"] for d in data])
    cluster_pts = np.array([d["avg_cluster_pts"] for d in data])
    noise = np.array([d["avg_noise"] for d in data])

    x = np.arange(len(labels))
    w = 0.25
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax2 = ax1.twinx()

    b1 = ax1.bar(x - w, n_clusters, w, label="障碍簇数", color="#2ecc71")
    ax1.set_ylabel("簇数")

    b2 = ax2.bar(x, cluster_pts, w, label="簇点数", color="#3498db")
    b3 = ax2.bar(x + w, noise, w, label="噪声点", color="#e67e22")
    ax2.set_ylabel("点数")

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=25, ha="right")
    ax1.set_title("后聚类结果")
    lines = [b1, b2, b3]
    labels_list = ["障碍簇数", "簇点数", "噪声点"]
    ax1.legend(lines, labels_list, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "clusters.png", dpi=150)
    plt.close(fig)
    print(f"  → clusters.png")


def plot_scatter(data: list[dict]):
    """剩余点数 vs 聚类耗时（每帧散点）"""
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"d0.06": "#3498db", "d0.08": "#e74c3c"}
    markers = {"g0.050": "o", "g0.080": "s", "g0.100": "^"}

    for d in data:
        prefix = d["combo"][:5]   # d0.06 / d0.08
        grad = "g" + d["combo"].split("_g")[1].split("_")[0]
        color = colors.get(prefix, "#888")
        marker = markers.get(grad, "o")

        for f in d["frames"]:
            remain = f["cluster_pts"] + f["noise_pts"]
            ax.scatter(remain, f["cluster_ms"], c=color, marker=marker,
                       alpha=0.7, s=40, edgecolors="white", linewidth=0.5)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#3498db", markersize=8, label="d=0.06"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#e74c3c", markersize=8, label="d=0.08"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#888", markersize=8, label="g=0.080"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#888", markersize=8, label="g=0.100"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9)
    ax.set_xlabel("剩余点数 (簇点 + 噪声)")
    ax.set_ylabel("聚类耗时 (ms)")
    ax.set_title("剩余点数 vs 聚类耗时")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "cluster_scatter.png", dpi=150)
    plt.close(fig)
    print(f"  → cluster_scatter.png")


def main():
    data = load()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("生成 BevEdLines 分析图...")
    plot_latency(data)
    plot_wall_pts(data)
    plot_clusters(data)
    plot_scatter(data)
    print(f"\n全部输出至: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
