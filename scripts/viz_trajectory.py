"""BEV 俯视轨迹图 — 按 track ID 绘制目标运动轨迹

用法:
    python scripts/viz_trajectory.py                          # 自动找最新的输出
    python scripts/viz_trajectory.py output/pipeline_xxx/pipeline.jsonl
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ─── 论文级样式 ───────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei"],
    "font.family": "sans-serif",
    "axes.unicode_minus": False,
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def find_latest_jsonl() -> Path:
    """自动在 output/ 下找最新的 pipeline_xxx/pipeline.jsonl"""
    output_dir = Path("output")
    if not output_dir.exists():
        raise FileNotFoundError("output/ 目录不存在")
    dirs = sorted(output_dir.glob("pipeline_*"), reverse=True)
    for d in dirs:
        j = d / "pipeline.jsonl"
        if j.exists():
            return j
    raise FileNotFoundError("未找到 pipeline.jsonl 文件")


def load_data(path: Path):
    """读取 JSONL，返回帧列表"""
    frames = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                frames.append(json.loads(line))
    return frames


def extract_tracks(frames):
    """按 track ID 分组提取轨迹数据"""
    tracks = {}  # id -> {frames: [], xs: [], ys: [], cls: str, speeds: []}
    for frm in frames:
        frame_idx = frm["frame"]
        for t in frm.get("targets", []):
            tid = t["id"]
            if tid not in tracks:
                tracks[tid] = {
                    "frames": [],
                    "xs": [],
                    "ys": [],
                    "speeds": [],
                    "classification": t["classification"],
                    "class_type": t["class_type"],
                }
            tracks[tid]["frames"].append(frame_idx)
            tracks[tid]["xs"].append(t["x"])
            tracks[tid]["ys"].append(t["y"])
            tracks[tid]["speeds"].append(t["speed"])
    return tracks


# ─── 调色板 ───────────────────────────────────────────────────────────────────
COLORS = [
    "#E63946", "#457B9D", "#2A9D8F", "#E9C46A", "#F4A261",
    "#6D597A", "#B56576", "#219EBC", "#023047", "#8ECAE6",
    "#D62828", "#003049", "#669BBC", "#1B4332", "#95D5B2",
    "#7B2CBF", "#9D4EDD", "#FF6B6B", "#4ECDC4", "#FFE66D",
]


def plot_trajectory(frames, tracks, output_path: Path):
    fig, ax = plt.subplots(figsize=(9, 8))

    # 轨道地图：用所有 x/y 确定范围
    all_x = [x for t in tracks.values() for x in t["xs"]]
    all_y = [y for t in tracks.values() for y in t["ys"]]
    if not all_x:
        print("[WARN] 没有轨迹数据")
        ax.set_title("BEV 轨迹图（无数据）")
        fig.savefig(output_path)
        plt.close(fig)
        return

    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    x_range = max(x_max - x_min, 1.0)
    y_range = max(y_max - y_min, 1.0)
    margin = max(x_range, y_range) * 0.15
    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(y_min - margin, y_max + margin)
    ax.set_aspect("equal")

    # 绘制每条轨迹
    for i, (tid, trk) in enumerate(sorted(tracks.items())):
        color = COLORS[i % len(COLORS)]
        xs = trk["xs"]
        ys = trk["ys"]
        clss = trk["classification"]
        ctype = trk["class_type"]
        avg_spd = np.mean(trk["speeds"]) if trk["speeds"] else 0.0

        # 主线
        ax.plot(xs, ys, "-", color=color, linewidth=1.2, alpha=0.8, zorder=3)

        # 起点 (圆点) + 终点 (箭头)
        ax.scatter(xs[0], ys[0], color=color, s=40, marker="o", zorder=4, edgecolors="white", linewidth=0.5)
        ax.scatter(xs[-1], ys[-1], color=color, s=50, marker="s", zorder=4, edgecolors="white", linewidth=0.5)

        # 速度渐变 overlay（用 scatter 的颜色深浅表示速度大小）
        speeds = trk["speeds"]
        if len(xs) > 2:
            sc = ax.scatter(
                xs, ys, c=speeds, cmap="plasma", s=12,
                alpha=0.6, zorder=2, vmin=0, vmax=max(speeds) if speeds else 1,
            )

        # 轨迹中点标注 ID
        mid_idx = len(xs) // 2
        ax.annotate(
            str(tid), (xs[mid_idx], ys[mid_idx]),
            fontsize=7, color=color, fontweight="bold",
            ha="center", va="bottom",
            bbox=dict(boxstyle="round,pad=0.1", fc="white", ec=color, alpha=0.7, linewidth=0.5),
        )

        # 图例行
        label = f"ID {tid} | {ctype} ({clss}) | {avg_spd:.2f}m/s"
        ax.plot([], [], "-", color=color, linewidth=2, label=label)

    # 速度 colorbar (如果画了 scatter)
    if any(len(t["xs"]) > 2 for t in tracks.values()):
        cbar = fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.02)
        cbar.set_label("速度 (m/s)", fontsize=10)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("BEV 俯视轨迹图", fontweight="bold", pad=10)

    # 图例放在图外右侧
    ax.legend(
        loc="upper left", bbox_to_anchor=(1.02, 1),
        framealpha=0.9, edgecolor="#ccc", fontsize=7,
        title="Track ID", title_fontsize=8,
    )

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  [OK] BEV 轨迹图 → {output_path}")


# ─── 辅助图: 速度随时间变化（每 ID 一条线）───────────────────────────────────────

def plot_speed_curves(tracks, n_frames, output_path: Path):
    fig, ax = plt.subplots(figsize=(10, 5))

    has_data = False
    for i, (tid, trk) in enumerate(sorted(tracks.items())):
        color = COLORS[i % len(COLORS)]
        frames = trk["frames"]
        speeds = trk["speeds"]
        if len(frames) < 2:
            continue
        has_data = True
        # 平滑曲线
        ax.plot(frames, speeds, "-o", color=color, linewidth=1.0, markersize=3,
                label=f"ID {tid} ({trk['class_type']}, {trk['classification']})")

        # 均值虚线
        avg = np.mean(speeds)
        ax.axhline(y=avg, color=color, linestyle=":", alpha=0.4, linewidth=0.8)
        ax.text(n_frames * 0.98, avg, f"{avg:.2f}", fontsize=6, color=color, alpha=0.6,
                va="center", ha="left")

    if not has_data:
        ax.set_title("速度曲线（无数据）")
    else:
        ax.set_xlabel("帧号")
        ax.set_ylabel("速度 (m/s)")
        ax.set_title("各目标速度随时间变化", fontweight="bold")
        ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1),
                  title="Track ID", title_fontsize=8)
        ax.set_xlim(0, n_frames)
        ax.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  [OK] 速度曲线图 → {output_path}")


# ─── 统计分布图 ───────────────────────────────────────────────────────────────

def plot_stats(frames, n_frames, output_path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    frame_indices = [f["frame"] for f in frames]
    n_targets = [f["stats"]["n_targets"] for f in frames]
    n_person = [f["stats"]["n_person"] for f in frames]
    n_clusters = [f["stats"]["n_clusters"] for f in frames]

    # 左: 目标数变化
    ax = axes[0]
    ax.plot(frame_indices, n_targets, "-", color="#457B9D", linewidth=0.8, label="总目标")
    ax.plot(frame_indices, n_person, "-", color="#E63946", linewidth=0.8, label="行人")
    ax.fill_between(frame_indices, n_person, alpha=0.15, color="#E63946")
    ax.set_xlabel("帧号")
    ax.set_ylabel("目标数")
    ax.set_title("每帧检测目标数")
    ax.legend(fontsize=8)
    ax.set_xlim(0, n_frames)
    ax.set_ylim(bottom=0)

    # 中: 聚类数
    ax = axes[1]
    ax.plot(frame_indices, n_clusters, "-", color="#2A9D8F", linewidth=0.8)
    ax.fill_between(frame_indices, n_clusters, alpha=0.15, color="#2A9D8F")
    ax.set_xlabel("帧号")
    ax.set_ylabel("聚类数")
    ax.set_title("每帧聚类数")
    ax.set_xlim(0, n_frames)
    ax.set_ylim(bottom=0)

    # 右: 分类分布堆叠
    ax = axes[2]
    cats = ["moving", "static", "movable", "floating"]
    colors_cat = ["#E63946", "#457B9D", "#2A9D8F", "#E9C46A"]
    bottom = np.zeros(len(frame_indices))
    for cat, color in zip(cats, colors_cat):
        values = np.array([f["stats"].get(f"n_{cat}", 0) for f in frames])
        if values.sum() > 0:
            ax.bar(frame_indices, values, bottom=bottom, width=1.0,
                   color=color, label=cat, alpha=0.8, edgecolor="none")
            bottom += values
    ax.set_xlabel("帧号")
    ax.set_ylabel("目标数")
    ax.set_title("目标分类分布")
    ax.legend(fontsize=7, loc="upper left")
    ax.set_xlim(0, n_frames)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  [OK] 统计分布图 → {output_path}")


# ─── 延迟时间图 ───────────────────────────────────────────────────────────────

def plot_latency(frames, n_frames, output_path: Path):
    fig, ax = plt.subplots(figsize=(10, 4.5))

    frame_indices = [f["frame"] for f in frames]
    stages = ["join", "fuse", "io", "tracker"]
    colors_stage = ["#A8DADC", "#457B9D", "#1D3557", "#E63946"]
    bottom = np.zeros(len(frame_indices))

    for stage, color in zip(stages, colors_stage):
        values = np.array([f["stages_ms"][stage] for f in frames])
        if values.sum() > 0:
            ax.bar(frame_indices, values, bottom=bottom, width=1.0,
                   color=color, label=stage, alpha=0.85, edgecolor="none")
            bottom += values

    total = [f["elapsed_ms"] for f in frames]
    ax.plot(frame_indices, total, "-", color="black", linewidth=0.6, alpha=0.5, label="总计")

    ax.set_xlabel("帧号")
    ax.set_ylabel("耗时 (ms)")
    ax.set_title("管线各阶段延迟分解")
    ax.legend(fontsize=8)
    ax.set_xlim(0, n_frames)
    ax.set_ylim(bottom=0)

    # 均值线
    avg_total = np.mean(total)
    ax.axhline(y=avg_total, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.text(n_frames * 0.98, avg_total, f"均值 {avg_total:.0f}ms",
            fontsize=8, color="gray", va="bottom", ha="right")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  [OK] 延迟分解图 → {output_path}")


# ─── 主入口 ───────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) > 1:
        jsonl_path = Path(sys.argv[1])
    else:
        jsonl_path = find_latest_jsonl()

    if not jsonl_path.exists():
        print(f"[ERR] 文件不存在: {jsonl_path}")
        sys.exit(1)

    out_dir = jsonl_path.parent

    print(f"读取数据: {jsonl_path}")
    frames = load_data(jsonl_path)
    print(f"  {len(frames)} 帧")

    n_frames = max(f["frame"] for f in frames) + 1 if frames else 0
    tracks = extract_tracks(frames)
    print(f"  {len(tracks)} 条轨迹")

    # 生成图表
    print()
    plot_trajectory(frames, tracks, out_dir / "fig_trajectory_bev.png")
    plot_speed_curves(tracks, n_frames, out_dir / "fig_speed_curves.png")
    plot_stats(frames, n_frames, out_dir / "fig_stats.png")
    plot_latency(frames, n_frames, out_dir / "fig_latency.png")

    print()
    print(f"所有图表已保存至: {out_dir.resolve()}")
    print(f"  fig_trajectory_bev.png  — BEV 轨迹图")
    print(f"  fig_speed_curves.png    — 速度曲线")
    print(f"  fig_stats.png           — 统计分布")
    print(f"  fig_latency.png         — 延迟分解")


if __name__ == "__main__":
    main()
