"""BEV 俯视轨迹图 — 按 track ID 绘制目标运动轨迹

用法:
    .venv/Scripts/python.exe scripts/viz_trajectory.py
    .venv/Scripts/python.exe scripts/viz_trajectory.py output/pipeline_xxx/pipeline.jsonl
"""
import json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.chart_style import savefig, C_BLUE, C_RED, C_GREEN, C_YELLOW, COLORS_10

COLORS = COLORS_10 + [
    "#D62828", "#003049", "#669BBC", "#1B4332", "#95D5B2",
    "#7B2CBF", "#9D4EDD", "#FF6B6B", "#4ECDC4", "#FFE66D",
]


def find_latest_jsonl():
    output_dir = Path("output")
    if not output_dir.exists():
        raise FileNotFoundError("output/ 目录不存在")
    dirs = sorted(output_dir.glob("pipeline_*"), reverse=True)
    for d in dirs:
        j = d / "pipeline.jsonl"
        if j.exists():
            return j
    raise FileNotFoundError("未找到 pipeline.jsonl 文件")


def load_data(path):
    frames = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                frames.append(json.loads(line))
    return frames


def extract_tracks(frames):
    tracks = {}
    for frm in frames:
        fi = frm["frame"]
        for t in frm.get("targets", []):
            tid = t["id"]
            if tid not in tracks:
                tracks[tid] = {"frames": [], "xs": [], "ys": [], "speeds": [],
                               "classification": t["classification"], "class_type": t["class_type"]}
            tracks[tid]["frames"].append(fi)
            tracks[tid]["xs"].append(t["x"])
            tracks[tid]["ys"].append(t["y"])
            tracks[tid]["speeds"].append(t["speed"])
    return tracks


def plot_trajectory(frames, tracks, output_path):
    fig, ax = plt.subplots(figsize=(9, 8))

    all_x = [x for t in tracks.values() for x in t["xs"]]
    all_y = [y for t in tracks.values() for y in t["ys"]]
    if not all_x:
        savefig(fig, output_path)
        return

    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    margin = max(x_max - x_min, y_max - y_min, 1.0) * 0.15
    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(y_min - margin, y_max + margin)
    ax.set_aspect("equal")

    for i, (tid, trk) in enumerate(sorted(tracks.items())):
        color = COLORS[i % len(COLORS)]
        xs, ys = trk["xs"], trk["ys"]
        avg_spd = np.mean(trk["speeds"]) if trk["speeds"] else 0.0
        ax.plot(xs, ys, "-", color=color, linewidth=1.2, alpha=0.8, zorder=3)
        ax.scatter(xs[0], ys[0], color=color, s=40, marker="o", zorder=4, edgecolors="white", linewidth=0.5)
        ax.scatter(xs[-1], ys[-1], color=color, s=50, marker="s", zorder=4, edgecolors="white", linewidth=0.5)

        if len(xs) > 2:
            sc = ax.scatter(xs, ys, c=trk["speeds"], cmap="plasma", s=12,
                           alpha=0.6, zorder=2, vmin=0, vmax=max(trk["speeds"]) or 1)

        mid_idx = len(xs) // 2
        ax.annotate(str(tid), (xs[mid_idx], ys[mid_idx]), fontsize=7, color=color,
                    fontweight="bold", ha="center", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", ec=color, alpha=0.7, linewidth=0.5))
        ax.plot([], [], "-", color=color, linewidth=2,
                label=f"ID {tid} | {trk['class_type']} ({trk['classification']}) | {avg_spd:.2f}m/s")

    if any(len(t["xs"]) > 2 for t in tracks.values()):
        cbar = fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.02)
        cbar.set_label("速度 (m/s)", fontsize=10)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1),
              framealpha=0.9, edgecolor="#ccc", fontsize=7,
              title="Track ID", title_fontsize=8)

    savefig(fig, output_path)
    print(f"  [OK] 轨迹图 → {output_path}")


def plot_speed_curves(tracks, n_frames, output_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    has_data = False
    for i, (tid, trk) in enumerate(sorted(tracks.items())):
        color = COLORS[i % len(COLORS)]
        frames = trk["frames"]
        speeds = trk["speeds"]
        if len(frames) < 2:
            continue
        has_data = True
        ax.plot(frames, speeds, "-o", color=color, linewidth=1.0, markersize=3,
                label=f"ID {tid} ({trk['class_type']}, {trk['classification']})")
        avg = np.mean(speeds)
        ax.axhline(y=avg, color=color, linestyle=":", alpha=0.4, linewidth=0.8)
        ax.text(n_frames * 0.98, avg, f"{avg:.2f}", fontsize=6, color=color, alpha=0.6, va="center", ha="left")

    if has_data:
        ax.set_xlabel("帧号")
        ax.set_ylabel("速度 (m/s)")
        ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1), title="Track ID", title_fontsize=8)
        ax.set_xlim(0, n_frames)
        ax.set_ylim(bottom=0)

    savefig(fig, output_path)
    print(f"  [OK] 速度曲线 → {output_path}")


def plot_stats(frames, n_frames, output_path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    fi = [f["frame"] for f in frames]
    nt = [f["stats"]["n_targets"] for f in frames]
    np_ = [f["stats"]["n_person"] for f in frames]
    nc = [f["stats"]["n_clusters"] for f in frames]

    ax = axes[0]
    ax.plot(fi, nt, "-", color=C_BLUE, linewidth=0.8, label="总目标")
    ax.plot(fi, np_, "-", color=C_RED, linewidth=0.8, label="行人")
    ax.fill_between(fi, np_, alpha=0.15, color=C_RED)
    ax.set_xlabel("帧号"); ax.set_ylabel("目标数")
    ax.legend(fontsize=8); ax.set_xlim(0, n_frames); ax.set_ylim(bottom=0)

    ax = axes[1]
    ax.plot(fi, nc, "-", color=C_GREEN, linewidth=0.8)
    ax.fill_between(fi, nc, alpha=0.15, color=C_GREEN)
    ax.set_xlabel("帧号"); ax.set_ylabel("聚类数")
    ax.set_xlim(0, n_frames); ax.set_ylim(bottom=0)

    ax = axes[2]
    cats = ["moving", "static", "movable", "floating"]
    colors_cat = [C_RED, C_BLUE, C_GREEN, C_YELLOW]
    bottom = np.zeros(len(fi))
    for cat, color in zip(cats, colors_cat):
        values = np.array([f["stats"].get(f"n_{cat}", 0) for f in frames])
        if values.sum() > 0:
            ax.bar(fi, values, bottom=bottom, width=1.0, color=color, label=cat, alpha=0.8, edgecolor="none")
            bottom += values
    ax.set_xlabel("帧号"); ax.set_ylabel("目标数")
    ax.legend(fontsize=7, loc="upper left"); ax.set_xlim(0, n_frames)

    savefig(fig, output_path)
    print(f"  [OK] 统计图 → {output_path}")


def plot_latency(frames, n_frames, output_path):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    fi = [f["frame"] for f in frames]
    stages = ["join", "fuse", "io", "tracker"]
    colors_stage = ["#A8DADC", C_BLUE, "#1D3557", C_RED]
    bottom = np.zeros(len(fi))
    for stage, color in zip(stages, colors_stage):
        values = np.array([f["stages_ms"][stage] for f in frames])
        if values.sum() > 0:
            ax.bar(fi, values, bottom=bottom, width=1.0, color=color, label=stage, alpha=0.85, edgecolor="none")
            bottom += values
    total = [f["elapsed_ms"] for f in frames]
    ax.plot(fi, total, "-", color="black", linewidth=0.6, alpha=0.5, label="总计")
    ax.set_xlabel("帧号"); ax.set_ylabel("耗时 (ms)")
    ax.legend(fontsize=8); ax.set_xlim(0, n_frames); ax.set_ylim(bottom=0)
    avg_total = np.mean(total)
    ax.axhline(y=avg_total, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.text(n_frames * 0.98, avg_total, f"均值 {avg_total:.0f}ms", fontsize=8, color="gray", va="bottom", ha="right")

    savefig(fig, output_path)
    print(f"  [OK] 延迟图 → {output_path}")


def main():
    if len(sys.argv) > 1:
        jsonl_path = Path(sys.argv[1])
    else:
        jsonl_path = find_latest_jsonl()
    if not jsonl_path.exists():
        print(f"[ERR] 文件不存在: {jsonl_path}")
        sys.exit(1)

    out_dir = jsonl_path.parent
    frames = load_data(jsonl_path)
    n_frames = max(f["frame"] for f in frames) + 1 if frames else 0
    tracks = extract_tracks(frames)
    print(f"读取数据: {jsonl_path} ({len(frames)} 帧, {len(tracks)} 条轨迹)")

    plot_trajectory(frames, tracks, out_dir / "fig_trajectory_bev.png")
    plot_speed_curves(tracks, n_frames, out_dir / "fig_speed_curves.png")
    plot_stats(frames, n_frames, out_dir / "fig_stats.png")
    plot_latency(frames, n_frames, out_dir / "fig_latency.png")
    print(f"\n所有图表保存至: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
