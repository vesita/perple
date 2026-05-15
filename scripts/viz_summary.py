""" 论文实验数据汇总 — 生成统计表 + LaTeX 导出

用法:
    python scripts/viz_summary.py                          # 自动找最新的输出
    python scripts/viz_summary.py output/pipeline_xxx/pipeline.jsonl
"""

import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei"],
    "font.family": "sans-serif",
    "axes.unicode_minus": False,
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 9,
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def find_latest_jsonl() -> Path:
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
    frames = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                frames.append(json.loads(line))
    return frames


def compute_summary_stats(frames):
    """计算全局统计指标"""
    stats = {}
    n_frames = len(frames)

    # 延迟统计
    elapsed = np.array([f["elapsed_ms"] for f in frames])
    stats["latency"] = {
        "mean_ms": float(np.mean(elapsed)),
        "std_ms": float(np.std(elapsed)),
        "min_ms": float(np.min(elapsed)),
        "max_ms": float(np.max(elapsed)),
        "median_ms": float(np.median(elapsed)),
        "fps": 1000.0 / float(np.mean(elapsed)) if np.mean(elapsed) > 0 else 0,
    }

    # 各阶段耗时
    for stage in ["join", "fuse", "io", "tracker"]:
        vals = np.array([f["stages_ms"][stage] for f in frames])
        stats[f"stage_{stage}"] = {
            "mean_ms": float(np.mean(vals)),
            "std_ms": float(np.std(vals)),
        }

    # 每帧目标数
    n_targets = np.array([f["stats"]["n_targets"] for f in frames])
    stats["targets"] = {
        "mean": float(np.mean(n_targets)),
        "std": float(np.std(n_targets)),
        "min": int(np.min(n_targets)),
        "max": int(np.max(n_targets)),
        "total_frames_with_targets": int(np.sum(n_targets > 0)),
    }

    # 分类分布（全量统计）
    total = Counter()
    n_person_frames = 0
    total_person = 0
    for f in frames:
        s = f["stats"]
        total["moving"] += s.get("n_moving", 0)
        total["static"] += s.get("n_static", 0)
        total["movable"] += s.get("n_movable", 0)
        total["floating"] += s.get("n_floating", 0)
        if s.get("n_person", 0) > 0:
            n_person_frames += 1
        total_person += s.get("n_person", 0)

    stats["classification"] = dict(total)
    stats["person"] = {
        "frames_with_person": n_person_frames,
        "person_ratio": n_person_frames / n_frames if n_frames > 0 else 0,
        "total_person_occurrences": total_person,
    }

    # 聚类数
    n_clusters = np.array([f["stats"]["n_clusters"] for f in frames])
    stats["clusters"] = {
        "mean": float(np.mean(n_clusters)),
        "std": float(np.std(n_clusters)),
        "min": int(np.min(n_clusters)),
        "max": int(np.max(n_clusters)),
    }

    # 轨迹数
    all_ids = set()
    for f in frames:
        for t in f.get("targets", []):
            all_ids.add(t["id"])
    stats["unique_tracks"] = len(all_ids)

    # 速度统计
    all_speeds = []
    for f in frames:
        for t in f.get("targets", []):
            all_speeds.append(t["speed"])
    if all_speeds:
        stats["speed"] = {
            "mean": float(np.mean(all_speeds)),
            "std": float(np.std(all_speeds)),
            "max": float(np.max(all_speeds)),
        }
    else:
        stats["speed"] = {"mean": 0, "std": 0, "max": 0}

    # 轨迹生存分析
    track_frames = Counter()
    for f in frames:
        seen = set()
        for t in f.get("targets", []):
            if t["id"] not in seen:
                track_frames[t["id"]] += 1
                seen.add(t["id"])
    if track_frames:
        lifespans = list(track_frames.values())
        stats["track_lifespan"] = {
            "mean_frames": float(np.mean(lifespans)),
            "min_frames": int(np.min(lifespans)),
            "max_frames": int(np.max(lifespans)),
        }
    else:
        stats["track_lifespan"] = {"mean_frames": 0, "min_frames": 0, "max_frames": 0}

    return stats


def print_table(stats):
    """终端打印统计表"""
    l = stats["latency"]
    print()
    print("=" * 60)
    print("  论文实验数据汇总")
    print("=" * 60)
    print()
    print(f"  【延迟性能】")
    print(f"    平均每帧:    {l['mean_ms']:.1f} ms")
    print(f"    标准差:      {l['std_ms']:.1f} ms")
    print(f"    中位数:      {l['median_ms']:.1f} ms")
    print(f"    最慢帧:      {l['max_ms']:.1f} ms")
    print(f"    最快帧:      {l['min_ms']:.1f} ms")
    print(f"    等效帧率:    {l['fps']:.1f} FPS")
    print()

    print(f"  【各阶段耗时】")
    for stage in ["join", "fuse", "io", "tracker"]:
        s = stats[f"stage_{stage}"]
        print(f"    {stage:>8}: {s['mean_ms']:.1f} ± {s['std_ms']:.1f} ms")
    print()

    t = stats["targets"]
    print(f"  【检测统计】")
    print(f"    总帧数:        {len(frames)}")
    print(f"    每帧目标数:    {t['mean']:.1f} ± {t['std']:.1f} [{t['min']}-{t['max']}]")
    print(f"    含目标帧数:    {t['total_frames_with_targets']}")
    print()

    p = stats["person"]
    print(f"  【行人检测】")
    print(f"    有行人帧数:    {p['frames_with_person']} ({p['person_ratio']*100:.1f}%)")
    print(f"    行人累计出现:  {p['total_person_occurrences']} 次")
    print()

    c = stats["classification"]
    total = sum(c.values()) or 1
    print(f"  【分类分布】")
    for cat in ["moving", "static", "movable", "floating"]:
        v = c.get(cat, 0)
        print(f"    {cat:>9}: {v:>4} ({v/total*100:.1f}%)")
    print()

    cl = stats["clusters"]
    print(f"  【聚类统计】")
    print(f"    每帧聚类数:    {cl['mean']:.1f} ± {cl['std']:.1f} [{cl['min']}-{cl['max']}]")
    print()

    print(f"  【轨迹分析】")
    print(f"    唯一轨迹数:    {stats['unique_tracks']}")
    tl = stats["track_lifespan"]
    print(f"    轨迹帧寿命:    {tl['mean_frames']:.1f} ± [{tl['min_frames']}-{tl['max_frames']}] 帧")
    sp = stats["speed"]
    print(f"    运动速度:      {sp['mean']:.2f} ± {sp['std']:.2f} m/s (最大 {sp['max']:.2f} m/s)")
    print()
    print("=" * 60)


def export_latex(stats, output_path: Path):
    """导出 LaTeX 表格"""
    l = stats["latency"]
    lines = [
        "% 自动生成 — 论文实验数据",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{管线性能与检测统计结果}",
        r"\label{tab:experiment_results}",
        r"\begin{tabular}{l r}",
        r"\toprule",
        r"\textbf{指标} & \textbf{数值} \\",
        r"\midrule",
    ]

    def row(name, val):
        lines.append(f"    {name} & {val} \\\\")

    row("总帧数", f"{len(frames)}")
    row("每帧延迟 (mean±std)", f"{l['mean_ms']:.1f}±{l['std_ms']:.1f} ms")
    row("中位数延迟", f"{l['median_ms']:.1f} ms")
    row("等效帧率", f"{l['fps']:.1f} FPS")

    t = stats["targets"]
    row("每帧目标数 (mean±std)", f"{t['mean']:.1f}±{t['std']:.1f}")

    p = stats["person"]
    row("行人帧占比", f"{p['person_ratio']*100:.1f}\\%")

    cl = stats["clusters"]
    row("每帧聚类数 (mean±std)", f"{cl['mean']:.1f}±{cl['std']:.1f}")

    tl = stats["track_lifespan"]
    row("轨迹平均寿命", f"{tl['mean_frames']:.1f} 帧")

    sp = stats["speed"]
    row("目标平均速度", f"{sp['mean']:.2f} m/s")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  [OK] LaTeX 表格 → {output_path}")


def export_csv(stats, output_path: Path):
    """导出 CSV 汇总"""
    import csv

    with open(output_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["指标", "数值"])
        l = stats["latency"]
        w.writerow(["总帧数", len(frames)])
        w.writerow(["平均延迟 (ms)", f"{l['mean_ms']:.1f}"])
        w.writerow(["延迟标准差 (ms)", f"{l['std_ms']:.1f}"])
        w.writerow(["中位数延迟 (ms)", f"{l['median_ms']:.1f}"])
        w.writerow(["最小延迟 (ms)", f"{l['min_ms']:.1f}"])
        w.writerow(["最大延迟 (ms)", f"{l['max_ms']:.1f}"])
        w.writerow(["等效帧率 (FPS)", f"{l['fps']:.1f}"])
        for stage in ["join", "fuse", "io", "tracker"]:
            s = stats[f"stage_{stage}"]
            w.writerow([f"{stage} 耗时 (ms)", f"{s['mean_ms']:.1f}±{s['std_ms']:.1f}"])
        t = stats["targets"]
        w.writerow(["每帧目标数", f"{t['mean']:.1f}±{t['std']:.1f}"])
        cl = stats["clusters"]
        w.writerow(["每帧聚类数", f"{cl['mean']:.1f}±{cl['std']:.1f}"])
        p = stats["person"]
        w.writerow(["行人帧占比", f"{p['person_ratio']*100:.1f}%"])
        c = stats["classification"]
        for cat in ["moving", "static", "movable", "floating"]:
            w.writerow([f"{cat} 目标数", c.get(cat, 0)])
        w.writerow(["唯一轨迹数", stats["unique_tracks"]])
        sp = stats["speed"]
        w.writerow(["平均速度 (m/s)", f"{sp['mean']:.2f}"])
        w.writerow(["最大速度 (m/s)", f"{sp['max']:.2f}"])
    print(f"  [OK] CSV 汇总 → {output_path}")


def plot_classification_pie(stats, output_path: Path):
    """分类分布饼图"""
    c = stats["classification"]
    total = sum(c.values())
    if total == 0:
        return

    labels = {
        "moving": "运动中\n(Moving)",
        "static": "静止\n(Static)",
        "movable": "可运动\n(Movable)",
        "floating": "待定\n(Floating)",
    }
    colors = ["#E63946", "#457B9D", "#2A9D8F", "#E9C46A"]
    sizes = [c.get(k, 0) for k in labels]
    lbls = [labels[k] for k in labels]

    fig, ax = plt.subplots(figsize=(5, 5))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=lbls, colors=colors, autopct="%1.1f%%",
        startangle=90, pctdistance=0.6,
        wedgeprops={"edgecolor": "white", "linewidth": 1.2},
    )
    for t in autotexts:
        t.set_fontsize(9)
        t.set_fontweight("bold")
    for t in texts:
        t.set_fontsize(9)
    ax.set_title("目标分类分布", fontweight="bold", pad=15)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  [OK] 分类饼图 → {output_path}")


# ─── 主入口 ───────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) > 1:
        jsonl_path = Path(sys.argv[1])
    else:
        jsonl_path = find_latest_jsonl()

    if not jsonl_path.exists():
        print(f"[ERR] 文件不存在: {jsonl_path}")
        sys.exit(1)

    global frames
    frames = load_data(jsonl_path)
    out_dir = jsonl_path.parent

    print(f"读取数据: {jsonl_path}  ({len(frames)} 帧)")
    stats = compute_summary_stats(frames)

    print_table(stats)
    export_csv(stats, out_dir / "summary.csv")
    export_latex(stats, out_dir / "summary_table.tex")
    plot_classification_pie(stats, out_dir / "fig_classification_pie.png")


if __name__ == "__main__":
    main()
