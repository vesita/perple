"""论文实验数据汇总 — 生成统计表 + LaTeX 导出

用法:
    .venv/Scripts/python.exe scripts/viz_summary.py
    .venv/Scripts/python.exe scripts/viz_summary.py output/pipeline_xxx/pipeline.jsonl
"""
import json, sys, csv
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.chart_style import savefig, C_RED, C_BLUE, C_GREEN, C_YELLOW


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


def compute_summary_stats(frames, out_dir):
    n_frames = len(frames)
    elapsed = np.array([f["elapsed_ms"] for f in frames])
    stats = {"latency": {
        "mean_ms": float(np.mean(elapsed)),
        "std_ms": float(np.std(elapsed)),
        "min_ms": float(np.min(elapsed)),
        "max_ms": float(np.max(elapsed)),
        "median_ms": float(np.median(elapsed)),
        "fps": 1000.0 / float(np.mean(elapsed)) if np.mean(elapsed) > 0 else 0,
    }}
    for stage in ["join", "fuse", "io", "tracker"]:
        vals = np.array([f["stages_ms"][stage] for f in frames])
        stats[f"stage_{stage}"] = {"mean_ms": float(np.mean(vals)), "std_ms": float(np.std(vals))}

    nt = np.array([f["stats"]["n_targets"] for f in frames])
    stats["targets"] = {"mean": float(np.mean(nt)), "std": float(np.std(nt)),
                        "min": int(np.min(nt)), "max": int(np.max(nt)),
                        "total_frames_with_targets": int(np.sum(nt > 0))}

    total = Counter()
    np_frame, np_total = 0, 0
    for f in frames:
        s = f["stats"]
        for k in ["moving", "static", "movable", "floating"]:
            total[k] += s.get(f"n_{k}", 0)
        if s.get("n_person", 0) > 0:
            np_frame += 1
        np_total += s.get("n_person", 0)
    stats["classification"] = dict(total)
    stats["person"] = {"frames_with_person": np_frame,
                       "person_ratio": np_frame / n_frames if n_frames > 0 else 0,
                       "total_person_occurrences": np_total}

    nc = np.array([f["stats"]["n_clusters"] for f in frames])
    stats["clusters"] = {"mean": float(np.mean(nc)), "std": float(np.std(nc)),
                         "min": int(np.min(nc)), "max": int(np.max(nc))}

    all_ids = set()
    for f in frames:
        for t in f.get("targets", []):
            all_ids.add(t["id"])
    stats["unique_tracks"] = len(all_ids)

    speeds = [t["speed"] for f in frames for t in f.get("targets", [])]
    stats["speed"] = {"mean": float(np.mean(speeds)), "std": float(np.std(speeds)),
                      "max": float(np.max(speeds))} if speeds else {"mean": 0, "std": 0, "max": 0}

    track_frames = Counter()
    for f in frames:
        seen = set()
        for t in f.get("targets", []):
            if t["id"] not in seen:
                track_frames[t["id"]] += 1
                seen.add(t["id"])
    if track_frames:
        lf = list(track_frames.values())
        stats["track_lifespan"] = {"mean_frames": float(np.mean(lf)),
                                   "min_frames": int(np.min(lf)), "max_frames": int(np.max(lf))}
    else:
        stats["track_lifespan"] = {"mean_frames": 0, "min_frames": 0, "max_frames": 0}

    return stats


def print_table(stats, frames):
    l = stats["latency"]
    print("\n" + "=" * 60)
    print("  论文实验数据汇总")
    print("=" * 60)
    print(f"\n  【延迟性能】")
    print(f"    平均每帧:    {l['mean_ms']:.1f} ms")
    print(f"    标准差:      {l['std_ms']:.1f} ms")
    print(f"    等效帧率:    {l['fps']:.1f} FPS")
    print(f"\n  【各阶段耗时】")
    for stage in ["join", "fuse", "io", "tracker"]:
        s = stats[f"stage_{stage}"]
        print(f"    {stage:>8}: {s['mean_ms']:.1f} ± {s['std_ms']:.1f} ms")
    t = stats["targets"]
    print(f"\n  【检测统计】")
    print(f"    总帧数:        {len(frames)}")
    print(f"    每帧目标数:    {t['mean']:.1f} ± {t['std']:.1f}")
    p = stats["person"]
    print(f"    有行人帧数:    {p['frames_with_person']} ({p['person_ratio']*100:.1f}%)")
    cl = stats["clusters"]
    print(f"    每帧聚类数:    {cl['mean']:.1f} ± {cl['std']:.1f}")
    print(f"\n  【轨迹分析】")
    print(f"    唯一轨迹数:    {stats['unique_tracks']}")
    tl = stats["track_lifespan"]
    print(f"    轨迹帧寿命:    {tl['mean_frames']:.1f} 帧")
    sp = stats["speed"]
    print(f"    运动速度:      {sp['mean']:.2f} ± {sp['std']:.2f} m/s")
    print("=" * 60)


def export_csv(stats, output_path):
    with open(output_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["指标", "数值"])
        l = stats["latency"]
        w.writerow(["总帧数", len(frames)])
        w.writerow(["平均延迟 (ms)", f"{l['mean_ms']:.1f}"])
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
    print(f"  [OK] CSV → {output_path}")


def export_latex(stats, output_path):
    l = stats["latency"]
    lines = [
        "% 自动生成",
        r"\begin{table}[htbp]", r"\centering",
        r"\caption{管线性能与检测统计}", r"\label{tab:experiment_results}",
        r"\begin{tabular}{l r}", r"\toprule",
        r"\textbf{指标} & \textbf{数值} \\", r"\midrule",
    ]
    def row(name, val):
        lines.append(f"    {name} & {val} \\\\")
    row("总帧数", f"{len(frames)}")
    row("每帧延迟", f"{l['mean_ms']:.1f}±{l['std_ms']:.1f} ms")
    row("等效帧率", f"{l['fps']:.1f} FPS")
    t = stats["targets"]
    row("每帧目标数", f"{t['mean']:.1f}±{t['std']:.1f}")
    p = stats["person"]
    row("行人帧占比", f"{p['person_ratio']*100:.1f}\\%")
    cl = stats["clusters"]
    row("每帧聚类数", f"{cl['mean']:.1f}±{cl['std']:.1f}")
    sp = stats["speed"]
    row("目标平均速度", f"{sp['mean']:.2f} m/s")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  [OK] LaTeX → {output_path}")


def plot_classification_pie(stats, output_path):
    c = stats["classification"]
    total = sum(c.values())
    if total == 0:
        return
    labels = {"moving": "Moving", "static": "Static", "movable": "Movable", "floating": "Floating"}
    colors = [C_RED, C_BLUE, C_GREEN, C_YELLOW]
    sizes = [c.get(k, 0) for k in labels]

    fig, ax = plt.subplots(figsize=(5, 5))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=[labels[k] for k in labels], colors=colors,
        autopct="%1.1f%%", startangle=90, pctdistance=0.6,
        wedgeprops={"edgecolor": "white", "linewidth": 1.2})
    for t in autotexts:
        t.set_fontsize(9); t.set_fontweight("bold")
    for t in texts:
        t.set_fontsize(9)
    savefig(fig, output_path)
    print(f"  [OK] 饼图 → {output_path}")


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
    print(f"读取数据: {jsonl_path} ({len(frames)} 帧)")
    stats = compute_summary_stats(frames, out_dir)

    print_table(stats, frames)
    export_csv(stats, out_dir / "summary.csv")
    export_latex(stats, out_dir / "summary_table.tex")
    plot_classification_pie(stats, out_dir / "fig_classification_pie.png")


if __name__ == "__main__":
    main()
