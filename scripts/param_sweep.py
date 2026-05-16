"""
参数批量扫描 + 多维度可视化分析脚本

用法：
    uv run python scripts/param_sweep.py cluster              # 聚类扫参 + 图
    uv run python scripts/param_sweep.py wall                  # 墙体扫参 + 图
    uv run python scripts/param_sweep.py ground                # 地面扫参 + 图
    uv run python scripts/param_sweep.py all                   # 全部
    uv run python scripts/param_sweep.py all --frames=5        # 自定义帧数
    uv run python scripts/param_sweep.py ground --strategy=peak_scan --frames=20  # CLI 传参

生成的图表输出到 output/<bench>/viz/ 目录。
"""

import json
import sys
import os
import subprocess
import re
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ── 中文字体配置 ──────────────────────────────────────────────────────

def setup_chinese_font():
    """尝试加载系统中文字体，fallback 到 sans-serif"""
    candidates = [
        "Microsoft YaHei", "SimHei", "WenQuanYi Micro Hei",
        "Noto Sans CJK SC", "Source Han Sans SC", "PingFang SC",
        "Hiragino Sans GB", "STHeiti",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans", "Arial"]
            return name
    # fallback
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
    return None

CN_FONT = setup_chinese_font()
_s = "[param_sweep]"
if CN_FONT:
    print(f"{_s} Chinese font: {CN_FONT}")
else:
    print(f"{_s} WARNING: no CJK font found, using English labels")

# ── 工具函数 ──────────────────────────────────────────────────────────

BENCH_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_BASE = os.path.join(BENCH_DIR, "output")

def run_bench(bench: str, extra_args: list[str] = None) -> list[dict]:
    """运行 bench --sweep --json，返回解析后的 JSON 结果列表"""
    if extra_args is None:
        extra_args = []
    cmd = ["cargo", "run", "--example", f"{bench}_bench", "--"]
    cmd.extend(extra_args)
    cmd.extend(["--sweep", "--json"])

    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=BENCH_DIR,
                            capture_output=True, text=True,
                            encoding='utf-8', errors='replace', timeout=600)

    if result.returncode != 0:
        print(f"  STDERR: {result.stderr[:500]}")
        raise RuntimeError(f"{bench}_bench failed with code {result.returncode}")

    # 解析 === JSON === 块
    m = re.search(r"=== JSON ===\n(\[.*\])", result.stdout, re.DOTALL)
    if not m:
        print(f"  stdout tail:\n{result.stdout[-1000:]}")
        raise RuntimeError(f"Cannot find JSON block in {bench}_bench output")

    return json.loads(m.group(1))


def save_summary(stats: list[dict], path: str):
    """保存汇总 CSV"""
    import csv
    if not stats:
        return
    keys = list(stats[0].keys())
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(stats)


def short_name(name: str, max_len: int = 30) -> str:
    return name if len(name) <= max_len else name[:max_len - 1] + "…"


def percentile_based_cmap(values, cmap_name="RdYlGn_r"):
    """用百分位映射颜色，避免极端值漂白全局"""
    import matplotlib.colors as mcolors
    norm = mcolors.PercentileNorm()
    cmap = plt.get_cmap(cmap_name)
    return [cmap(norm(v)) for v in values]


# ── 聚类参数扫描分析 ─────────────────────────────────────────────────

def analyze_cluster(records: list[dict], out_dir: str):
    print(f"\n  聚类策略分析: {len(records)} 策略")

    # 按策略类型分组
    wall_c = [r for r in records if r["name"].startswith("wall_c")]
    lv_dot = [r for r in records if r["name"].startswith("lv_d")]
    xy_raw = [r for r in records if r["name"].startswith("xy_e")]
    ri = [r for r in records if r["name"].startswith("ri_")]
    db_ad = [r for r in records if r["name"].startswith("db_")]

    # 1. 总览条形图 (按平均耗时排序)
    sorted_recs = sorted(records, key=lambda r: r["avg_ms"])
    fig, ax = plt.subplots(figsize=(14, max(8, len(sorted_recs) * 0.25)))
    names = [short_name(r["name"]) for r in sorted_recs]
    means = [r["avg_ms"] for r in sorted_recs]
    clusters = [r.get("avg_clusters", 0) for r in sorted_recs]

    colors = ["#e74c3c" if m > 100 else "#3498db" for m in means]
    bars = ax.barh(range(len(names)), means, color=colors, height=0.7)
    for i, (bar, c) in enumerate(zip(bars, clusters)):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f" {c:.0f}簇", va="center", fontsize=6, color="#555")

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("平均耗时 (ms)")
    ax.set_title("聚类策略性能对比 (← 越快) — 右侧为平均簇数")
    ax.axvline(x=33, color="orange", linestyle="--", linewidth=1, alpha=0.5, label="33ms (30fps)")
    ax.axvline(x=100, color="#e74c3c", linestyle="--", linewidth=1, alpha=0.5, label="100ms")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "cluster_bar.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    → cluster_bar.png")

    # 2. 簇数 vs 耗时散点图
    fig, ax = plt.subplots(figsize=(12, 8))
    groups = [("墙体聚类", wall_c, "o", "#3498db"),
              ("LV-DOT降采样", lv_dot, "s", "#2ecc71"),
              ("XY平面DBSCAN", xy_raw, "^", "#e67e22"),
              ("距离图像", ri, "D", "#9b59b6"),
              ("自适应DBSCAN", db_ad, "v", "#e74c3c")]
    for label, data, marker, color in groups:
        if not data:
            continue
        xs = [r["avg_ms"] for r in data]
        ys = [r.get("avg_clusters", 0) for r in data]
        sizes = [max(20, min(c * 5, 200)) for c in ys]
        sc = ax.scatter(xs, ys, c=color, marker=marker, s=sizes, alpha=0.6,
                        label=f"{label} (大小=簇数)", zorder=3)
        for r in data:
            if r.get("avg_clusters", 0) >= 5 and r["avg_ms"] < 33:
                ax.annotate(short_name(r["name"], 20), (r["avg_ms"], r.get("avg_clusters", 0)),
                            fontsize=5, alpha=0.7, xytext=(3, 3), textcoords="offset points")

    ax.set_xlabel("平均耗时 (ms)")
    ax.set_ylabel("平均簇数")
    ax.set_title("聚类策略: 簇数 vs 耗时 (点越大=簇数越多)")
    ax.axvline(x=33, color="orange", linestyle="--", linewidth=0.8, alpha=0.4)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "cluster_scatter.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    → cluster_scatter.png")

    # 3. 墙体聚类参数热力图: eps × 最小点数 → 簇数
    if wall_c:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax_i, metric in enumerate(["avg_clusters", "avg_ms"]):
            ax = axes[ax_i]
            eps_vals = sorted(set(r["name"].split("_")[2].lstrip("e") for r in wall_c))
            mp_vals = sorted(set(r["name"].split("_")[3].lstrip("m") for r in wall_c),
                            key=lambda x: float(x))
            for cs_val in sorted(set(r["name"].split("_")[1].lstrip("c") for r in wall_c)):
                subset = [r for r in wall_c if r["name"].split("_")[1] == f"c{cs_val}"]
                if len(subset) < 4:
                    continue
                heat = np.full((len(mp_vals), len(eps_vals)), np.nan)
                for r in subset:
                    parts = r["name"].split("_")
                    eps = parts[2].lstrip("e")
                    mp = parts[3].lstrip("m")
                    if eps in eps_vals and mp in mp_vals:
                        ei = eps_vals.index(eps)
                        mi = mp_vals.index(mp)
                        heat[mi, ei] = r.get(metric, 0)

                im = ax.imshow(heat, aspect="auto", cmap="YlOrRd" if metric == "avg_ms" else "Greens",
                               interpolation="nearest")
                ax.set_xticks(range(len(eps_vals)))
                ax.set_xticklabels([f"{float(e):.2f}" for e in eps_vals], fontsize=8)
                ax.set_yticks(range(len(mp_vals)))
                ax.set_yticklabels([f"最小={m}" for m in mp_vals], fontsize=8)
                for ri_i in range(heat.shape[0]):
                    for ci in range(heat.shape[1]):
                        v = heat[ri_i, ci]
                        if not np.isnan(v):
                            ax.text(ci, ri_i, f"{v:.0f}",
                                    ha="center", va="center", fontsize=7,
                                    color="white" if v > np.nanmedian(heat) else "black")
                ax.set_xlabel("邻域半径 (m)")
                ax.set_ylabel("最小点数")
                mlabel = "平均簇数" if metric == "avg_clusters" else "平均耗时(ms)"
                ax.set_title(f"网格大小={cs_val}  {mlabel}")
                plt.colorbar(im, ax=ax, shrink=0.8)

        fig.suptitle("墙体聚类参数灵敏度", fontsize=13)
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, "cluster_wall_heatmap.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"    → cluster_wall_heatmap.png")

    # 3b. 距离图像参数分析: 方位角分辨率 × 阈值 → 耗时
    if ri:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax_i, metric in enumerate(["avg_clusters", "avg_ms"]):
            ax = axes[ax_i]
            az_vals = sorted(set(float(r["name"].split("_")[1].lstrip("a")) for r in ri))
            el_vals = sorted(set(float(r["name"].split("_")[2].lstrip("e")) for r in ri))
            th_vals = sorted(set(float(r["name"].split("_")[3].lstrip("t")) for r in ri))
            for el in el_vals:
                subset = [r for r in ri if abs(float(r["name"].split("_")[2].lstrip("e")) - el) < 1e-6]
                if not subset:
                    continue
                xs = [float(r["name"].split("_")[3].lstrip("t")) for r in subset]
                ys = [r.get(metric, 0) for r in subset]
                ax.plot(xs, ys, marker="o", label=f"俯仰={el:.0f}°")
            ax.set_xlabel("距离阈值 (m)")
            mlabel = "平均簇数" if metric == "avg_clusters" else "平均耗时(ms)"
            ax.set_ylabel(mlabel)
            ax.set_title(f"距离图像: {mlabel}")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, "cluster_rangeimage_params.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"    → cluster_rangeimage_params.png")

    # 3c. 自适应DBSCAN参数: 容忍度 × 斜率 → 簇数
    if db_ad:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax_i, metric in enumerate(["avg_clusters", "avg_ms"]):
            ax = axes[ax_i]
            pat_vals = sorted(set(float(r["name"].split("_")[1].lstrip("p")) for r in db_ad))
            slp_vals = sorted(set(float(r["name"].split("_")[2].lstrip("s")) for r in db_ad))
            heat = np.full((len(slp_vals), len(pat_vals)), np.nan)
            for r in db_ad:
                parts = r["name"].split("_")
                pat = float(parts[1].lstrip("p"))
                slp = float(parts[2].lstrip("s"))
                if pat in pat_vals and slp in slp_vals:
                    pi = pat_vals.index(pat)
                    si = slp_vals.index(slp)
                    heat[si, pi] = r.get(metric, 0)
            im = ax.imshow(heat, aspect="auto", cmap="Greens" if metric == "avg_clusters" else "YlOrRd",
                           interpolation="nearest")
            ax.set_xticks(range(len(pat_vals)))
            ax.set_xticklabels([f"{p:.2f}" for p in pat_vals])
            ax.set_yticks(range(len(slp_vals)))
            ax.set_yticklabels([f"{s:.2f}" for s in slp_vals])
            for ri_i in range(heat.shape[0]):
                for ci in range(heat.shape[1]):
                    v = heat[ri_i, ci]
                    if not np.isnan(v):
                        ax.text(ci, ri_i, f"{v:.1f}", ha="center", va="center", fontsize=9,
                                color="white" if v > np.nanmedian(heat) else "black")
            ax.set_xlabel("容忍度")
            ax.set_ylabel("斜率")
            mlabel = "平均簇数" if metric == "avg_clusters" else "平均耗时(ms)"
            ax.set_title(f"自适应DBSCAN: {mlabel}")
            plt.colorbar(im, ax=ax, shrink=0.8)
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, "cluster_dbscan_params.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"    → cluster_dbscan_params.png")

    # 4. (已移除人形检测 — 点云阶段不输出行人信息，融合后确定)


# ── 墙体参数扫描分析 ─────────────────────────────────────────────────

def analyze_wall(records: list[dict], out_dir: str):
    print(f"\n  墙体策略分析: {len(records)} 策略")

    sorted_recs = sorted(records, key=lambda r: r["avg_ms"])

    # 1. 总览条形图
    fig, ax = plt.subplots(figsize=(14, max(8, len(sorted_recs) * 0.22)))
    names = [short_name(r["name"]) for r in sorted_recs]
    means = [r["avg_ms"] for r in sorted_recs]
    wall_pts = [r.get("avg_wall_pts", 0) for r in sorted_recs]
    obstacles = [r.get("avg_obstacles", 0) for r in sorted_recs]

    colors = ["#e74c3c" if m > 100 else "#3498db" for m in means]
    bars = ax.barh(range(len(names)), means, color=colors, height=0.7)
    for i, (bar, wp, ob) in enumerate(zip(bars, wall_pts, obstacles)):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f" {wp:.0f}墙点 {ob:.0f}障碍物", va="center", fontsize=6, color="#555")

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("平均耗时 (ms)")
    ax.set_title("墙体策略性能对比 (← 越快) — 右侧为墙点/障碍物数")
    ax.axvline(x=33, color="orange", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axvline(x=100, color="#e74c3c", linestyle="--", linewidth=0.8, alpha=0.5)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "wall_bar.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    → wall_bar.png")

    # 2. 墙点 vs 耗时 (左上: 墙点多+耗时少为佳)
    fig, ax = plt.subplots(figsize=(12, 8))
    td = [r for r in records if r["name"].startswith("td_")]
    xy = [r for r in records if r["name"].startswith("xy_")]
    nw = [r for r in records if r["name"].startswith("nw_")]
    qt = [r for r in records if r["name"].startswith("qt_")]
    sf = [r for r in records if r["name"].startswith("sf_")]
    ad = [r for r in records if r["name"].startswith("ad_")]
    xw = [r for r in records if r["name"].startswith("xw_")]

    for label, data, marker, color in [("自上而下聚类", td, "o", "#3498db"),
                                        ("XY平面RANSAC", xy, "s", "#e74c3c"),
                                        ("法向量墙面", nw, "^", "#2ecc71"),
                                        ("四叉树", qt, "D", "#9b59b6"),
                                        ("顺序拟合", sf, "v", "#f39c12"),
                                        ("自适应DBSCAN", ad, "P", "#1abc9c"),
                                        ("XY平面DBSCAN", xw, "X", "#34495e")]:
        if not data:
            continue
        xs = [r["avg_ms"] for r in data]
        ys = [r.get("avg_wall_pts", 0) for r in data]
        obstacles = [r.get("avg_obstacles", 0) for r in data]
        sizes = [max(20, o * 20 + 20) for o in obstacles]
        ax.scatter(xs, ys, c=color, marker=marker, s=sizes, alpha=0.6, label=label, zorder=3)
        for r in data:
            if r.get("avg_wall_pts", 0) > 5000 and r["avg_ms"] < 20:
                ax.annotate(short_name(r["name"], 18), (r["avg_ms"], r.get("avg_wall_pts", 0)),
                            fontsize=5, alpha=0.7, xytext=(3, 3), textcoords="offset points")

    ax.set_xlabel("平均耗时 (ms)")
    ax.set_ylabel("平均墙面点数")
    ax.set_title("墙体策略: 墙点提取 vs 耗时 (点大小=障碍物数) — 左上最佳")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "wall_scatter.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    → wall_scatter.png")

    # 3. 障碍物 vs 耗时
    fig, ax = plt.subplots(figsize=(12, 7))
    for label, data, marker, color in [("自上而下聚类", td, "o", "#3498db"),
                                        ("XY平面RANSAC", xy, "s", "#e74c3c"),
                                        ("法向量墙面", nw, "^", "#2ecc71"),
                                        ("四叉树", qt, "D", "#9b59b6"),
                                        ("顺序拟合", sf, "v", "#f39c12"),
                                        ("自适应DBSCAN", ad, "P", "#1abc9c"),
                                        ("XY平面DBSCAN", xw, "X", "#34495e")]:
        if not data:
            continue
        xs = [r["avg_ms"] for r in data]
        ys = [r.get("avg_obstacles", 0) for r in data]
        far = [r.get("avg_far_obstacles", 0) for r in data]
        sizes = [max(20, f * 40 + 20) for f in far]
        ax.scatter(xs, ys, c=color, marker=marker, s=sizes, alpha=0.6, label=label, zorder=3)
    ax.set_xlabel("平均耗时 (ms)")
    ax.set_ylabel("平均近距障碍物数")
    ax.set_title("近距障碍物 vs 耗时 (点大小=远距障碍物)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "wall_obstacles.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    → wall_obstacles.png")

    # 4. XY平面RANSAC参数灵敏度
    if xy:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax_i, metric in enumerate(["avg_wall_pts", "avg_ms"]):
            ax = axes[ax_i]
            xy_s42 = [r for r in xy if "s42" in r["name"]]
            xy_rng = [r for r in xy if "rng" in r["name"]]
            for sub, lbl, mk in [(xy_s42, "固定种子=42", "o"), (xy_rng, "随机种子", "x")]:
                dists = sorted(set(r["name"].split("_")[1].lstrip("d") for r in sub))
                iters_list = sorted(set(r["name"].split("_")[2].lstrip("i") for r in sub),
                                   key=lambda x: int(x))
                for d in dists:
                    vals = []
                    for it in iters_list:
                        match = [r for r in sub
                                 if r["name"].split("_")[1] == f"d{d}"
                                 and r["name"].split("_")[2] == f"i{it}"]
                        vals.append(match[0].get(metric, 0) if match else 0)
                    ax.plot(iters_list, vals, marker=mk, label=f"距离={d} {lbl}")
            ax.set_xlabel("迭代次数")
            mlabel = "平均墙点数" if metric == "avg_wall_pts" else "平均耗时(ms)"
            ax.set_ylabel(mlabel)
            ax.set_title(f"XY平面RANSAC: {mlabel}")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, "wall_xy_ransac_params.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"    → wall_xy_ransac_params.png")

    # 5. 顺序拟合参数分析
    if sf:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax_i, metric in enumerate(["avg_wall_pts", "avg_ms"]):
            ax = axes[ax_i]
            dists = sorted(set(float(r["name"].split("_")[1].lstrip("d")) for r in sf))
            th_vals = sorted(set(float(r["name"].split("_")[2].lstrip("t")) for r in sf))
            for d in dists:
                for th in th_vals:
                    subset = [r for r in sf
                              if abs(float(r["name"].split("_")[1].lstrip("d")) - d) < 1e-6
                              and abs(float(r["name"].split("_")[2].lstrip("t")) - th) < 1e-6]
                    mw_vals = sorted(set(int(r["name"].split("_")[3].lstrip("w")) for r in subset))
                    vals = [next((r.get(metric, 0) for r in subset
                                  if int(r["name"].split("_")[3].lstrip("w")) == mw), 0)
                            for mw in mw_vals]
                    if mw_vals:
                        ax.plot(mw_vals, vals, marker="o", label=f"距离={d:.2f} 阈值={th:.1f}")
            ax.set_xlabel("最大墙面数")
            mlabel = "平均墙点数" if metric == "avg_wall_pts" else "平均耗时(ms)"
            ax.set_ylabel(mlabel)
            ax.set_title(f"顺序拟合: {mlabel}")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, "wall_seqfit_params.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"    → wall_seqfit_params.png")

    # 6. 自适应DBSCAN参数分析
    if ad:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax_i, metric in enumerate(["avg_wall_pts", "avg_ms"]):
            ax = axes[ax_i]
            for ds_label in ["g", "f"]:
                subset = [r for r in ad if r["name"].endswith(ds_label)]
                if not subset:
                    continue
                be_vals = sorted(set(float(r["name"].split("_")[1].lstrip("be")) for r in subset))
                sf_vals = sorted(set(float(r["name"].split("_")[2].lstrip("s")) for r in subset))
                for be in be_vals:
                    vals = []
                    for s in sf_vals:
                        match = [r for r in subset
                                 if abs(float(r["name"].split("_")[1].lstrip("be")) - be) < 1e-6
                                 and abs(float(r["name"].split("_")[2].lstrip("s")) - s) < 1e-6]
                        vals.append(np.mean([r.get(metric, 0) for r in match]) if match else 0)
                    ax.plot(sf_vals, vals, marker="o", label=f"基础eps={be:.3f} 下采样={'网格' if ds_label == 'g' else 'FPS'}")
            ax.set_xlabel("尺度因子")
            mlabel = "平均墙点数" if metric == "avg_wall_pts" else "平均耗时(ms)"
            ax.set_ylabel(mlabel)
            ax.set_title(f"自适应DBSCAN: {mlabel}")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, "wall_adaptdbscan_params.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"    → wall_adaptdbscan_params.png")


# ── 地面参数扫描分析 ─────────────────────────────────────────────────

def analyze_ground(records: list[dict], out_dir: str):
    print(f"\n  地面策略分析: {len(records)} 策略")

    sorted_recs = sorted(records, key=lambda r: r["avg_ms"])

    # 1. 总览条形图 (地面占比作为颜色映射)
    fig, ax = plt.subplots(figsize=(14, max(8, len(sorted_recs) * 0.22)))
    names = [short_name(r["name"]) for r in sorted_recs]
    means = [r["avg_ms"] for r in sorted_recs]
    ratios = [r.get("ground_ratio", 0) for r in sorted_recs]

    norm = plt.Normalize(min(ratios), max(ratios))
    cmap = plt.get_cmap("RdYlGn")
    colors = [cmap(norm(r)) for r in ratios]

    bars = ax.barh(range(len(names)), means, color=colors, height=0.7)
    for i, (bar, ratio) in enumerate(zip(bars, ratios)):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                f" {ratio:.1f}%", va="center", fontsize=7, color="#555")

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("平均耗时 (ms)")
    ax.set_title("地面策略性能对比 (颜色越绿=地面占比越高)")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "ground_bar.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    → ground_bar.png")

    # 2. 地面占比 vs 耗时 (左上最佳: 占比高+耗时少)
    fig, ax = plt.subplots(figsize=(10, 7))
    hist = [r for r in records if r["name"].startswith("hist_")]
    peak = [r for r in records if r["name"].startswith("peak_")]
    ransac = [r for r in records if r["name"].startswith("ransac_")]
    seed = [r for r in records if r["name"].startswith("seed_")]
    gpf = [r for r in records if r["name"].startswith("gpf_")]

    groups2 = [("直方图扩展", hist, "o", "#3498db"),
              ("峰值扫描", peak, "s", "#e74c3c"),
              ("RANSAC", ransac, "^", "#2ecc71"),
              ("直方图种子", seed, "D", "#9b59b6"),
              ("GPF", gpf, "v", "#f39c12")]

    for label, data, marker, color in groups2:
        if not data:
            continue
        xs = [r["avg_ms"] for r in data]
        ys = [r.get("ground_ratio", 0) for r in data]
        ax.scatter(xs, ys, c=color, marker=marker, s=50, alpha=0.7, label=label, zorder=3)
        for r in data:
            if r.get("ground_ratio", 0) > 18 and r["avg_ms"] < 15:
                ax.annotate(short_name(r["name"], 20), (r["avg_ms"], r.get("ground_ratio", 0)),
                            fontsize=6, alpha=0.7, xytext=(3, 3), textcoords="offset points")

    ax.set_xlabel("平均耗时 (ms)")
    ax.set_ylabel("地面占比 (%)")
    ax.set_title("地面策略: 提取比例 vs 耗时 (左上最佳)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "ground_scatter.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    → ground_scatter.png")

    # 3. 峰值扫描参数热力图
    if peak:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for ax_i, metric in enumerate(["ground_ratio", "avg_ms"]):
            ax = axes[ax_i]
            th_vals = sorted(set(float(r["name"].split("_")[1].lstrip("t")) for r in peak))
            ex_vals = sorted(set(float(r["name"].split("_")[2].lstrip("e")) for r in peak))
            heat = np.full((len(ex_vals), len(th_vals)), np.nan)
            for r in peak:
                parts = r["name"].split("_")
                th = float(parts[1].lstrip("t"))
                ex = float(parts[2].lstrip("e"))
                if th in th_vals and ex in ex_vals:
                    ti = th_vals.index(th)
                    ei = ex_vals.index(ex)
                    heat[ei, ti] = r.get(metric, 0)

            im = ax.imshow(heat, aspect="auto", cmap="Greens" if metric == "ground_ratio" else "YlOrRd",
                           interpolation="nearest")
            ax.set_xticks(range(len(th_vals)))
            ax.set_xticklabels([f"{t:.2f}" for t in th_vals], fontsize=9)
            ax.set_yticks(range(len(ex_vals)))
            ax.set_yticklabels([f"扩展={e:.2f}" for e in ex_vals], fontsize=9)
            for ri in range(heat.shape[0]):
                for ci in range(heat.shape[1]):
                    v = heat[ri, ci]
                    if not np.isnan(v):
                        ax.text(ci, ri, f"{v:.1f}", ha="center", va="center", fontsize=8,
                                color="white" if v > np.nanmedian(heat) else "black")
            ax.set_xlabel("阈值 (m)")
            ax.set_ylabel("扩展距离 (m)")
            mlabel = "地面占比(%)" if metric == "ground_ratio" else "平均耗时(ms)"
            ax.set_title(f"峰值扫描: {mlabel}")
            plt.colorbar(im, ax=ax, shrink=0.8)

        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, "ground_peak_heatmap.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"    → ground_peak_heatmap.png")

    # 4. RANSAC 参数分析
    if ransac:
        fig, ax = plt.subplots(figsize=(8, 4))
        dists = sorted(set(float(r["name"].split("_")[1].lstrip("d")) for r in ransac))
        iters_vals = sorted(set(int(r["name"].split("_")[2].lstrip("i")) for r in ransac))
        for d in dists:
            vals = []
            for it in iters_vals:
                match = [r for r in ransac
                         if float(r["name"].split("_")[1].lstrip("d")) == d
                         and int(r["name"].split("_")[2].lstrip("i")) == it]
                vals.append(match[0]["avg_ms"] if match else 0)
            ax.plot(iters_vals, vals, marker="o", label=f"距离={d}")
        ax.set_xlabel("迭代次数")
        ax.set_ylabel("平均耗时 (ms)")
        ax.set_title("RANSAC 参数 vs 耗时")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, "ground_ransac_params.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"    → ground_ransac_params.png")

    # 5. GPF 参数分析
    if gpf:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax_i, metric in enumerate(["ground_ratio", "avg_ms"]):
            ax = axes[ax_i]
            ts_vals = sorted(set(float(r["name"].split("_")[2].lstrip("s")) for r in gpf))
            td_vals = sorted(set(float(r["name"].split("_")[3].lstrip("d")) for r in gpf))
            heat = np.full((len(td_vals), len(ts_vals)), np.nan)
            for r in gpf:
                parts = r["name"].split("_")
                ts = float(parts[2].lstrip("s"))
                td = float(parts[3].lstrip("d"))
                if ts in ts_vals and td in td_vals:
                    ti = ts_vals.index(ts)
                    di = td_vals.index(td)
                    heat[di, ti] = r.get(metric, 0)
            im = ax.imshow(heat, aspect="auto", cmap="Greens" if metric == "ground_ratio" else "YlOrRd",
                           interpolation="nearest")
            ax.set_xticks(range(len(ts_vals)))
            ax.set_xticklabels([f"种子阈值={t:.1f}" for t in ts_vals], fontsize=8)
            ax.set_yticks(range(len(td_vals)))
            ax.set_yticklabels([f"距离阈值={d:.2f}" for d in td_vals], fontsize=8)
            for ri in range(heat.shape[0]):
                for ci in range(heat.shape[1]):
                    v = heat[ri, ci]
                    if not np.isnan(v):
                        ax.text(ci, ri, f"{v:.1f}", ha="center", va="center", fontsize=8,
                                color="white" if v > np.nanmedian(heat) else "black")
            ax.set_xlabel("种子阈值")
            ax.set_ylabel("距离阈值")
            mlabel = "地面占比(%)" if metric == "ground_ratio" else "平均耗时(ms)"
            ax.set_title(f"GPF: {mlabel}")
            plt.colorbar(im, ax=ax, shrink=0.8)
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, "ground_gpf_params.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"    → ground_gpf_params.png")


# ── 入口 ──────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        sys.exit(1)

    target = sys.argv[1]
    extra_args = sys.argv[2:] if len(sys.argv) > 2 else []
    # 过滤掉目标参数
    if target in ("all", "cluster", "wall", "ground"):
        pass
    else:
        # 可能是老的用法
        extra_args = sys.argv[1:]

    benches = []
    if target in ("all", "cluster"):
        benches.append("cluster")
    if target in ("all", "wall"):
        benches.append("wall")
    if target in ("all", "ground"):
        benches.append("ground")
    if not benches:
        print(f"未知目标: {target}")
        print(__doc__)
        sys.exit(1)

    for bench in benches:
        print(f"\n{'='*60}")
        print(f"  扫描 {bench}_bench ...")
        print(f"{'='*60}")

        out_dir = os.path.join(OUTPUT_BASE, f"{bench}_bench", "viz")
        os.makedirs(out_dir, exist_ok=True)

        try:
            records = run_bench(bench, extra_args)
            print(f"  获取 {len(records)} 条策略结果")

            if not records:
                print("  无数据，跳过")
                continue

            # 保存原始 JSON
            json_path = os.path.join(out_dir, "sweep_results.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(records, f, indent=2, ensure_ascii=False)
            print(f"  原始数据 → {json_path}")

            # 保存 CSV 汇总
            csv_path = os.path.join(out_dir, "sweep_summary.csv")
            save_summary(records, csv_path)
            print(f"  汇总 → {csv_path}")

            # 分析
            if bench == "cluster":
                analyze_cluster(records, out_dir)
            elif bench == "wall":
                analyze_wall(records, out_dir)
            elif bench == "ground":
                analyze_ground(records, out_dir)

        except Exception as e:
            print(f"  错误: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\nDone! 图表输出到 {OUTPUT_BASE}/<bench>/viz/")


if __name__ == "__main__":
    main()
