#!/usr/bin/env python3
"""
Bench 级联流水线：快速测试 → 全量测试 → 分析。
每个阶段只管理自己目录下的文件。

用法：
    uv run python scripts/bench_pipeline.py [--tasks ground,cluster,wall] [--quick-only]

流程：
  1. 快速测试：对所有任务/策略执行 1 帧测试
  2. 保存快速测试快照到 analysis/quick/
  3. 全量测试：对未被跳过的策略执行多帧测试
  4. 保存全量测试快照到 analysis/full/
  5. 交叉对比到 analysis/cross/
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BENCH_DIR = Path(__file__).resolve().parent.parent
OUTPUT_BASE = BENCH_DIR / "output" / "bench"
ANALYSIS_DIR = OUTPUT_BASE / "analysis"

TASKS = ["ground", "cluster", "wall", "denoise"]

# ── 中文 ───────────────────────────────────────────────────

CHINESE_FONTS = [
    "Microsoft YaHei", "SimHei", "WenQuanYi Micro Hei",
    "Noto Sans CJK SC", "Source Han Sans CN",
    "PingFang SC", "Hiragino Sans GB", "STHeiti",
]

def setup_chinese_font():
    for font in CHINESE_FONTS:
        try:
            matplotlib.font_manager.findfont(font, fallback_to_default=False)
            plt.rcParams["font.sans-serif"] = [font]
            plt.rcParams["axes.unicode_minus"] = False
            return
        except Exception:
            continue
    print("WARN: 未找到中文字体", file=sys.stderr)
setup_chinese_font()

# ── 工具函数 ────────────────────────────────────────────────

PROCESS_GAP_SECS = 3  # 进程间间隔，给系统回收内存的时间

def bench_exe_path(task: str, release: bool = False) -> str:
    """已编译的 bench 二进制路径。"""
    mode_dir = "release" if release else "debug"
    exe = f"target/{mode_dir}/examples/{task}_bench"
    return exe + (".exe" if os.name == "nt" else "")

def build_benches(tasks: list[str], release: bool = False) -> bool:
    """一次编译所有 bench 二进制，避免并发时 cargo target 锁竞争。"""
    cmd = ["cargo", "build"]
    if release:
        cmd.append("--release")
    for t in tasks:
        cmd += ["--example", f"{t}_bench"]
    print(f"\n  >>> 编译 bench 二进制: {' '.join(cmd)}")
    try:
        r = subprocess.run(cmd, cwd=str(BENCH_DIR), capture_output=True,
                           text=True, encoding="utf-8", errors="replace", timeout=600)
        if r.returncode != 0:
            err = r.stderr.strip()
            if err:
                print(f"  STDERR: {err}", file=sys.stderr)
            return False
        print(" 编译完成")
        return True
    except Exception as e:
        print(f"  编译失败: {e}", file=sys.stderr)
        return False

def run_bench(task: str, mode: str, release: bool = False) -> bool:
    """（旧接口）通过 cargo run 执行 bench，向后兼容。"""
    global LAST_TASK_TIME
    cmd = ["cargo", "run"]
    if release:
        cmd.append("--release")
    cmd += ["--example", f"{task}_bench", "--", f"--mode={mode}"]

    gap = PROCESS_GAP_SECS - (time.time() - LAST_TASK_TIME)
    if gap > 0:
        time.sleep(gap)

    print(f"\n  >>> {task} {mode}: {' '.join(cmd)}")
    for attempt in range(2):
        try:
            r = subprocess.run(cmd, cwd=str(BENCH_DIR), capture_output=True,
                               text=True, encoding="utf-8", errors="replace", timeout=600)
            print(r.stdout)
            LAST_TASK_TIME = time.time()
            if r.returncode != 0:
                err = r.stderr.strip()
                if err:
                    print(f"  STDERR: {err}", file=sys.stderr)
                if "memory allocation" in err and attempt == 0:
                    print(f"  检测到内存分配失败，等待 {PROCESS_GAP_SECS}s 后重试...")
                    time.sleep(PROCESS_GAP_SECS)
                    continue
                return False
            return True
        except subprocess.TimeoutExpired:
            print(f"  超时: {task} {mode}", file=sys.stderr)
            LAST_TASK_TIME = time.time()
            return False
        except Exception as e:
            print(f"  错误: {e}", file=sys.stderr)
            LAST_TASK_TIME = time.time()
            return False
    return False

def run_bench_binary(task: str, mode: str, release: bool = False) -> tuple[bool, str]:
    """直接运行已编译的 bench 二进制（无 cargo 介入，适合并发）。

    Returns (success, stdout_output) 以便调用方控制打印时机。
    """
    global LAST_TASK_TIME
    exe = os.path.abspath(bench_exe_path(task, release))
    if not os.path.exists(exe):
        print(f"  二进制不存在 (先编译): {exe}")
        return False, ""

    # CliArgs 仅支持 --key=value 格式，不支持空格分隔
    cmd = [exe, f"--mode={mode}"]

    gap = PROCESS_GAP_SECS - (time.time() - LAST_TASK_TIME)
    if gap > 0:
        time.sleep(gap)

    for attempt in range(2):
        try:
            r = subprocess.run(cmd, cwd=str(BENCH_DIR), capture_output=True,
                               text=True, encoding="utf-8", errors="replace", timeout=600)
            LAST_TASK_TIME = time.time()
            if r.returncode != 0:
                err = r.stderr.strip()
                if err:
                    print(f"  STDERR: {err}", file=sys.stderr)
                if "memory allocation" in err and attempt == 0:
                    print(f"  检测到内存分配失败，等待 {PROCESS_GAP_SECS}s 后重试...")
                    time.sleep(PROCESS_GAP_SECS)
                    continue
                return False, r.stdout
            return True, r.stdout
        except subprocess.TimeoutExpired:
            print(f"  超时: {task} {mode}", file=sys.stderr)
            LAST_TASK_TIME = time.time()
            return False, ""
        except Exception as e:
            print(f"  错误: {e}", file=sys.stderr)
            LAST_TASK_TIME = time.time()
            return False, ""
    return False, ""

LAST_TASK_TIME = 0.0

def collect_results(task: str) -> list[dict]:
    """读取 output/bench/{task}/*/info.json，返回扁平化结果列表。"""
    results = []
    task_dir = OUTPUT_BASE / task
    if not task_dir.is_dir():
        return results
    for strategy_dir in sorted(task_dir.iterdir()):
        if not strategy_dir.is_dir():
            continue
        info_path = strategy_dir / "info.json"
        if not info_path.exists():
            continue
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
            for entry in info.get("results", []):
                entry["strategy"] = info.get("strategy", strategy_dir.name)
                entry["mode"] = info.get("mode", "")
                results.append(entry)
        except Exception as e:
            print(f"  WARN: {info_path} - {e}", file=sys.stderr)
    return results

def clean_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)

def fmt_ms(ms: float) -> str:
    return f"{ms:.1f}"

# ── 图表 ───────────────────────────────────────────────────

def plot_speed_bar(results: list[dict], title: str, out: Path):
    if not results:
        return
    grp: dict[str, list[float]] = {}
    for r in results:
        grp.setdefault(r.get("strategy", "?"), []).append(r.get("avg_ms", 0))
    names = sorted(grp, key=lambda k: np.median(grp[k]))
    avgs = [np.mean(grp[k]) for k in names]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(names)))
    fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.35)))
    bars = ax.barh(range(len(names)), avgs, color=colors, height=0.6)
    for bar, v in zip(bars, avgs):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                fmt_ms(v), va="center", fontsize=8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("平均耗时 (ms)")
    ax.set_title(title)
    ax.invert_yaxis()
    ax.axvline(100, color="red", linestyle="--", alpha=0.5, linewidth=0.8, label="100ms 阈值")
    if any(a > 80 for a in avgs):
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [图] {out.name}")

def plot_speed_scatter(results: list[dict], title: str, out: Path):
    if not results:
        return
    strategies = sorted({r.get("strategy", "?") for r in results})
    colors = plt.cm.tab10(np.linspace(0, 1, len(strategies)))
    fig, ax = plt.subplots(figsize=(10, 6))
    for si, s_name in enumerate(strategies):
        pts = [r for r in results if r.get("strategy") == s_name]
        xs = [r.get("avg_ms", 0) for r in pts]
        ys = [r.get("frame_count", 0) for r in pts]
        ax.scatter(xs, ys, c=[colors[si]], label=s_name, alpha=0.6, s=30)
    ax.set_xlabel("平均耗时 (ms)")
    ax.set_ylabel("帧数")
    ax.set_title(title)
    ax.axvline(100, color="red", linestyle="--", alpha=0.5, linewidth=0.8)
    if len(strategies) <= 10:
        ax.legend(fontsize=7, loc="best", ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [图] {out.name}")

# ── 地面分析图 ──────────────────────────────────────────

def plot_ground_ratio_vs_speed(results: list[dict], title: str, out: Path):
    """地面策略：ground_ratio vs avg_ms 散点，看速度-精度权衡。"""
    strategies = sorted({r.get("strategy", "?") for r in results if r.get("extra", {}).get("ground_ratio") is not None})
    if not strategies:
        return
    colors = plt.cm.tab10(np.linspace(0, 1, len(strategies)))
    markers = ["o", "s", "D", "^", "v", "p"]
    fig, ax = plt.subplots(figsize=(10, 7))
    for si, s_name in enumerate(strategies):
        pts = [r for r in results if r.get("strategy") == s_name]
        xs = [r.get("avg_ms", 0) for r in pts]
        ys = [r.get("extra", {}).get("ground_ratio", 0) for r in pts]
        labels_parts = []
        for r in pts:
            p = r.get("params", {})
            labels_parts.append(", ".join(f"{k}={v}" for k, v in sorted(p.items())))
        sc = ax.scatter(xs, ys, c=[colors[si]], marker=markers[si % len(markers)],
                        label=s_name, alpha=0.7, s=60, zorder=5)
        # 为每个点标注参数
        for x, y, lbl in zip(xs, ys, labels_parts):
            ax.annotate(lbl, (x, y), textcoords="offset points", xytext=(4, 4),
                        fontsize=6, alpha=0.8)
    ax.set_xlabel("平均耗时 (ms)")
    ax.set_ylabel("地面占比 (%)")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [图] {out.name}")

def plot_ground_sweep(results: list[dict], out_dir: Path):
    """各策略关键参数的敏感性曲线（折线图）。"""
    strategies = sorted({r.get("strategy", "?") for r in results if r.get("extra", {}).get("ground_ratio") is not None})
    for s_name in strategies:
        pts = [r for r in results if r.get("strategy") == s_name]
        if not pts:
            continue
        # 找到主要的 sweep 参数（变化最多的那个）
        param_keys = set()
        for r in pts:
            param_keys.update(r.get("params", {}).keys())
        if not param_keys:
            continue
        # 按变化最多的参数排序
        param_variants = {k: len(set(r.get("params", {}).get(k) for r in pts)) for k in param_keys}
        main_param = max(param_variants, key=param_variants.get)
        # 固定其他参数，展示 main_param 对 ground_ratio 和 speed 的影响
        other_params = [k for k in param_keys if k != main_param and param_variants[k] > 0]
        # 按 main_param 排序
        pts_sorted = sorted(pts, key=lambda r: r.get("params", {}).get(main_param, 0))
        main_vals = [r.get("params", {}).get(main_param) for r in pts_sorted]
        ratios = [r.get("extra", {}).get("ground_ratio", 0) for r in pts_sorted]
        speeds = [r.get("avg_ms", 0) for r in pts_sorted]
        fig, ax1 = plt.subplots(figsize=(9, 5))
        color_ratio = "tab:blue"
        color_speed = "tab:red"
        l1 = ax1.plot(main_vals, ratios, "o-", color=color_ratio, label="地面占比 (%)", linewidth=1.5)
        ax1.set_xlabel(main_param)
        ax1.set_ylabel("地面占比 (%)", color=color_ratio)
        ax1.tick_params(axis="y", labelcolor=color_ratio)
        ax2 = ax1.twinx()
        l2 = ax2.plot(main_vals, speeds, "s--", color=color_speed, label="耗时 (ms)", linewidth=1.5)
        ax2.set_ylabel("耗时 (ms)", color=color_speed)
        ax2.tick_params(axis="y", labelcolor=color_speed)
        # 在图例中标注固定的其他参数
        fixed_info = ""
        if other_params:
            for r in pts_sorted:
                p = r.get("params", {})
                fixed_info = ", ".join(f"{k}={p[k]}" for k in other_params if k in p)
                break
        title = f"{s_name} 参数敏感性"
        if fixed_info:
            title += f" (固定: {fixed_info})"
        ax1.set_title(title)
        lines = l1 + l2
        ax1.legend(lines, [l.get_label() for l in lines], loc="upper left")
        ax1.grid(True, alpha=0.3)
        fig.tight_layout()
        out_path = out_dir / f"{s_name}_sweep.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  [图] {out_path.name}")

# ── 聚类分析图 ──────────────────────────────────────────


def plot_cluster_detail(results: list[dict], title: str, out: Path):
    """聚类细节：带 std 误差条的簇数/人检数/簇大小的多面板图。"""
    strategies = sorted({r.get("strategy", "?") for r in results if r.get("extra", {}).get("avg_clusters") is not None})
    if not strategies:
        return
    colors = plt.cm.tab10(np.linspace(0, 1, len(strategies)))
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    # 子图 1：簇数（带 std 误差条或 min~max 范围）
    ax = axes[0]
    for si, s_name in enumerate(strategies):
        pts = [r for r in results if r.get("strategy") == s_name]
        xs = [r.get("avg_ms", 0) for r in pts]
        ys = [r.get("extra", {}).get("avg_clusters", 0) for r in pts]
        yerr = [r.get("extra", {}).get("std_clusters", 0) for r in pts]
        ax.errorbar(xs, ys, yerr=yerr, fmt="o", c=colors[si], alpha=0.5, capsize=2)
        ax.scatter(xs, ys, c=[colors[si]], label=s_name if si == 0 else "", alpha=0.7, s=30, zorder=5)
    ax.set_xlabel("平均耗时 (ms)"); ax.set_ylabel("平均簇数")
    ax.set_title("簇数 (标准差)"); ax.grid(True, alpha=0.3)
    # 子图 2：噪声数（带 std 误差条）
    ax = axes[1]
    for si, s_name in enumerate(strategies):
        pts = [r for r in results if r.get("strategy") == s_name]
        xs = [r.get("avg_ms", 0) for r in pts]
        ys = [r.get("extra", {}).get("avg_noise", 0) for r in pts]
        yerr = [r.get("extra", {}).get("std_noise", 0) for r in pts]
        ax.errorbar(xs, ys, yerr=yerr, fmt="o", c=colors[si], alpha=0.5, capsize=2)
        ax.scatter(xs, ys, c=[colors[si]], label=s_name if si == 0 else "", alpha=0.7, s=30, zorder=5)
    ax.set_xlabel("平均耗时 (ms)"); ax.set_ylabel("平均噪声数")
    ax.set_title("噪声数 (标准差)"); ax.grid(True, alpha=0.3)
    # 子图 3：簇大小 vs 速度
    ax = axes[2]
    for si, s_name in enumerate(strategies):
        pts = [r for r in results if r.get("strategy") == s_name]
        xs = [r.get("avg_ms", 0) for r in pts]
        ys = [r.get("extra", {}).get("avg_cluster_size", 0) for r in pts]
        ax.scatter(xs, ys, c=[colors[si]], label=s_name if si == 0 else "", alpha=0.7, s=30, zorder=5)
    ax.set_xlabel("平均耗时 (ms)"); ax.set_ylabel("平均簇大小 (pts)")
    ax.set_title("簇大小 vs 速度"); ax.grid(True, alpha=0.3)
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[i], markersize=8)
               for i in range(len(strategies))]
    fig.legend(handles, strategies, loc="lower center", ncol=min(len(strategies), 6), fontsize=8)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0.08, 1, 0.97])
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [图] {out.name}")

def plot_cluster_count_comparison(results: list[dict], title: str, out: Path):
    """聚类策略：avg_clusters vs avg_ms 散点，看聚类数量分布。带 std 误差线。"""
    strategies = sorted({r.get("strategy", "?") for r in results if r.get("extra", {}).get("avg_clusters") is not None})
    if not strategies:
        return
    colors = plt.cm.tab10(np.linspace(0, 1, len(strategies)))
    markers = ["o", "s", "D", "^", "v", "p"]
    fig, ax = plt.subplots(figsize=(10, 7))
    for si, s_name in enumerate(strategies):
        pts = [r for r in results if r.get("strategy") == s_name]
        xs = [r.get("avg_ms", 0) for r in pts]
        ys = [r.get("extra", {}).get("avg_clusters", 0) for r in pts]
        yerr = [r.get("extra", {}).get("std_clusters", 0) for r in pts]
        sizes = [max(20, min(r.get("extra", {}).get("avg_cluster_size", 5) * 2, 200)) for r in pts]
        ax.errorbar(xs, ys, yerr=yerr, fmt="none", ecolor=colors[si], alpha=0.3, capsize=2)
        ax.scatter(xs, ys, c=[colors[si]], marker=markers[si % len(markers)],
                   label=s_name, alpha=0.7, s=sizes, zorder=5)
        labels_parts = []
        for r in pts:
            p = r.get("params", {})
            labels_parts.append(", ".join(f"{k}={v}" for k, v in sorted(p.items())))
        for x, y, lbl in zip(xs, ys, labels_parts):
            ax.annotate(lbl, (x, y), textcoords="offset points", xytext=(4, 4),
                        fontsize=5.5, alpha=0.8)
    ax.set_xlabel("平均耗时 (ms)")
    ax.set_ylabel("聚类数量")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    # 图例：点大小代表簇大小
    h1 = ax.scatter([], [], s=16, c="gray", alpha=0.5)
    h2 = ax.scatter([], [], s=60, c="gray", alpha=0.5)
    h3 = ax.scatter([], [], s=120, c="gray", alpha=0.5)
    leg2 = ax.legend([h1, h2, h3], ["小簇 (~5)", "中簇 (~30)", "大簇 (~80)"],
                     loc="lower right", fontsize=7, title="簇大小", title_fontsize=8)
    ax.add_artist(leg2)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [图] {out.name}")


def plot_stage_comparison(results: list[dict], title: str, out: Path, x_key: str = "avg_ms",
                          y_key: str = None, y_label: str = None,
                          err_key: str = None, size_key: str = None, size_label: str = None,
                          annotate: bool = True):
    """通用多策略散点对比图，适用于所有三个阶段。

    Args:
        results: 结果列表
        title: 图表标题
        out: 输出路径
        x_key: x 轴数据键名 (默认 avg_ms)
        y_key: y 轴数据键名 (extra 中)
        y_label: y 轴标签
        err_key: 误差线键名 (extra 中，可选)
        size_key: 点大小键名 (extra 中，可选)
        size_label: 点大小图例标签
        annotate: 是否标注参数
    """
    if y_key is None:
        return
    strategies = sorted({r.get("strategy", "?") for r in results if r.get("extra", {}).get(y_key) is not None})
    if not strategies:
        return
    colors = plt.cm.tab10(np.linspace(0, 1, len(strategies)))
    markers = ["o", "s", "D", "^", "v", "p", "<", ">"]
    fig, ax = plt.subplots(figsize=(10, 7))
    for si, s_name in enumerate(strategies):
        pts = [r for r in results if r.get("strategy") == s_name]
        xs = [r.get(x_key, 0) for r in pts]
        ys = [r.get("extra", {}).get(y_key, 0) for r in pts]
        if err_key:
            yerr = [max(0, r.get("extra", {}).get(err_key, 0)) for r in pts]
            ax.errorbar(xs, ys, yerr=yerr, fmt="none", ecolor=colors[si], alpha=0.3, capsize=2)
        sizes = 60
        if size_key:
            sizes = [max(20, min(r.get("extra", {}).get(size_key, 10) * 3, 250)) for r in pts]
        ax.scatter(xs, ys, c=[colors[si]], marker=markers[si % len(markers)],
                   label=s_name, alpha=0.7, s=sizes, zorder=5)
        if annotate:
            labels_parts = []
            for r in pts:
                p = r.get("params", {})
                labels_parts.append(", ".join(f"{k}={v}" for k, v in sorted(p.items())))
            for x, y, lbl in zip(xs, ys, labels_parts):
                ax.annotate(lbl, (x, y), textcoords="offset points", xytext=(4, 4),
                            fontsize=5.5, alpha=0.8)
    ax.set_xlabel("平均耗时 (ms)")
    ax.set_ylabel(y_label or y_key)
    ax.set_title(title)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    if size_key:
        vals = [r.get("extra", {}).get(size_key, 0) for r in results]
        vmin, vmax = min(vals), max(vals)
        h1 = ax.scatter([], [], s=max(20, min(vmin * 3, 250)), c="gray", alpha=0.5)
        h2 = ax.scatter([], [], s=max(20, min((vmin + vmax) / 2 * 3, 250)), c="gray", alpha=0.5)
        h3 = ax.scatter([], [], s=max(20, min(vmax * 3, 250)), c="gray", alpha=0.5)
        leg2 = ax.legend([h1, h2, h3],
                         [f"低 {size_label or size_key} ({vmin:.0f})",
                          f"中 {size_label or size_key} ({(vmin+vmax)/2:.0f})",
                          f"高 {size_label or size_key} ({vmax:.0f})"],
                         loc="lower right", fontsize=7, title=size_label or size_key, title_fontsize=8)
        ax.add_artist(leg2)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [图] {out.name}")


def plot_wall_ratio_analysis(results: list[dict], title: str, out: Path):
    """墙体策略：wall_ratio vs avg_ms 散点，看墙体提取效率。"""
    strategies = sorted({r.get("strategy", "?") for r in results if r.get("extra", {}).get("wall_ratio") is not None})
    if not strategies:
        return
    colors = plt.cm.tab10(np.linspace(0, 1, len(strategies)))
    markers = ["o", "s", "D", "^", "v", "p"]
    fig, ax = plt.subplots(figsize=(10, 7))
    for si, s_name in enumerate(strategies):
        pts = [r for r in results if r.get("strategy") == s_name]
        xs = [r.get("avg_ms", 0) for r in pts]
        ys = [r.get("extra", {}).get("wall_ratio", 0) for r in pts]
        obstacles = [r.get("extra", {}).get("avg_obstacles", 0) for r in pts]
        labels_parts = []
        for r in pts:
            p = r.get("params", {})
            labels_parts.append(", ".join(f"{k}={v}" for k, v in sorted(p.items())))
        sc = ax.scatter(xs, ys, c=[colors[si]], marker=markers[si % len(markers)],
                        label=s_name, alpha=0.7, s=[max(30, o * 5) for o in obstacles], zorder=5)
        for x, y, lbl in zip(xs, ys, labels_parts):
            ax.annotate(lbl, (x, y), textcoords="offset points", xytext=(4, 4),
                        fontsize=5.5, alpha=0.8)
    ax.set_xlabel("平均耗时 (ms)")
    ax.set_ylabel("墙体占比 (%)")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [图] {out.name}")

def plot_cluster_noise_analysis(results: list[dict], title: str, out: Path):
    """聚类策略：avg_noise vs avg_clusters 散点，看分割质量。"""
    strategies = sorted({r.get("strategy", "?") for r in results if r.get("extra", {}).get("avg_noise") is not None})
    if not strategies:
        return
    colors = plt.cm.tab10(np.linspace(0, 1, len(strategies)))
    markers = ["o", "s", "D", "^", "v", "p"]
    fig, ax = plt.subplots(figsize=(10, 7))
    for si, s_name in enumerate(strategies):
        pts = [r for r in results if r.get("strategy") == s_name]
        xs = [r.get("extra", {}).get("avg_clusters", 0) for r in pts]
        ys = [r.get("extra", {}).get("avg_noise", 0) for r in pts]
        speeds = [r.get("avg_ms", 0) for r in pts]
        labels_parts = []
        for r in pts:
            p = r.get("params", {})
            labels_parts.append(", ".join(f"{k}={v}" for k, v in sorted(p.items())))
        ax.scatter(xs, ys, c=[colors[si]], marker=markers[si % len(markers)],
                   label=s_name, alpha=0.7, s=60, zorder=5)
        for x, y, lbl in zip(xs, ys, labels_parts):
            ax.annotate(lbl, (x, y), textcoords="offset points", xytext=(4, 4),
                        fontsize=5.5, alpha=0.8)
    ax.set_xlabel("簇数量")
    ax.set_ylabel("噪声点数量")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    # 理想区域标注（低簇数 + 低噪声）
    ax.annotate("理想区\n少簇+低噪", xy=(10, 500), fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.5))
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [图] {out.name}")


def plot_cross_comparison(qr: list[dict], fr: list[dict], out: Path):
    if not qr or not fr:
        return
    def avg_by_strategy(data):
        d: dict[str, list[float]] = {}
        for r in data:
            d.setdefault(r.get("strategy", "?"), []).append(r.get("avg_ms", 0))
        return {k: np.mean(v) for k, v in d.items()}
    q = avg_by_strategy(qr)
    f = avg_by_strategy(fr)
    common = sorted(set(q) & set(f))
    if not common:
        return
    qv = [q[k] for k in common]
    fv = [f[k] for k in common]
    fig, ax = plt.subplots(figsize=(8, 6))
    x = range(len(common))
    w = 0.35
    ax.bar([i - w/2 for i in x], qv, w, label="快速测试", alpha=0.8)
    ax.bar([i + w/2 for i in x], fv, w, label="全量测试", alpha=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(common, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("平均耗时 (ms)")
    ax.set_title("快速 vs 全量 速度对比")
    ax.legend()
    for i, (qvv, fvv) in enumerate(zip(qv, fv)):
        if qvv > 0:
            ax.text(i, max(qvv, fvv) + 1, f"{(fvv - qvv)/qvv*100:+.0f}%",
                    ha="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [图] {out.name}")

def write_stats_table(results: list[dict], out: Path):
    if not results:
        return
    grp: dict[str, list[float]] = {}
    for r in results:
        grp.setdefault(r.get("strategy", "?"), []).append(r.get("avg_ms", 0))
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["策略", "参数数", "最快(ms)", "最慢(ms)", "平均(ms)", "中位(ms)"])
        for name in sorted(grp):
            vals = grp[name]
            w.writerow([name, len(vals), fmt_ms(min(vals)), fmt_ms(max(vals)),
                        fmt_ms(np.mean(vals)), fmt_ms(np.median(vals))])
    print(f"  [CSV] {out.name}")

def print_toml_summary(tasks: list[str]):
    import datetime
    print(f"\n  TOML 统计摘要:")
    print(f"  {'任务':<6} {'策略':<16} {'最快':>7} {'最慢':>7} {'平均':>7} {'中位':>7}  {'最近运行'}")
    print(f"  {'-'*70}")
    for task in tasks:
        toml_dir = BENCH_DIR / "config" / "bench" / task
        for tf in sorted(toml_dir.glob("*.toml")):
            try:
                import tomllib
                data = tomllib.loads(tf.read_text(encoding="utf-8"))
                st = data.get("stats", {})
                lr = st.get("last_run", "")
                if lr:
                    try:
                        dt = datetime.datetime.fromtimestamp(int(lr))
                        lr_str = dt.strftime("%m-%d %H:%M")
                    except (ValueError, OSError):
                        lr_str = str(lr)
                else:
                    lr_str = "-"
                print(f"  {task:<6} {tf.stem:<16} {st.get('fastest_ms',0):>6.1f} {st.get('slowest_ms',0):>6.1f} {st.get('avg_ms',0):>6.1f} {st.get('median_ms',0):>6.1f}  [{lr_str}]")
            except Exception:
                pass

# ── 主流程 ────────────────────────────────────────────────

def run_pipeline(tasks: list[str], quick_only: bool, release: bool = False):
    print("\n" + "█"*60)
    print("  Bench 级联流水线")
    print(f"  任务: {', '.join(tasks)}")
    print("█"*60)

    # ── 第 0 步：一次编译所有 bench 二进制 ──
    if not build_benches(tasks, release):
        print("  编译失败，终止流水线", file=sys.stderr)
        return

    # ── 第 1 步：快速测试（串行） ──
    print("\n" + "="*60)
    print("  阶段 1/3: 快速测试")
    print("="*60)
    quick_snapshots: dict[str, list[dict]] = {}
    for task in tasks:
        ok, output = run_bench_binary(task, "quick", release)
        print(f"  >>> {task} quick:\n{output.strip()}\n")
        quick_snapshots[task] = collect_results(task)

    # 保存快速测试分析
    for task in tasks:
        results = quick_snapshots[task]
        if not results:
            continue
        td = ANALYSIS_DIR / "quick" / task
        td.mkdir(parents=True, exist_ok=True)
        write_stats_table(results, td / "stats.csv")
        plot_speed_bar(results, f"{task} 快速测试", td / "speed_bar.png")

    if quick_only:
        print_toml_summary(tasks)
        print(f"\n快速测试图表: {ANALYSIS_DIR / 'quick'}")
        return

    # ── 第 2 步：全量测试（task 间并发，直接运行已编译的二进制） ──
    print("\n" + "="*60)
    print("  阶段 2/3: 全量测试（{} 任务并发）".format(len(tasks)))
    print("="*60)
    full_results: dict[str, list[dict]] = {}

    with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
        fut_map = {pool.submit(run_bench_binary, t, "full", release): t for t in tasks}
        for f in as_completed(fut_map):
            task = fut_map[f]
            ok, output = f.result()
            output = output.strip()
            if output:
                print(f"\n── {task} full {'─'*52}\n{output}\n")
            else:
                print(f"\n── {task} full {'─'*52}")
            if ok:
                full_results[task] = collect_results(task)
            else:
                print(f"  WARN: {task} 全量测试失败", file=sys.stderr)

    for task in tasks:
        full_results.setdefault(task, [])

    # 保存全量测试分析
    for task in tasks:
        results = full_results[task]
        if not results:
            continue
        td = ANALYSIS_DIR / "full" / task
        td.mkdir(parents=True, exist_ok=True)
        write_stats_table(results, td / "stats.csv")
        plot_speed_bar(results, f"{task} 全量测试", td / "speed_bar.png")
        plot_speed_scatter(results, f"{task} 全量测试", td / "speed_scatter.png")
        # 地面专属分析
        if task == "ground":
            plot_ground_ratio_vs_speed(results, "地面策略 精度-速度权衡", td / "ground_ratio_vs_speed.png")
            plot_ground_sweep(results, td)
        # 聚类专属分析
        if task == "cluster":
            plot_cluster_count_comparison(results, "聚类策略 聚类数量对比", td / "cluster_count_comparison.png")
            plot_cluster_detail(results, "聚类策略 详细指标", td / "cluster_detail.png")
            plot_cluster_noise_analysis(results, "聚类策略 噪声-簇数分析", td / "cluster_noise_analysis.png")
            # 通用对比图：可用于 all three stages
            plot_stage_comparison(results, "聚类策略 速度-聚类数", td / "cluster_speed_vs_clusters.png",
                                  y_key="avg_clusters", y_label="平均聚类数",
                                  err_key="std_clusters", size_key="avg_noise", size_label="噪声数")
        # 墙体专属分析
        if task == "wall":
            plot_wall_ratio_analysis(results, "墙体策略 效率分析", td / "wall_ratio_analysis.png")
            plot_stage_comparison(results, "墙体策略 速度-墙体占比", td / "wall_speed_vs_wallratio.png",
                                  y_key="wall_ratio", y_label="墙体占比 (%)",
                                  size_key="avg_obstacles", size_label="障碍物数")
        # 地面专属分析
        if task == "ground":
            plot_stage_comparison(results, "地面策略 速度-地面占比", td / "ground_speed_vs_ratio.png",
                                  y_key="ground_ratio", y_label="地面占比 (%)",
                                  err_key=None, size_key="avg_input", size_label="输入点量")

        # 降噪专属分析
        if task == "denoise":
            plot_stage_comparison(results, "降噪策略 保留率-速度", td / "denoise_retention_vs_speed.png",
                                  y_key="retention_pct", y_label="保留率 (%)",
                                  size_key="avg_input", size_label="输入点量")

    # ── 第 3 步：交叉对比 ──
    print("\n" + "="*60)
    print("  阶段 3/3: 交叉分析")
    print("="*60)
    cross_dir = ANALYSIS_DIR / "cross"
    clean_dir(cross_dir)
    for task in tasks:
        qr = quick_snapshots.get(task, [])
        fr = full_results.get(task, [])
        if qr and fr:
            plot_cross_comparison(qr, fr, cross_dir / f"{task}_comparison.png")

    # 跨任务聚合对比：合并所有地面策略 vs 合并所有聚类策略
    all_ground = full_results.get("ground", [])
    all_cluster = full_results.get("cluster", [])
    if all_ground:
        plot_ground_ratio_vs_speed(all_ground, "地面提取策略总对比（全量）",
                                    cross_dir / "ground_all_strategies.png")
    if all_cluster:
        plot_cluster_count_comparison(all_cluster, "后聚类策略 聚类数量对比（全量）",
                                       cross_dir / "cluster_count_all.png")
        plot_cluster_noise_analysis(all_cluster, "后聚类策略 噪声分析（全量）",
                                     cross_dir / "cluster_noise_all.png")

    all_wall = full_results.get("wall", [])
    if all_wall:
        plot_wall_ratio_analysis(all_wall, "墙体策略总对比（全量）",
                                  cross_dir / "wall_all_strategies.png")

    all_denoise = full_results.get("denoise", [])
    if all_denoise:
        plot_stage_comparison(all_denoise, "降噪策略总对比（全量）",
                               cross_dir / "denoise_all_strategies.png",
                               y_key="retention_pct", y_label="保留率 (%)",
                               size_key="avg_input", size_label="输入点量")

    # ── 汇总 ──
    print("\n" + "█"*60)
    print("  汇总")
    print("█"*60)
    print_toml_summary(tasks)
    print(f"\n  快速测试: {ANALYSIS_DIR / 'quick'}")
    print(f"  全量测试: {ANALYSIS_DIR / 'full'}")
    print(f"  交叉对比: {ANALYSIS_DIR / 'cross'}")
    print(f"\n  输出数据: {OUTPUT_BASE}/{{task}}/{{strategy}}/")

def run_analysis_only(tasks: list[str]):
    """仅从已有的 output 数据重新生成分析图，不执行 bench。"""
    print("\n" + "█"*60)
    print("  分析模式 (从已有数据重新生成图表)")
    print(f"  任务: {', '.join(tasks)}")
    print("█"*60)

    full_results: dict[str, list[dict]] = {}
    for task in tasks:
        full_results[task] = collect_results(task)

    # 重新生成所有分析图
    for task in tasks:
        results = full_results[task]
        if not results:
            print(f"  WARN: {task} 无数据", file=sys.stderr)
            continue
        td = ANALYSIS_DIR / "full" / task
        td.mkdir(parents=True, exist_ok=True)
        write_stats_table(results, td / "stats.csv")
        plot_speed_bar(results, f"{task} 全量测试", td / "speed_bar.png")
        plot_speed_scatter(results, f"{task} 全量测试", td / "speed_scatter.png")
        if task == "ground":
            plot_ground_ratio_vs_speed(results, "地面策略 精度-速度权衡", td / "ground_ratio_vs_speed.png")
            plot_ground_sweep(results, td)
        if task == "cluster":
            plot_cluster_count_comparison(results, "聚类策略 聚类数量对比", td / "cluster_count_comparison.png")
            plot_cluster_detail(results, "聚类策略 详细指标", td / "cluster_detail.png")
            plot_cluster_noise_analysis(results, "聚类策略 噪声-簇数分析", td / "cluster_noise_analysis.png")
            plot_stage_comparison(results, "聚类策略 速度-聚类数", td / "cluster_speed_vs_clusters.png",
                                  y_key="avg_clusters", y_label="平均聚类数",
                                  err_key="std_clusters", size_key="avg_noise", size_label="噪声数")
        if task == "wall":
            plot_wall_ratio_analysis(results, "墙体策略 效率分析", td / "wall_ratio_analysis.png")
            plot_stage_comparison(results, "墙体策略 速度-墙体占比", td / "wall_speed_vs_wallratio.png",
                                  y_key="wall_ratio", y_label="墙体占比 (%)",
                                  size_key="avg_obstacles", size_label="障碍物数")
        if task == "ground":
            plot_stage_comparison(results, "地面策略 速度-地面占比", td / "ground_speed_vs_ratio.png",
                                  y_key="ground_ratio", y_label="地面占比 (%)",
                                  err_key=None, size_key="avg_input", size_label="输入点量")

        if task == "denoise":
            plot_stage_comparison(results, "降噪策略 保留率-速度", td / "denoise_retention_vs_speed.png",
                                  y_key="retention_pct", y_label="保留率 (%)",
                                  size_key="avg_input", size_label="输入点量")

    cross_dir = ANALYSIS_DIR / "cross"
    cross_dir.mkdir(parents=True, exist_ok=True)
    all_ground = full_results.get("ground", [])
    all_cluster = full_results.get("cluster", [])
    all_wall = full_results.get("wall", [])
    if all_ground:
        plot_ground_ratio_vs_speed(all_ground, "地面策略总对比（全量）",
                                    cross_dir / "ground_all_strategies.png")
    if all_cluster:
        plot_cluster_count_comparison(all_cluster, "后聚类策略 聚类数量对比（全量）",
                                       cross_dir / "cluster_count_all.png")
        plot_cluster_detail(all_cluster, "后聚类策略 详细指标（全量）",
                             cross_dir / "cluster_detail_all.png")
        plot_cluster_noise_analysis(all_cluster, "后聚类策略 噪声分析（全量）",
                                     cross_dir / "cluster_noise_all.png")
    if all_wall:
        plot_wall_ratio_analysis(all_wall, "墙体策略总对比（全量）",
                                  cross_dir / "wall_all_strategies.png")

    all_denoise = full_results.get("denoise", [])
    if all_denoise:
        plot_stage_comparison(all_denoise, "降噪策略总对比（全量）",
                               cross_dir / "denoise_all_strategies.png",
                               y_key="retention_pct", y_label="保留率 (%)",
                               size_key="avg_input", size_label="输入点量")

    print(f"\n  全量分析: {ANALYSIS_DIR / 'full'}")
    print(f"  交叉分析: {ANALYSIS_DIR / 'cross'}")

def main():
    parser = argparse.ArgumentParser(description="Bench 级联流水线")
    parser.add_argument("--tasks", default=",".join(TASKS),
                        help=f"任务列表 (默认: {','.join(TASKS)})")
    parser.add_argument("--quick-only", action="store_true",
                        help="仅执行快速测试")
    parser.add_argument("--analysis-only", action="store_true",
                        help="仅从已有数据重新生成分析图，不执行 bench")
    parser.add_argument("--release", action="store_true",
                        help="使用 release 模式编译（内存更小、速度更快）")
    args = parser.parse_args()
    task_list = [t.strip() for t in args.tasks.split(",") if t.strip()]
    if args.analysis_only:
        run_analysis_only(task_list)
    else:
        run_pipeline(task_list, args.quick_only, args.release)

if __name__ == "__main__":
    main()
