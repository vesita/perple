# -*- coding: utf-8 -*-
"""运行 40 次 eval_labeled，收集指标，绘制运行时间 & 准确度曲线"""

import subprocess
import json
import csv
import time
import locale
import sys
import threading
import concurrent.futures
from pathlib import Path

# Python 自身 stdout 输出 UTF-8（解决管道捕获时中文乱码）
sys.stdout.reconfigure(encoding='utf-8')

OUTPUT_DIR = Path("output/batch_40")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_FILE = OUTPUT_DIR / "results.csv"
PLOT_SPEED = OUTPUT_DIR / "fig_speed_curve.png"
PLOT_ACCURACY = OUTPUT_DIR / "fig_accuracy_curve.png"
PLOT_SPATIAL = OUTPUT_DIR / "fig_spatial_curve.png"

N_RUNS = 40
FRAMES = 408
CENTER_DIST = 0.5


def read_metrics(run_dir: Path) -> dict:
    """Read metrics from eval_result.json and eval_result.csv."""
    m = {}

    # JSON — person + spatial metrics
    json_path = run_dir / "eval_result.json"
    if json_path.exists():
        with open(json_path, 'r') as f:
            data = json.load(f)
        m['gt'] = data.get('n_gt', 0)
        m['person_detections'] = data.get('n_detections', 0)
        m['person_tp'] = data.get('tp', 0)
        m['person_fp'] = data.get('fp', 0)
        m['person_fn'] = data.get('fn_', 0)
        m['person_precision'] = data.get('precision', 0) * 100  # JSON stores as ratio
        m['person_recall'] = data.get('recall', 0) * 100
        m['person_f1'] = data.get('f1', 0)
        m['spatial_detections'] = data.get('n_detections_spatial', 0)
        m['spatial_tp'] = data.get('tp_spatial', 0)
        m['spatial_fp'] = data.get('fp_spatial', 0)
        m['spatial_fn'] = data.get('fn_spatial', 0)
        m['spatial_precision'] = data.get('precision_spatial', 0) * 100
        m['spatial_recall'] = data.get('recall_spatial', 0) * 100
        m['spatial_f1'] = data.get('f1_spatial', 0)
        m['person_correct'] = data.get('tp_person', 0)
        # misclassified = spatial TP that are NOT classified as person
        m['misclassified'] = data.get('tp_nonperson', 0)
        # spatial_matched = spatial TP (detections that found a GT match)
        m['spatial_matched'] = data.get('tp_spatial', 0)

    # CSV — elapsed_s
    csv_path = run_dir / "eval_result.csv"
    if csv_path.exists():
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) == 2 and row[0] == 'elapsed_s':
                    m['elapsed_s'] = float(row[1])

    return m


# 预编译 — 避免并发时 cargo 文件锁串行
BINARY = Path("E:/code/perple/target/release/examples/eval_labeled.exe")
if not BINARY.exists():
    subprocess.run(["cargo", "build", "--release", "--example", "eval_labeled"],
                   check=True, cwd="E:/code/perple")
else:
    print(f"Binary already exists: {BINARY}")


def run_single(run_id: int) -> dict:
    """Run eval_labeled binary once and return metrics."""
    run_dir = OUTPUT_DIR / f"run_{run_id + 1:02d}"
    print(f"\n[{run_id + 1}/{N_RUNS}] Running eval_labeled (center-dist={CENTER_DIST}, frames={FRAMES})...")
    t0 = time.time()

    result = subprocess.run(
        [BINARY,
         "--center-dist", str(CENTER_DIST), "--frames", str(FRAMES),
         "--output", str(run_dir)],
        capture_output=True,
        cwd="E:/code/perple"
    )

    elapsed = time.time() - t0
    # Rust 输出为 UTF-8，尝试解码；若失败回退到系统编码（如 cp936）
    raw = result.stdout
    try:
        stdout = raw.decode('utf-8')
    except UnicodeDecodeError:
        sys_enc = locale.getpreferredencoding(do_setlocale=False)
        stdout = raw.decode(sys_enc, errors='replace')

    # Print progress line from output
    for line in stdout.split('\n'):
        if '进度:' in line:
            print(f"  {line.strip()}")

    # Read metrics from JSON/CSV
    metrics = read_metrics(run_dir)
    metrics['run_id'] = run_id + 1
    metrics['wall_clock'] = round(elapsed, 1)

    # Print summary
    if 'person_f1' in metrics and metrics['person_f1'] > 0:
        print(f"  => F1={metrics['person_f1']:.4f}  P={metrics.get('person_precision', 0):.1f}%  "
              f"R={metrics.get('person_recall', 0):.1f}%  "
              f"TP={metrics.get('person_tp', 0)} FP={metrics.get('person_fp', 0)}  "
              f"time={metrics.get('elapsed_s', 0):.1f}s (wall={elapsed:.1f}s)")
    else:
        print(f"  => FAILED to parse metrics")
        print(f"  run_dir: {run_dir}")
        # Fallback: try parsing stdout for timing at least
        import re
        timing = re.search(r'耗时:\s+([\d.]+)s', stdout)
        if timing:
            metrics['elapsed_s'] = float(timing.group(1))
            print(f"  (fallback) elapsed_s={metrics['elapsed_s']}")

    # Save full stdout for debugging
    log_path = OUTPUT_DIR / f"run_{run_id + 1:02d}.log"
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(stdout)
        stderr = result.stderr.decode('utf-8', errors='replace')
        if stderr:
            f.write("\n\n=== STDERR ===\n")
            f.write(stderr)

    return metrics


def write_csv(results: list[dict]):
    fields = [
        'run_id', 'elapsed_s', 'wall_clock',
        'gt', 'person_detections', 'person_tp', 'person_fp', 'person_fn',
        'person_precision', 'person_recall', 'person_f1',
        'spatial_detections', 'spatial_tp', 'spatial_fp', 'spatial_fn',
        'spatial_precision', 'spatial_recall', 'spatial_f1',
        'spatial_matched', 'person_correct', 'misclassified',
    ]
    with open(RESULTS_FILE, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(results)
    print(f"\nResults saved to {RESULTS_FILE}")


def smart_fmt(v):
    """Auto-select decimal places based on magnitude."""
    if abs(v) >= 100: return f"{v:.0f}"
    if abs(v) >= 1: return f"{v:.1f}"
    if abs(v) >= 0.01: return f"{v:.4f}"
    return f"{v:.6f}"


def _setup_chinese_font():
    import matplotlib.pyplot as plt
    # 论文格式：SimHei（黑体）用于中文，Times New Roman 用于英文/数字
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.unicode_minus'] = False


def plot_single_metric(results, runs, values, ylabel, title, filename,
                       color='C0', ylim=None, unit=''):
    """Plot a single metric with mean line, std band, and annotations."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    _setup_chinese_font()

    fig, ax = plt.subplots(figsize=(10, 5))
    mean_val = np.mean(values)
    std_val = np.std(values)

    ax.plot(runs, values, color=color, marker='o', markersize=5, linewidth=1.2, label=title)
    ax.axhline(mean_val, color=color, linestyle='--', linewidth=1, alpha=0.7)
    ax.fill_between(runs, mean_val - std_val, mean_val + std_val,
                    color=color, alpha=0.12, label=f'±1σ ({smart_fmt(std_val)}{unit})')

    # Annotation box
    text = f'均值: {smart_fmt(mean_val)}{unit}\n标准差: {smart_fmt(std_val)}{unit}'
    ax.text(0.97, 0.95, text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='wheat', alpha=0.6))

    ax.set_xlabel('Run', fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.tick_params(labelsize=12)
    ax.set_xlim(0.5, N_RUNS + 0.5)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=11, loc='lower right')

    plt.tight_layout()
    filepath = OUTPUT_DIR / filename
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  √ {filepath.name}")


def plot_results(results: list[dict]):
    import numpy as np

    runs = [r['run_id'] for r in results]

    # ── Person F1 ──
    plot_single_metric(
        results, runs, [r.get('person_f1', 0) for r in results],
        'F1 值', '行人检测 F1 分数 (Person F1)',
        'fig_person_f1.png', color='#d62728', ylim=(0, 1))

    # ── Person Precision ──
    plot_single_metric(
        results, runs, [r.get('person_precision', 0) for r in results],
        '精确率 (%)', '行人检测精确率 (Person Precision)',
        'fig_person_precision.png', color='#2ca02c', ylim=(50, 100), unit='%')

    # ── Person Recall ──
    plot_single_metric(
        results, runs, [r.get('person_recall', 0) for r in results],
        '召回率 (%)', '行人检测召回率 (Person Recall)',
        'fig_person_recall.png', color='#1f77b4', ylim=(0, 100), unit='%')

    # ── Spatial F1 ──
    plot_single_metric(
        results, runs, [r.get('spatial_f1', 0) for r in results],
        'F1 值', '空间匹配 F1 分数 (Spatial F1)',
        'fig_spatial_f1.png', color='#9467bd', ylim=(0, 1))

    # ── TP / FP / FN (combined on one axis, separate lines) ──
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _setup_chinese_font()

    fig, ax = plt.subplots(figsize=(10, 5))
    tps = [r.get('person_tp', 0) for r in results]
    fps = [r.get('person_fp', 0) for r in results]
    fns = [r.get('person_fn', 0) for r in results]

    ax.plot(runs, tps, 'o-', color='#2ca02c', markersize=5, linewidth=1.2, label='TP (正确检测)')
    ax.plot(runs, fps, 's-', color='#d62728', markersize=5, linewidth=1.2, label='FP (误检)')
    ax.plot(runs, fns, '^-', color='#7f7f7f', markersize=5, linewidth=1.2, label='FN (漏检)')

    for vals, color, label in [(tps, '#2ca02c', 'TP'), (fps, '#d62728', 'FP'), (fns, '#7f7f7f', 'FN')]:
        m, s = np.mean(vals), np.std(vals)
        ax.axhline(m, color=color, linestyle='--', linewidth=0.8, alpha=0.5)
        ax.text(0.97, m, f'{label} μ={smart_fmt(m)} σ={smart_fmt(s)}',
                transform=ax.get_yaxis_transform(), fontsize=9,
                va='bottom', ha='left', color=color)

    ax.set_xlabel('运行次数', fontsize=14)
    ax.set_ylabel('计数', fontsize=14)
    ax.set_title('行人检测 TP / FP / FN 分布', fontsize=16, fontweight='bold')
    ax.tick_params(labelsize=12)
    ax.set_xlim(0.5, N_RUNS + 0.5)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=12)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig_tp_fp_fn.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  √ fig_tp_fp_fn.png")

    # ── Runtime ──
    plot_single_metric(
        results, runs, [r.get('elapsed_s', 0) for r in results],
        '耗时 (s)', '单次评估处理耗时 (408 帧)',
        'fig_runtime.png', color='#1f77b4', unit='s')

    # ── Detections count ──
    plot_single_metric(
        results, runs, [r.get('person_detections', 0) for r in results],
        '检测数', 'YOLO 行人检测数量',
        'fig_detections.png', color='#ff7f0e')

    # ── Summary table ──
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')

    keys = [
        ('person_f1', 'Person F1'),
        ('person_precision', 'Person Precision (%)'),
        ('person_recall', 'Person Recall (%)'),
        ('spatial_f1', 'Spatial F1'),
        ('person_tp', 'TP'),
        ('person_fp', 'FP'),
        ('person_fn', 'FN'),
        ('elapsed_s', '耗时 (s)'),
        ('person_detections', '检测数'),
    ]
    cell_text = []
    for key, label in keys:
        vals = [r.get(key, 0) for r in results]
        m, s = np.mean(vals), np.std(vals)
        cell_text.append([label, smart_fmt(m), smart_fmt(s)])

    table = ax.table(cellText=cell_text, colLabels=['指标', '均值', '标准差'],
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(13)
    table.scale(1, 1.8)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#40466e')
            cell.set_text_props(color='white', fontweight='bold')
        elif row % 2 == 0:
            cell.set_facecolor('#f0f0f0')

    ax.set_title('40 次运行评估汇总', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig_summary_table.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  √ fig_summary_table.png")


def print_summary(results: list[dict]):
    """Print final summary statistics."""
    import numpy as np

    print("\n" + "=" * 60)
    print("  40-RUN SUMMARY")
    print("=" * 60)

    for key, label, fmt in [
        ('person_precision', 'Person Precision', '.1f'),
        ('person_recall', 'Person Recall', '.1f'),
        ('person_f1', 'Person F1', '.4f'),
        ('elapsed_s', 'Pipeline Time (s)', '.1f'),
        ('person_detections', 'Person Detections', '.0f'),
        ('person_tp', 'TP', '.0f'),
        ('person_fp', 'FP', '.0f'),
        ('person_fn', 'FN', '.0f'),
        ('spatial_precision', 'Spatial Precision', '.1f'),
        ('spatial_recall', 'Spatial Recall', '.1f'),
        ('spatial_f1', 'Spatial F1', '.4f'),
        ('spatial_matched', 'Spatial Matched', '.0f'),
        ('misclassified', 'Misclassified', '.0f'),
    ]:
        vals = [r.get(key, 0) for r in results]
        mean = np.mean(vals)
        std = np.std(vals)
        if any(c in label for c in ['Precision', 'Recall']):
            print(f"  {label:20s} = {mean:{fmt}}% ± {std:{fmt}}")
        else:
            print(f"  {label:20s} = {mean:{fmt}}  ± {std:{fmt}}")

    print("=" * 60)


def main():
    MAX_WORKERS = 2
    print(f"Running {N_RUNS}x eval_labeled (center-dist={CENTER_DIST}, frames={FRAMES}) — {MAX_WORKERS} 路并行")
    print(f"Output: {OUTPUT_DIR}")

    results = [None] * N_RUNS
    write_lock = threading.Lock()

    def run_and_collect(run_idx):
        metrics = run_single(run_idx)
        with write_lock:
            results[run_idx] = metrics
            valid = [r for r in results if r is not None]
            write_csv(valid)
        return metrics

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(run_and_collect, i) for i in range(N_RUNS)]
        concurrent.futures.wait(futures)
        for f in futures:
            if f.exception():
                print(f"  Worker failed: {f.exception()}")

    results.sort(key=lambda r: r['run_id'] if r else 0)
    valid = [r for r in results if r]
    print_summary(valid)

    print("\nGenerating plots...")
    try:
        plot_results(valid)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Plotting failed: {e}")
        print("Results CSV is available for manual plotting.")

    print(f"\nDone. All outputs in {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()
