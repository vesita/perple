"""运行 40 次 eval_labeled，收集指标，绘制运行时间 & 准确度曲线"""

import subprocess
import json
import csv
import time
import threading
import concurrent.futures
from pathlib import Path

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
BINARY = "E:/code/perple/target/release/examples/eval_labeled.exe"
subprocess.run(["cargo", "build", "--release", "--example", "eval_labeled"],
               check=True, cwd="E:/code/perple")


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
    stdout = result.stdout.decode('utf-8', errors='replace')

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


def plot_results(results: list[dict]):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    runs = [r['run_id'] for r in results]

    # Figure 1: Speed curve (running time per run)
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    ax = axes[0]
    times = [r.get('elapsed_s', 0) for r in results]
    wall = [r.get('wall_clock', 0) for r in results]
    ax.plot(runs, times, 'b-o', label='Pipeline Time (s)', markersize=4, linewidth=1)
    ax.plot(runs, wall, 'c--s', label='Wall Clock (s)', markersize=4, linewidth=1)
    mean_t = np.mean(times)
    ax.axhline(mean_t, color='gray', linestyle=':', alpha=0.7)
    ax.text(0.98, mean_t, f'mean={mean_t:.1f}s', transform=ax.get_yaxis_transform(),
            ha='left', va='bottom', fontsize=8, color='gray')
    ax.set_ylabel('Time (s)')
    ax.set_title(f'Run Time per Eval ({FRAMES} frames)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, N_RUNS + 0.5)

    # Figure 2: Person detections count over runs
    ax = axes[1]
    dets = [r.get('person_detections', 0) for r in results]
    ax.plot(runs, dets, 'm-o', markersize=4, linewidth=1)
    mean_d = np.mean(dets)
    ax.axhline(mean_d, color='gray', linestyle=':', alpha=0.7)
    ax.text(0.98, mean_d, f'mean={mean_d:.0f}', transform=ax.get_yaxis_transform(),
            ha='left', va='bottom', fontsize=8, color='gray')
    ax.set_xlabel('Run #')
    ax.set_ylabel('Person Detections')
    ax.set_title('YOLO Person Detection Count Variability (non-determinism)')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, N_RUNS + 0.5)

    plt.tight_layout()
    plt.savefig(PLOT_SPEED, dpi=150)
    print(f"Speed plot saved to {PLOT_SPEED}")

    # Figure 2: Person accuracy curves (P/R/F1 over runs)
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # Subplot 1: Person P/R/F1
    ax = axes[0]
    precisions = [r.get('person_precision', 0) for r in results]
    recalls = [r.get('person_recall', 0) for r in results]
    f1s = [r.get('person_f1', 0) for r in results]
    ax.plot(runs, precisions, 'g-o', label='Precision', markersize=4, linewidth=1)
    ax.plot(runs, recalls, 'b-o', label='Recall', markersize=4, linewidth=1)
    ax.plot(runs, f1s, 'r-o', label='F1', markersize=4, linewidth=1)
    mean_f1 = np.mean(f1s)
    ax.axhline(mean_f1, color='red', linestyle=':', alpha=0.5)
    ax.text(0.98, mean_f1, f'F1 mean={mean_f1:.3f}', transform=ax.get_yaxis_transform(),
            ha='left', va='bottom', fontsize=8, color='red')
    ax.set_ylabel('Percentage')
    ax.set_title('Person-Only Metrics (Precision / Recall / F1)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)
    ax.set_xlim(0.5, N_RUNS + 0.5)

    # Subplot 2: Spatial P/R/F1
    ax = axes[1]
    sp_prec = [r.get('spatial_precision', 0) for r in results]
    sp_rec = [r.get('spatial_recall', 0) for r in results]
    sp_f1 = [r.get('spatial_f1', 0) for r in results]
    ax.plot(runs, sp_prec, 'g-o', label='Precision', markersize=4, linewidth=1)
    ax.plot(runs, sp_rec, 'b-o', label='Recall', markersize=4, linewidth=1)
    ax.plot(runs, sp_f1, 'r-o', label='F1', markersize=4, linewidth=1)
    mean_sf1 = np.mean(sp_f1)
    ax.axhline(mean_sf1, color='red', linestyle=':', alpha=0.5)
    ax.text(0.98, mean_sf1, f'F1 mean={mean_sf1:.3f}', transform=ax.get_yaxis_transform(),
            ha='left', va='bottom', fontsize=8, color='red')
    ax.set_ylabel('Percentage')
    ax.set_title('Spatial (All Detections) Metrics')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)
    ax.set_xlim(0.5, N_RUNS + 0.5)

    # Subplot 3: TP/FP/FN trends
    ax = axes[2]
    tps = [r.get('person_tp', 0) for r in results]
    fps = [r.get('person_fp', 0) for r in results]
    fns = [r.get('person_fn', 0) for r in results]
    ax.plot(runs, tps, 'b-o', label='TP', markersize=4, linewidth=1)
    ax.plot(runs, fps, 'r-o', label='FP', markersize=4, linewidth=1)
    ax.plot(runs, fns, color='gray', marker='o', label='FN', markersize=4, linewidth=1)
    mean_tp = np.mean(tps)
    mean_fp = np.mean(fps)
    mean_fn = np.mean(fns)
    ax.axhline(mean_tp, color='blue', linestyle=':', alpha=0.4)
    ax.axhline(mean_fp, color='red', linestyle=':', alpha=0.4)
    ax.axhline(mean_fn, color='gray', linestyle=':', alpha=0.4)
    ax.text(0.98, mean_tp, f'TP mean={mean_tp:.0f}', transform=ax.get_yaxis_transform(),
            ha='left', va='bottom', fontsize=8, color='blue')
    ax.text(0.98, mean_fp, f'FP mean={mean_fp:.0f}', transform=ax.get_yaxis_transform(),
            ha='left', va='bottom', fontsize=8, color='red')
    ax.text(0.98, mean_fn, f'FN mean={mean_fn:.0f}', transform=ax.get_yaxis_transform(),
            ha='left', va='bottom', fontsize=8, color='gray')
    ax.set_xlabel('Run #')
    ax.set_ylabel('Count')
    ax.set_title('TP / FP / FN per Run')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, N_RUNS + 0.5)

    plt.tight_layout()
    plt.savefig(PLOT_ACCURACY, dpi=150)
    print(f"Accuracy plot saved to {PLOT_ACCURACY}")


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
    MAX_WORKERS = 4
    print(f"Running {N_RUNS}x eval_labeled (center-dist={CENTER_DIST}, frames={FRAMES})")
    print(f"Concurrency: {MAX_WORKERS} workers")
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

    # Final sort by run_id
    results.sort(key=lambda r: r.get('run_id', 0) if r else 0)
    valid_results = [r for r in results if r is not None]

    print_summary(valid_results)

    print("\nGenerating plots...")
    try:
        plot_results(valid_results)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Plotting failed: {e}")
        print("Results CSV is available for manual plotting.")

    print(f"\nDone. All outputs in {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()
