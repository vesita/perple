# -*- coding: utf-8 -*-
"""串行运行 20 次 eval_labeled，专门评估管线运行速度"""

import subprocess
import json
import csv
import time
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

OUTPUT_DIR = Path("output/batch_20_speed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_FILE = OUTPUT_DIR / "results.csv"
N_RUNS = 20
FRAMES = 408
CENTER_DIST = 0.5

BINARY = Path("E:/code/perple/target/release/examples/eval_labeled.exe")
if not BINARY.exists():
    subprocess.run(["cargo", "build", "--release", "--example", "eval_labeled"],
                   check=True, cwd="E:/code/perple")


def parse_elapsed_from_stdout(stdout: str) -> float | None:
    """从 stdout 回退解析耗时"""
    m = re.search(r'耗时:\s+([\d.]+)s', stdout)
    return float(m.group(1)) if m else None


def run_single(run_id: int) -> dict:
    """串行运行一次 eval_labeled，返回计时相关的指标"""
    run_dir = OUTPUT_DIR / f"run_{run_id:02d}"
    print(f"\n[{run_id:02d}/{N_RUNS}] 运行中 ...", end=" ", flush=True)

    t0 = time.time()
    result = subprocess.run(
        [BINARY,
         "--center-dist", str(CENTER_DIST), "--frames", str(FRAMES),
         "--output", str(run_dir)],
        capture_output=True, cwd="E:/code/perple",
    )
    wall_clock = time.time() - t0

    raw = result.stdout
    try:
        stdout = raw.decode('utf-8')
    except UnicodeDecodeError:
        stdout = raw.decode('utf-8', errors='replace')

    # 从 JSON 读取管线计时
    metrics: dict = {}
    json_path = run_dir / "eval_result.json"
    if json_path.exists():
        with open(json_path, 'r') as f:
            data = json.load(f)
        metrics['elapsed_s'] = data.get('elapsed_s', None)
        metrics['n_frames'] = data.get('n_frames', FRAMES)

    # 从 CSV 读取（备用）
    if not metrics.get('elapsed_s'):
        csv_path = run_dir / "eval_result.csv"
        if csv_path.exists():
            with open(csv_path, 'r') as f:
                for row in csv.reader(f):
                    if len(row) == 2 and row[0] == 'elapsed_s':
                        metrics['elapsed_s'] = float(row[1])

    # fallback
    if not metrics.get('elapsed_s'):
        metrics['elapsed_s'] = parse_elapsed_from_stdout(stdout) or 0.0

    pipeline_s = metrics.get('elapsed_s', 0.0)
    fps = FRAMES / pipeline_s if pipeline_s > 0 else 0.0

    print(f"管线={pipeline_s:.1f}s  wall={wall_clock:.1f}s  "
          f"{FRAMES / pipeline_s:.1f}帧/s  ({1000 * pipeline_s / FRAMES:.1f}ms/帧)",
          flush=True)

    return {
        'run_id': run_id,
        'elapsed_s': round(pipeline_s, 2),
        'wall_clock': round(wall_clock, 2),
        'fps': round(fps, 2),
        'ms_per_frame': round(1000.0 * pipeline_s / FRAMES, 2),
    }


def write_csv(results: list[dict]):
    fields = ['run_id', 'elapsed_s', 'wall_clock', 'fps', 'ms_per_frame']
    with open(RESULTS_FILE, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(results)
    print(f"\n结果已保存到 {RESULTS_FILE}")


def plot_results(results: list[dict]):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    # 论文格式字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.unicode_minus'] = False

    runs = [r['run_id'] for r in results]
    elapsed = [r['elapsed_s'] for r in results]
    fps = [r['fps'] for r in results]
    ms = [r['ms_per_frame'] for r in results]

    mean_t = np.mean(elapsed)
    std_t = np.std(elapsed)
    mean_fps = np.mean(fps)
    mean_ms = np.mean(ms)

    # ── 1. 管线耗时 ──
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(runs, elapsed, 'o-', color='#1f77b4', markersize=5, linewidth=1.2,
            label='管线耗时')
    ax.axhline(mean_t, color='#1f77b4', linestyle='--', linewidth=1, alpha=0.7,
               label=f'均值: {mean_t:.1f}s')
    ax.fill_between(runs, mean_t - std_t, mean_t + std_t,
                    color='#1f77b4', alpha=0.12,
                    label=f'±1σ ({std_t:.2f}s)')

    # 20Hz 参考线
    target_s = FRAMES / 20.0  # 50ms/frame 对应的总耗时
    ax.axhline(target_s, color='red', linestyle=':', linewidth=1.2, alpha=0.8,
               label=f'20Hz 目标 ({target_s:.1f}s)')

    ax.text(0.97, 0.95,
            f'均值: {mean_t:.1f}s\n标准差: {std_t:.2f}s\n目标(20Hz): {target_s:.1f}s',
            transform=ax.transAxes, fontsize=11,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='wheat', alpha=0.6))

    ax.set_xlabel('运行轮次', fontsize=14)
    ax.set_ylabel('耗时 (s)', fontsize=14)
    ax.set_title('管线处理耗时 (408 帧)', fontsize=16, fontweight='bold')
    ax.tick_params(labelsize=12)
    ax.set_xlim(0.5, N_RUNS + 0.5)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=11, loc='lower right')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig_pipeline_time.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  √ fig_pipeline_time.png")

    # ── 2. 帧率 ──
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(runs, fps, 's-', color='#2ca02c', markersize=5, linewidth=1.2,
            label='帧率 (FPS)')
    ax.axhline(mean_fps, color='#2ca02c', linestyle='--', linewidth=1, alpha=0.7,
               label=f'均值: {mean_fps:.1f} FPS')
    ax.axhline(20, color='red', linestyle=':', linewidth=1.2, alpha=0.8,
               label='20Hz 目标')

    ax.text(0.97, 0.95,
            f'均值: {mean_fps:.1f} FPS\n目标: 20.0 FPS',
            transform=ax.transAxes, fontsize=11,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='wheat', alpha=0.6))

    ax.set_xlabel('运行轮次', fontsize=14)
    ax.set_ylabel('帧率 (FPS)', fontsize=14)
    ax.set_title('管线处理帧率', fontsize=16, fontweight='bold')
    ax.tick_params(labelsize=12)
    ax.set_xlim(0.5, N_RUNS + 0.5)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=11, loc='lower right')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig_fps.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  √ fig_fps.png")

    # ── 3. 每帧耗时 (ms) ──
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(runs, ms, '^-', color='#9467bd', markersize=5, linewidth=1.2,
            label='每帧耗时')
    ax.axhline(mean_ms, color='#9467bd', linestyle='--', linewidth=1, alpha=0.7,
               label=f'均值: {mean_ms:.1f}ms')
    ax.axhline(50, color='red', linestyle=':', linewidth=1.2, alpha=0.8,
               label='20Hz 目标 (50ms)')

    mean_ms_val = np.mean(ms)
    std_ms_val = np.std(ms)
    ax.text(0.97, 0.95,
            f'均值: {mean_ms_val:.1f}ms\n标准差: {std_ms_val:.2f}ms\n目标(20Hz): 50ms',
            transform=ax.transAxes, fontsize=11,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='wheat', alpha=0.6))

    ax.set_xlabel('运行轮次', fontsize=14)
    ax.set_ylabel('每帧耗时 (ms)', fontsize=14)
    ax.set_title('每帧处理耗时', fontsize=16, fontweight='bold')
    ax.tick_params(labelsize=12)
    ax.set_xlim(0.5, N_RUNS + 0.5)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=11, loc='lower right')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig_ms_per_frame.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  √ fig_ms_per_frame.png")

    # ── 4. 汇总表 ──
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.axis('off')
    cell_text = [
        ['管线耗时 (s)', f'{mean_t:.1f}', f'{std_t:.2f}',
         f'{min(elapsed):.1f}', f'{max(elapsed):.1f}'],
        ['帧率 (FPS)', f'{mean_fps:.1f}', f'{np.std(fps):.2f}',
         f'{min(fps):.1f}', f'{max(fps):.1f}'],
        ['每帧耗时 (ms)', f'{mean_ms:.1f}', f'{std_ms_val:.2f}',
         f'{min(ms):.1f}', f'{max(ms):.1f}'],
    ]
    table = ax.table(cellText=cell_text,
                     colLabels=['指标', '均值', '标准差', '最小', '最大'],
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(13)
    table.scale(1, 2.0)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#40466e')
            cell.set_text_props(color='white', fontweight='bold')
        elif row % 2 == 0:
            cell.set_facecolor('#f0f0f0')

    ax.set_title(f'20 次运行速度汇总 (408 帧)', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig_summary.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  √ fig_summary.png")


def print_summary(results: list[dict]):
    import numpy as np
    elapsed = [r['elapsed_s'] for r in results]
    fps = [r['fps'] for r in results]
    ms = [r['ms_per_frame'] for r in results]

    print("\n" + "=" * 60)
    print("  20 次运行速度汇总")
    print("=" * 60)
    print(f"  {'指标':20s} {'均值':>8s} {'标准差':>8s} {'最小':>8s} {'最大':>8s}")
    print(f"  {'-'*48}")
    print(f"  {'管线耗时 (s)':20s} {np.mean(elapsed):>8.1f} {np.std(elapsed):>8.2f} "
          f"{min(elapsed):>8.1f} {max(elapsed):>8.1f}")
    print(f"  {'帧率 (FPS)':20s} {np.mean(fps):>8.1f} {np.std(fps):>8.2f} "
          f"{min(fps):>8.1f} {max(fps):>8.1f}")
    print(f"  {'每帧耗时 (ms)':20s} {np.mean(ms):>8.1f} {np.std(ms):>8.2f} "
          f"{min(ms):>8.1f} {max(ms):>8.1f}")
    print("=" * 60)

    target_ms = 1000.0 / 20.0
    if np.mean(ms) <= target_ms:
        print(f"\n  ✅ 均值 {np.mean(ms):.1f}ms/帧，满足 20Hz (≤{target_ms:.0f}ms) 要求")
    else:
        print(f"\n  ⚠️  均值 {np.mean(ms):.1f}ms/帧，未达到 20Hz ({target_ms:.0f}ms) 目标"
              f"，差距 {np.mean(ms) - target_ms:.1f}ms")


def main():
    print(f"串行运行 {N_RUNS} 次 eval_labeled（center-dist={CENTER_DIST}, frames={FRAMES}）")
    print(f"输出目录: {OUTPUT_DIR}")

    results = []
    for i in range(1, N_RUNS + 1):
        m = run_single(i)
        results.append(m)
        # 每轮保存 CSV，防止中断丢失
        write_csv(results)

    print_summary(results)

    print("\n生成图表中 ...")
    try:
        plot_results(results)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"绘图失败: {e}")

    print(f"\n完成。所有输出在 {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()
