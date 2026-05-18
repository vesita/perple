"""
BevEDLines & 剪叶聚类 — 性能对比基准测试（视频素材用）

比较三种管线在点云聚类上的运行速度：
  1. raw_dbscan  — 直接 DBSCAN（无预处理，作为不可行的 baseline）
  2. lvdot       — 体素占用下采样 + DBSCAN
  3. prune_qt    — 四叉树剪叶 + 四叉树加速 DBSCAN（本文方法）

输出：
  - 逐帧耗时折线图（叠加 Rust 实测参考线）
  - 统计表格（均值/中位数/标准差/加速比）
  - 柱状对比图
  - 降维效果对比图
"""

import math
import time
import json
import numpy as np
from sklearn.cluster import DBSCAN
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── 配置 ──
DATA_DIR = Path("data/cloud/lidar")
EPS = 0.30
MIN_PTS = 3
VOXEL_SIZE = 0.15
MIN_OCC = 3
QT_MAX_PTS = 20
QT_MAX_DEPTH = 10
QT_MIN_OCC = 3
RNG_SEED = 42
np.random.seed(RNG_SEED)

# Rust 实测参考数据（从 config/bench/cluster/prune_qt.toml + lvdot.toml 取 median）
RUST_REF = {
    'prune_qt': {'median_ms': 16.77, 'label': '剪叶聚类 (Rust)'},
    'lvdot':    {'median_ms': 16.75, 'label': 'LV-DOT (Rust)'},
}


def load_pcd(path):
    with open(path) as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if line.startswith("DATA"):
            data_start = i + 1
            break
    pts = np.loadtxt(lines[data_start:], dtype=np.float32)
    mask = ~((pts[:, 0] == 0) & (pts[:, 1] == 0) & (pts[:, 2] == 0))
    return pts[mask]


def ground_removal(pts, z_max=0.95):
    return pts[pts[:, 2] > z_max]


# ── 四叉树 ──
class QuadNode:
    __slots__ = ('x_min', 'x_max', 'y_min', 'y_max', 'points',
                 'children', 'is_leaf', '_max_pts', '_max_depth')

    def __init__(self, x_min, x_max, y_min, y_max, max_pts=20, max_depth=10):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.points = []
        self.children = None
        self.is_leaf = True
        self._max_pts = max_pts
        self._max_depth = max_depth

    def insert(self, idx, pts, depth=0):
        x, y = pts[idx, 0], pts[idx, 1]
        if x < self.x_min or x >= self.x_max or y < self.y_min or y >= self.y_max:
            return
        self._insert_rec(idx, pts, depth)

    def _insert_rec(self, idx, pts, depth):
        if self.is_leaf:
            if len(self.points) < self._max_pts or depth >= self._max_depth:
                self.points.append(idx)
                return
            self._subdivide()
            old = self.points
            self.points = []
            old.append(idx)
            for pidx in old:
                self._insert_child(pidx, pts, depth + 1)
        else:
            self._insert_child(idx, pts, depth + 1)

    def _subdivide(self):
        xm = (self.x_min + self.x_max) / 2
        ym = (self.y_min + self.y_max) / 2
        self.children = [
            QuadNode(self.x_min, xm, self.y_min, ym, self._max_pts, self._max_depth),
            QuadNode(xm, self.x_max, self.y_min, ym, self._max_pts, self._max_depth),
            QuadNode(self.x_min, xm, ym, self.y_max, self._max_pts, self._max_depth),
            QuadNode(xm, self.x_max, ym, self.y_max, self._max_pts, self._max_depth),
        ]
        self.is_leaf = False

    def _insert_child(self, idx, pts, depth):
        x, y = pts[idx, 0], pts[idx, 1]
        xm = (self.x_min + self.x_max) / 2
        ym = (self.y_min + self.y_max) / 2
        ci = 0 if x < xm else 1
        ci += 0 if y < ym else 2
        self.children[ci]._insert_rec(idx, pts, depth)

    def collect_leaves(self):
        out = []
        self._collect(out)
        return out

    def _collect(self, out):
        if self.is_leaf:
            out.append(self)
        elif self.children:
            for c in self.children:
                c._collect(out)

    def query_range_pts(self, cx, cy, radius, pts):
        result = []
        self._query_range_pts(cx, cy, radius, pts, result)
        return result

    def _query_range_pts(self, cx, cy, radius, pts, result):
        hw = (self.x_max - self.x_min) / 2
        hh = (self.y_max - self.y_min) / 2
        dx = abs(cx - (self.x_min + self.x_max) / 2)
        dy = abs(cy - (self.y_min + self.y_max) / 2)
        if dx > hw + radius or dy > hh + radius:
            return
        if self.is_leaf:
            r2 = radius * radius
            for idx in self.points:
                p = pts[idx]
                if (p[0] - cx) ** 2 + (p[1] - cy) ** 2 <= r2:
                    result.append(idx)
        elif self.children:
            for c in self.children:
                c._query_range_pts(cx, cy, radius, pts, result)


def build_quadtree(pts, max_pts=20, max_depth=10):
    x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
    y_min, y_max = pts[:, 1].min(), pts[:, 1].max()
    pad = 0.1
    root = QuadNode(x_min - pad, x_max + pad, y_min - pad, y_max + pad, max_pts, max_depth)
    for i in range(len(pts)):
        root.insert(i, pts)
    return root


def quadtree_dbscan(sampled, eps, min_pts):
    if len(sampled) == 0:
        return []
    qt = build_quadtree(sampled, max_pts=20, max_depth=10)
    n = len(sampled)
    visited = [False] * n
    labels = [-1] * n
    clusters = []
    cluster_id = 0
    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        neighbors = qt.query_range_pts(sampled[i, 0], sampled[i, 1], eps, sampled)
        if len(neighbors) < min_pts:
            continue
        labels[i] = cluster_id
        cluster = [i]
        queue = list(neighbors)
        k = 0
        while k < len(queue):
            ni = queue[k]
            if not visited[ni]:
                visited[ni] = True
                more = qt.query_range_pts(sampled[ni, 0], sampled[ni, 1], eps, sampled)
                if len(more) >= min_pts:
                    queue.extend(more)
            if labels[ni] == -1:
                labels[ni] = cluster_id
                cluster.append(ni)
            k += 1
        clusters.append(cluster)
        cluster_id += 1
    return clusters


# ── 三种方法 ──

def method_raw_dbscan(pts, eps=EPS, min_pts=MIN_PTS):
    """直接 DBSCAN — 全量点云。"""
    t0 = time.perf_counter()
    if len(pts) == 0:
        return 0, 0, 0
    clustering = DBSCAN(eps=eps, min_samples=min_pts, metric='euclidean', n_jobs=1)
    labels = clustering.fit_predict(pts[:, :2])
    n_clusters = len(set(labels[labels >= 0]))
    elapsed = (time.perf_counter() - t0) * 1000
    return elapsed, n_clusters, len(pts)


def method_lvdot(pts, voxel_size=VOXEL_SIZE, min_occ=MIN_OCC, eps=EPS, min_pts=MIN_PTS):
    """体素下采样 + DBSCAN。"""
    t0 = time.perf_counter()
    if len(pts) == 0:
        return 0, 0, 0, 0
    inv = 1.0 / voxel_size
    voxel_map = {}
    for i, p in enumerate(pts):
        key = (math.floor(p[0] * inv), math.floor(p[1] * inv), math.floor(p[2] * inv))
        if key not in voxel_map:
            voxel_map[key] = []
        voxel_map[key].append(i)
    down_pts = []
    for indices in voxel_map.values():
        if len(indices) >= min_occ:
            centroid = pts[indices].mean(axis=0)
            down_pts.append(centroid[:2])
    down_pts = np.array(down_pts, dtype=np.float32)
    if len(down_pts) == 0:
        return (time.perf_counter() - t0) * 1000, 0, 0, len(pts)
    clustering = DBSCAN(eps=eps, min_samples=min_pts, metric='euclidean', n_jobs=1)
    labels = clustering.fit_predict(down_pts)
    n_clusters = len(set(labels[labels >= 0]))
    elapsed = (time.perf_counter() - t0) * 1000
    return elapsed, n_clusters, len(down_pts), len(pts)


def method_prune_qt(pts, min_occ=QT_MIN_OCC, eps=EPS, min_pts=MIN_PTS,
                    max_pts=QT_MAX_PTS, max_depth=QT_MAX_DEPTH):
    """四叉树剪叶 + 四叉树加速 DBSCAN。"""
    t0 = time.perf_counter()
    if len(pts) == 0:
        return 0, 0, 0, 0
    qt = build_quadtree(pts, max_pts, max_depth)
    leaves = qt.collect_leaves()
    sampled_pts = []
    for leaf in leaves:
        if len(leaf.points) >= min_occ:
            indices = leaf.points
            cx = pts[indices, 0].mean()
            cy = pts[indices, 1].mean()
            sampled_pts.append([cx, cy])
    sampled = np.array(sampled_pts, dtype=np.float32)
    if len(sampled) == 0:
        return (time.perf_counter() - t0) * 1000, 0, 0, len(pts)
    clusters = quadtree_dbscan(sampled, eps, min_pts)
    n_clusters = len(clusters)
    elapsed = (time.perf_counter() - t0) * 1000
    return elapsed, n_clusters, len(sampled), len(pts)


# ── 主流程 ──
def run_benchmark():
    import sys
    sys.stdout.reconfigure(encoding='utf-8')

    print("=" * 60)
    print("BevEDLines & 剪叶聚类 — 性能对比基准测试")
    print(f"EPS={EPS}, MIN_PTS={MIN_PTS}, 数据目录={DATA_DIR}")
    print("=" * 60)

    # 获取所有有效文件
    all_pcds = sorted(DATA_DIR.glob("*.pcd"))
    print(f"共发现 {len(all_pcds)} 个 PCD 文件")

    results = {
        'raw_dbscan': {'time_ms': [], 'n_clusters': [], 'n_points': [], 'frame_ids': []},
        'lvdot': {'time_ms': [], 'n_clusters': [], 'n_down': [], 'n_points': [], 'frame_ids': []},
        'prune_qt': {'time_ms': [], 'n_clusters': [], 'n_down': [], 'n_points': [], 'frame_ids': []},
    }

    # 每 10 帧测试一个，覆盖全部数据
    test_files = all_pcds[::3]  # 每3帧取1帧，约210帧

    for idx, path in enumerate(test_files):
        pts = load_pcd(path)
        pts_ng = ground_removal(pts)

        if len(pts_ng) < MIN_PTS:
            continue

        fnum = int(path.stem)

        # 方法1: 直接 DBSCAN
        t1, nc1, np1 = method_raw_dbscan(pts_ng)
        results['raw_dbscan']['time_ms'].append(t1)
        results['raw_dbscan']['n_clusters'].append(nc1)
        results['raw_dbscan']['n_points'].append(np1)
        results['raw_dbscan']['frame_ids'].append(fnum)

        # 方法2: LV-DOT
        t2, nc2, nd2, np2 = method_lvdot(pts_ng)
        results['lvdot']['time_ms'].append(t2)
        results['lvdot']['n_clusters'].append(nc2)
        results['lvdot']['n_down'].append(nd2)
        results['lvdot']['n_points'].append(np2)
        results['lvdot']['frame_ids'].append(fnum)

        # 方法3: 剪叶聚类
        t3, nc3, nd3, np3 = method_prune_qt(pts_ng)
        results['prune_qt']['time_ms'].append(t3)
        results['prune_qt']['n_clusters'].append(nc3)
        results['prune_qt']['n_down'].append(nd3)
        results['prune_qt']['n_points'].append(np3)
        results['prune_qt']['frame_ids'].append(fnum)

        if (idx + 1) % 20 == 0 or idx == 0:
            speedup = t1 / max(t3, 0.01)
            print(f"  [{idx+1}/{len(test_files)}] #{fnum}: "
                  f"raw={t1:.0f}ms lvdot={t2:.0f}ms({nd2}pt) qt={t3:.0f}ms({nd3}pt) "
                  f"speedup={speedup:.1f}x vs raw")

    # ── 统计 ──
    print("\n" + "=" * 60)
    print("统计结果")
    print("=" * 60)
    header = f"{'方法':<22} {'均值(ms)':<10} {'中位数(ms)':<12} {'标准差(ms)':<12} {'帧数':<8}"
    print(header)
    print("-" * 64)

    stats = {}
    for name in ['raw_dbscan', 'lvdot', 'prune_qt']:
        times = np.array(results[name]['time_ms'])
        if len(times) == 0:
            continue
        mean_t = times.mean()
        med_t = np.median(times)
        std_t = times.std()
        labels = {
            'raw_dbscan': 'Raw DBSCAN (全量)',
            'lvdot': 'LV-DOT (体素+DBSCAN)',
            'prune_qt': '剪叶聚类 (本文)'
        }
        print(f"{labels[name]:<22} {mean_t:<10.2f} {med_t:<12.2f} {std_t:<12.2f} {len(times):<8}")
        stats[name] = {'mean': float(mean_t), 'median': float(med_t), 'std': float(std_t)}

    if 'raw_dbscan' in stats and 'prune_qt' in stats:
        r = stats['raw_dbscan']['median'] / stats['prune_qt']['median']
        print(f"\n剪叶聚类 vs 直接 DBSCAN (中位数加速比): {r:.1f}x")
    if 'lvdot' in stats and 'prune_qt' in stats:
        r = stats['lvdot']['median'] / stats['prune_qt']['median']
        print(f"剪叶聚类 vs LV-DOT (中位数加速比): {r:.1f}x")

    print(f"\nRust 参考: 剪叶聚类中位数={RUST_REF['prune_qt']['median_ms']}ms, "
          f"LV-DOT中位数={RUST_REF['lvdot']['median_ms']}ms")

    # ── 图表 ──
    out_dir = Path("output/bench_video")
    out_dir.mkdir(parents=True, exist_ok=True)

    colors = {'raw_dbscan': '#E74C3C', 'lvdot': '#F39C12', 'prune_qt': '#27AE60'}
    labels_map = {
        'raw_dbscan': 'Raw DBSCAN',
        'lvdot': 'Voxel+DBSCAN',
        'prune_qt': 'Prune-QuadTree (Ours)',
    }

    # 图1: 逐帧耗时
    fig, axes = plt.subplots(3, 1, figsize=(12, 14), dpi=150)
    ax = axes[0]
    x = np.arange(len(results['prune_qt']['time_ms']))
    for name in ['raw_dbscan', 'lvdot', 'prune_qt']:
        ax.plot(x, results[name]['time_ms'], color=colors[name],
                label=labels_map[name], linewidth=1.5, alpha=0.85)
    ax.axhline(y=RUST_REF['prune_qt']['median_ms'], color='#27AE60',
               linestyle='--', linewidth=1, alpha=0.6,
               label=f"PruneQt Rust ref ({RUST_REF['prune_qt']['median_ms']}ms)")
    ax.axhline(y=RUST_REF['lvdot']['median_ms'], color='#F39C12',
               linestyle='--', linewidth=1, alpha=0.6,
               label=f"LV-DOT Rust ref ({RUST_REF['lvdot']['median_ms']}ms)")
    ax.set_xlabel('Frame', fontsize=12)
    ax.set_ylabel('Time (ms)', fontsize=12)
    ax.set_title('Per-frame Runtime Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # 图2: 降采样点数
    ax = axes[1]
    ax.plot(x, results['lvdot']['n_down'], color=colors['lvdot'],
            label='Voxel downsample (LV-DOT)', linewidth=1.5, alpha=0.8)
    ax.plot(x, results['prune_qt']['n_down'], color=colors['prune_qt'],
            label='Quadtree centroids (Ours)', linewidth=1.5, alpha=0.8)
    ax.set_xlabel('Frame', fontsize=12)
    ax.set_ylabel('Downsampled Points', fontsize=12)
    ax.set_title('Downsampled Point Count per Frame', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # 图3: 原始点数
    ax = axes[2]
    ax.plot(x, results['raw_dbscan']['n_points'], color='#7F8C8D',
            label='Input points (after ground removal)', linewidth=1.5, alpha=0.8)
    ax.set_xlabel('Frame', fontsize=12)
    ax.set_ylabel('Point Count', fontsize=12)
    ax.set_title('Non-ground Points per Frame', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_dir / "benchmark_comparison.png", dpi=150, bbox_inches='tight')
    print(f"\n[OK] {out_dir / 'benchmark_comparison.png'}")
    plt.close(fig)

    # 图2: 柱状对比
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6), dpi=150)
    names_list = ['Raw DBSCAN', 'Voxel+DBSCAN\n(LV-DOT)', 'Prune-QuadTree\n(Ours-Python)']
    means_list = [stats['raw_dbscan']['mean'], stats['lvdot']['mean'], stats['prune_qt']['mean']]
    stds_list = [stats['raw_dbscan']['std'], stats['lvdot']['std'], stats['prune_qt']['std']]
    bar_colors = [colors['raw_dbscan'], colors['lvdot'], colors['prune_qt']]

    bars = ax2.bar(names_list, means_list, yerr=stds_list, color=bar_colors,
                   edgecolor='white', linewidth=1.5, capsize=5, width=0.5)
    for bar, val in zip(bars, means_list):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(stds_list) * 0.1,
                 f'{val:.1f} ms', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # 叠加 Rust 参考线
    ax2.axhline(y=RUST_REF['prune_qt']['median_ms'], color='#27AE60',
                linestyle='--', linewidth=1.5, alpha=0.7,
                label=f"PruneQt Rust: {RUST_REF['prune_qt']['median_ms']}ms")
    ax2.legend(fontsize=11)
    ax2.set_ylabel('Average Time (ms)', fontsize=12)
    ax2.set_title('Average Runtime per Frame (Python impl)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    fig2.savefig(out_dir / "benchmark_bar.png", dpi=150, bbox_inches='tight')
    print(f"[OK] {out_dir / 'benchmark_bar.png'}")
    plt.close(fig2)

    # 图3: 加速比
    ratios = []
    ratio_labels = []
    if stats['raw_dbscan']['median'] > 0:
        ratios.append(stats['raw_dbscan']['median'] / stats['prune_qt']['median'])
        ratio_labels.append('vs Raw DBSCAN')
    if stats['lvdot']['median'] > 0:
        ratios.append(stats['lvdot']['median'] / stats['prune_qt']['median'])
        ratio_labels.append('vs LV-DOT (Python)')

    fig3, ax3 = plt.subplots(1, 1, figsize=(8, 4), dpi=150)
    bars3 = ax3.barh(ratio_labels, ratios, color=['#3498DB', '#2ECC71'],
                     edgecolor='white', height=0.4)
    for bar, val in zip(bars3, ratios):
        ax3.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                 f'{val:.1f}x speedup', ha='left', va='center', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Speedup Ratio', fontsize=12)
    ax3.set_title('Prune-QuadTree Speedup over Baselines', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')
    ax3.set_xlim(0, max(ratios) * 1.5)
    fig3.savefig(out_dir / "speedup_ratio.png", dpi=150, bbox_inches='tight')
    print(f"[OK] {out_dir / 'speedup_ratio.png'}")
    plt.close(fig3)

    # 保存统计
    stat_data = {}
    for name in ['raw_dbscan', 'lvdot', 'prune_qt']:
        stat_data[name] = {
            'mean_ms': stats[name]['mean'],
            'median_ms': stats[name]['median'],
            'std_ms': stats[name]['std'],
            'frames': len(results[name]['time_ms']),
        }
    stat_data['rust_reference'] = RUST_REF
    with open(out_dir / "benchmark_stats.json", 'w', encoding='utf-8') as f:
        json.dump(stat_data, f, ensure_ascii=False, indent=2)
    print(f"[OK] {out_dir / 'benchmark_stats.json'}")

    # 打印 Rust 对比摘要
    print("\n" + "=" * 60)
    print("Rust 实现性能参考 (来自现有 bench)")
    print("=" * 60)
    print(f"  PruneQt (Rust): median={RUST_REF['prune_qt']['median_ms']}ms")
    print(f"  LV-DOT (Rust):  median={RUST_REF['lvdot']['median_ms']}ms")
    print(f"  说明: Rust 实现通过编译优化、零成本抽象和无 GC 暂停,")
    print(f"        PruneQt 可达 ~17ms，满足 20Hz 实时需求 (~50ms/帧)")

    print("\nDONE")


if __name__ == '__main__':
    run_benchmark()
