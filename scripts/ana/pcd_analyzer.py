#.venv/bin/python3
# -*- coding: utf-8 -*-

"""
点云分析工具 — 输出论文级图表

提供 PCD 文件读取、高度直方图分析和点投影直方图分析功能，
所有图片已优化为适合插入 Word 的尺寸和字体。

输出：output/analyze/ 目录下的 PNG 图片
"""

import sys
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import fftpack

# ─── 论文级图表配置 ───────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["SimHei", "Microsoft YaHei", "SimSun"],
    "axes.unicode_minus": False,
})

FONT_TITLE = 14       # 图表标题（四号）
FONT_LABEL = 10.5     # 轴标签（五号）
FONT_TICK = 9         # 刻度数字（小五）
FONT_LEGEND = 9       # 图例（小五）
FIG_W = 5.9           # A4 文本区宽度（15cm ≈ 5.9in）
FIG_H = 4.0           # 默认高度

OUT_DIR = "output/analyze"

# ─── open3d 导入 ───────────────────────────────────────────────────────────
try:
    import open3d as o3d
    O3D_AVAILABLE = True
except ImportError:
    o3d = None
    O3D_AVAILABLE = False
    print("警告: 未安装 open3d 库，部分功能可能受限（pip install open3d）")


def read_pcd_file(file_path):
    """
    使用 open3d 读取 PCD 点云文件
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件 {file_path} 不存在")

    if not O3D_AVAILABLE:
        raise RuntimeError("未安装 open3d 库")

    pcd = o3d.io.read_point_cloud(file_path)
    points = np.asarray(pcd.points)
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
        points = np.hstack((points, colors))
    return points


def compute_h_histogram(points, bins=100, axis=2):
    """计算点云在指定轴上的高度直方图"""
    heights = points[:, axis]
    hist, bin_edges = np.histogram(heights, bins=bins)
    return hist, bin_edges


def compute_point_projection_histogram(points, reference_vector=None, resolution=0.1):
    """
    计算点云在指定参考向量上的投影直方图
    """
    if reference_vector is None:
        reference_vector = np.array([0, 0, 1])

    reference_vector = reference_vector / np.linalg.norm(reference_vector)
    projections = np.dot(points[:, :3], reference_vector)
    sorted_projections = np.sort(projections)

    min_proj = np.min(sorted_projections)
    max_proj = np.max(sorted_projections)
    bins = int(np.ceil((max_proj - min_proj) / resolution))
    bins = max(bins, 1)

    hist, bin_edges = np.histogram(sorted_projections, bins=bins)
    return hist, bin_edges


def plot_histogram(hist, bin_edges, title="直方图", xlabel="值", ylabel="频数",
                   color='skyblue', edgecolor='white'):
    """
    绘制论文级直方图
    """
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H * 0.8))
    ax.bar(bin_edges[:-1], hist, width=np.diff(bin_edges),
           color=color, edgecolor=edgecolor, align="edge", alpha=0.8)
    ax.set_title(title, fontsize=FONT_TITLE, fontweight='bold', pad=8)
    ax.set_xlabel(xlabel, fontsize=FONT_LABEL)
    ax.set_ylabel(ylabel, fontsize=FONT_LABEL)
    ax.tick_params(axis="both", labelsize=FONT_TICK)
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig


def compute_3d_fft_spectrum(points, grid_size=64):
    """对点云进行 3D FFT 频谱分析"""
    min_bound = np.min(points[:, :3], axis=0)
    max_bound = np.max(points[:, :3], axis=0)

    grid = np.zeros((grid_size, grid_size, grid_size))
    normalized_points = (points[:, :3] - min_bound) / (max_bound - min_bound + 1e-10)
    grid_coords = (normalized_points * (grid_size - 1)).astype(int)
    grid_coords = np.clip(grid_coords, 0, grid_size - 1)

    for coord in grid_coords:
        grid[coord[0], coord[1], coord[2]] += 1

    fft_result = fftpack.fftn(grid)
    fft_shifted = fftpack.fftshift(fft_result)
    magnitude_spectrum = np.abs(fft_shifted)

    freq_cube = np.fft.fftshift(np.fft.fftfreq(grid_size))
    freq_x, freq_y, freq_z = np.meshgrid(freq_cube, freq_cube, freq_cube, indexing='ij')
    freq_radius = np.sqrt(freq_x**2 + freq_y**2 + freq_z**2)

    max_radius = np.max(freq_radius) + 1e-10
    radial_bins = 100
    bin_indices = (freq_radius / max_radius * radial_bins).astype(int)
    bin_indices = np.clip(bin_indices, 0, radial_bins - 1)

    radial_profile = np.zeros(radial_bins)
    for i in range(radial_bins):
        mask = bin_indices == i
        if np.any(mask):
            radial_profile[i] = np.mean(magnitude_spectrum[mask])

    return magnitude_spectrum, freq_radius, radial_profile


def plot_3d_fft_analysis(points, output_dir=None):
    """绘制论文级 3D FFT 频谱分析结果"""
    magnitude_spectrum, freq_radius, radial_profile = compute_3d_fft_spectrum(points)
    center = magnitude_spectrum.shape[0] // 2

    # ─── 频谱中心切片 ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_W * 0.75))
    im = ax.imshow(np.log(1 + magnitude_spectrum[center, :, :]), cmap='hot')
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("对数幅度", fontsize=FONT_TICK)
    cbar.ax.tick_params(labelsize=FONT_TICK)
    ax.set_title("3D FFT 频谱（中心切片）", fontsize=FONT_TITLE, fontweight='bold', pad=8)
    ax.set_xlabel("频率索引", fontsize=FONT_LABEL)
    ax.set_ylabel("频率索引", fontsize=FONT_LABEL)
    ax.tick_params(axis="both", labelsize=FONT_TICK)
    plt.tight_layout()

    if output_dir:
        path = os.path.join(output_dir, "3d_fft_spectrum_slice.png")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  → {path}")
    else:
        plt.show()

    # ─── 径向平均轮廓 ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H * 0.75))
    ax.plot(radial_profile, linewidth=1.2, color="#e74c3c")
    ax.set_title("径向平均幅度轮廓", fontsize=FONT_TITLE, fontweight='bold', pad=8)
    ax.set_xlabel("径向 Bin", fontsize=FONT_LABEL)
    ax.set_ylabel("平均幅度", fontsize=FONT_LABEL)
    ax.tick_params(axis="both", labelsize=FONT_TICK)
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    if output_dir:
        path = os.path.join(output_dir, "radial_profile.png")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  → {path}")
    else:
        plt.show()


def analyze_and_plot_all_histograms(points, output_dir=None):
    """全面直方图分析并输出论文级图表"""
    os.makedirs(output_dir, exist_ok=True) if output_dir else None

    # 1. Z 轴高度直方图
    print("  [1/5] Z 轴高度直方图...")
    h_hist, h_bin_edges = compute_h_histogram(points, bins=100, axis=2)
    fig1 = plot_histogram(h_hist, h_bin_edges,
                          title="Z 轴高度分布直方图",
                          xlabel="高度 (Z)", ylabel="频数",
                          color='lightcoral')
    if output_dir:
        fig1.savefig(os.path.join(output_dir, "z_height_histogram.png"),
                     dpi=300, bbox_inches='tight')
        plt.close(fig1)

    # 2. 点投影直方图（Z 轴）
    print("  [2/5] 点投影直方图...")
    ppf_hist, ppf_bin_edges = compute_point_projection_histogram(points, resolution=0.1)
    fig2 = plot_histogram(ppf_hist, ppf_bin_edges,
                          title="Z 轴方向点投影直方图",
                          xlabel="Z 轴投影值", ylabel="频数",
                          color='lightgreen')
    if output_dir:
        fig2.savefig(os.path.join(output_dir, "projection_histogram.png"),
                     dpi=300, bbox_inches='tight')
        plt.close(fig2)

    # 3. X 轴分布直方图
    print("  [3/5] X 轴分布直方图...")
    x_hist, x_bin_edges = compute_h_histogram(points, bins=100, axis=0)
    fig3 = plot_histogram(x_hist, x_bin_edges,
                          title="X 轴分布直方图",
                          xlabel="X 坐标", ylabel="频数",
                          color='gold')
    if output_dir:
        fig3.savefig(os.path.join(output_dir, "x_histogram.png"),
                     dpi=300, bbox_inches='tight')
        plt.close(fig3)

    # 4. Y 轴分布直方图
    print("  [4/5] Y 轴分布直方图...")
    y_hist, y_bin_edges = compute_h_histogram(points, bins=100, axis=1)
    fig4 = plot_histogram(y_hist, y_bin_edges,
                          title="Y 轴分布直方图",
                          xlabel="Y 坐标", ylabel="频数",
                          color='plum')
    if output_dir:
        fig4.savefig(os.path.join(output_dir, "y_histogram.png"),
                     dpi=300, bbox_inches='tight')
        plt.close(fig4)

    # 5. 3D FFT 频谱分析
    print("  [5/5] 3D FFT 频谱分析...")
    plot_3d_fft_analysis(points, output_dir)


def print_statistical_summary(points):
    """打印点云统计摘要"""
    print(f"点云总数: {len(points)} 个点")
    print(f"数据维度: {points.shape[1] if points.ndim > 1 else 3}")

    axes = ['X', 'Y', 'Z']
    for i, axis in enumerate(axes):
        values = points[:, i]
        print(f"\n  {axis} 轴:")
        print(f"    范围: [{np.min(values):.3f}, {np.max(values):.3f}]")
        print(f"    均值: {np.mean(values):.3f}  中位数: {np.median(values):.3f}")


def main():
    default_pcd_path = "data/cloud/lidar/000101.pcd"

    if not os.path.exists(default_pcd_path):
        print(f"错误: PCD 文件 '{default_pcd_path}' 不存在")
        return

    try:
        print("正在读取点云文件...")
        points = read_pcd_file(default_pcd_path)
        print_statistical_summary(points)

        # 清理旧数据
        if os.path.exists(OUT_DIR):
            import shutil as _shutil
            _shutil.rmtree(OUT_DIR)
        os.makedirs(OUT_DIR, exist_ok=True)
        print(f"\n正在生成分析图表（输出至 {OUT_DIR}/）...\n")
        analyze_and_plot_all_histograms(points, OUT_DIR)

        print(f"\n完成！共 7 张图，300dpi，适合插入 Word。")
        print(f"图片说明（图注）：")
        print(f"  图1 Z 轴高度分布直方图 — 地面点云的高度频率分布")
        print(f"  图2 Z 轴方向点投影直方图 — 点云在 Z 轴的投影分布")
        print(f"  图3 X 轴分布直方图 — 点云在 X 方向的空间分布")
        print(f"  图4 Y 轴分布直方图 — 点云在 Y 方向的空间分布")
        print(f"  图5 3D FFT 频谱（中心切片）— 点云结构的频域特征")
        print(f"  图6 径向平均幅度轮廓 — 频谱的径向分布特征")

    except Exception as e:
        print(f"处理点云文件时出错: {e}")


if __name__ == "__main__":
    main()
