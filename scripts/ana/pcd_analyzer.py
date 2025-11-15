#.venv/bin/python3
# -*- coding: utf-8 -*-

"""
点云分析工具
提供PCD文件读取、h直方图分析和点投影直方图分析功能
"""

import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack


# 导入open3d库用于点云处理
try:
    import open3d as o3d
    O3D_AVAILABLE = True
except ImportError:
    o3d = None
    O3D_AVAILABLE = False
    print("警告: 未安装open3d库，部分功能可能受限")
    print("可通过 'pip install open3d' 安装")


def read_pcd_file(file_path):
    """
    使用open3d读取PCD点云文件
    
    Args:
        file_path (str): PCD文件路径
        
    Returns:
        numpy.ndarray: 点云数据 (N, 3) 数组
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件 {file_path} 不存在")
    
    # 使用open3d读取
    if O3D_AVAILABLE:
        try:
            pcd = o3d.io.read_point_cloud(file_path)
            points = np.asarray(pcd.points)
            # 如果存在颜色信息也一并提取
            if pcd.has_colors():
                colors = np.asarray(pcd.colors)
                points = np.hstack((points, colors))
            return points
        except Exception as e:
            print(f"使用open3d读取失败: {e}")
            raise RuntimeError("无法读取PCD文件")
    else:
        raise RuntimeError("未安装open3d库，请通过 'pip install open3d' 安装")


def compute_h_histogram(points, bins=100, axis=2):
    """
    计算点云在指定轴上的高度直方图
    
    Args:
        points (numpy.ndarray): 点云数据 (N, 3) 或 (N, 4) 数组
        bins (int): 直方图bins数量
        axis (int): 轴索引 (0=x, 1=y, 2=z)
        
    Returns:
        tuple: (hist, bin_edges) 直方图数据和边界
    """
    heights = points[:, axis]
    hist, bin_edges = np.histogram(heights, bins=bins)
    return hist, bin_edges


def compute_point_projection_histogram(points, reference_vector=None, resolution=0.1):
    """
    计算点云在指定参考向量上的投影，并生成投影值的分布直方图。
    
    该函数首先计算所有点在参考向量上的投影标量值，然后根据指定的分辨率
    自动确定直方图的分组(bin)数量，最后计算并返回投影值的频率分布。
    
    Args:
        points (numpy.ndarray): 点云数据 (N, 3) 数组
        reference_vector (numpy.ndarray): 参考向量，决定投影的方向。默认为Z轴方向 [0, 0, 1]。
        resolution (float): 直方图的分辨率，表示每个bin的宽度（即覆盖的投影值范围）。
        
    Returns:
        tuple: 包含两个元素的元组：
            - hist (numpy.ndarray): 直方图频数数组
            - bin_edges (numpy.ndarray): 直方图边界数组
    """
    if reference_vector is None:
        # 默认使用Z轴作为参考向量
        reference_vector = np.array([0, 0, 1])
    
    # 归一化参考向量
    reference_vector = reference_vector / np.linalg.norm(reference_vector)
    
    # 计算点到参考向量的投影
    projections = np.dot(points[:, :3], reference_vector)
    
    # 按投影值排序
    sorted_projections = np.sort(projections)
    
    # 根据分辨率计算bins数量
    min_proj = np.min(sorted_projections)
    max_proj = np.max(sorted_projections)
    bins = int(np.ceil((max_proj - min_proj) / resolution))
    bins = max(bins, 1)  # 确保至少有1个bin
    
    # 计算直方图
    hist, bin_edges = np.histogram(sorted_projections, bins=bins)
    return hist, bin_edges


def plot_histogram(hist, bin_edges, title="Histogram", xlabel="Value", ylabel="Frequency", 
                   color='skyblue', edgecolor='black'):
    """
    使用matplotlib绘制直方图
    
    Args:
        hist (numpy.ndarray): 直方图数据
        bin_edges (numpy.ndarray): 边界数据
        title (str): 图表标题
        xlabel (str): X轴标签
        ylabel (str): Y轴标签
        color (str): 直方图填充颜色
        edgecolor (str): 直方图边缘颜色
    """
    plt.figure(figsize=(10, 6))
    plt.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), 
            color=color, edgecolor=edgecolor, align="edge", alpha=0.7)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt.gcf()


def compute_3d_fft_spectrum(points, grid_size=64):
    """
    对点云进行3D FFT频谱分析
    
    Args:
        points (numpy.ndarray): 点云数据 (N, 3) 数组
        grid_size (int): 用于FFT的网格大小
        
    Returns:
        tuple: (magnitude_spectrum, freq_radius, radial_profile)
               包含幅度谱、频率半径和径向轮廓
    """
    # 获取点云边界
    min_bound = np.min(points[:, :3], axis=0)
    max_bound = np.max(points[:, :3], axis=0)
    
    # 创建3D网格
    grid = np.zeros((grid_size, grid_size, grid_size))
    
    # 将点云映射到网格上
    # 归一化点坐标到[0, 1]范围
    normalized_points = (points[:, :3] - min_bound) / (max_bound - min_bound)
    
    # 映射到网格坐标
    grid_coords = (normalized_points * (grid_size - 1)).astype(int)
    
    # 确保坐标不会越界
    grid_coords = np.clip(grid_coords, 0, grid_size - 1)
    
    # 在对应的网格位置增加值
    for coord in grid_coords:
        grid[coord[0], coord[1], coord[2]] += 1
    
    # 执行3D FFT
    fft_result = fftpack.fftn(grid)
    fft_shifted = fftpack.fftshift(fft_result)
    
    # 计算幅度谱
    magnitude_spectrum = np.abs(fft_shifted)
    
    # 计算频率空间中的半径
    freq_cube = np.fft.fftshift(np.fft.fftfreq(grid_size))
    freq_x, freq_y, freq_z = np.meshgrid(freq_cube, freq_cube, freq_cube, indexing='ij')
    freq_radius = np.sqrt(freq_x**2 + freq_y**2 + freq_z**2)
    
    # 计算径向平均轮廓
    # 将半径离散化为索引
    max_radius = np.max(freq_radius)
    radial_bins = 100
    bin_indices = (freq_radius / max_radius * radial_bins).astype(int)
    bin_indices = np.clip(bin_indices, 0, radial_bins - 1)
    
    # 计算每个径向bin的平均幅度
    radial_profile = np.zeros(radial_bins)
    for i in range(radial_bins):
        mask = bin_indices == i
        if np.any(mask):
            radial_profile[i] = np.mean(magnitude_spectrum[mask])
    
    return magnitude_spectrum, freq_radius, radial_profile


def plot_3d_fft_analysis(points, output_dir=None):
    """
    绘制3D FFT频谱分析结果
    
    Args:
        points (numpy.ndarray): 点云数据
        output_dir (str): 图表保存目录，如果为None则显示图表
    """
    # 创建保存目录
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 计算3D FFT
    magnitude_spectrum, freq_radius, radial_profile = compute_3d_fft_spectrum(points)
    
    # 绘制中心切片的频谱图
    center = magnitude_spectrum.shape[0] // 2
    plt.figure(figsize=(10, 8))
    plt.imshow(np.log(1 + magnitude_spectrum[center, :, :]), cmap='hot')
    plt.colorbar(label='Log Magnitude')
    plt.title('3D FFT Spectrum (Center Slice)', fontweight='bold')
    plt.xlabel('Frequency Index')
    plt.ylabel('Frequency Index')
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, "3d_fft_spectrum_slice.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"3D FFT频谱图(中心切片)已保存到: {os.path.join(output_dir, '3d_fft_spectrum_slice.png')}")
    else:
        plt.show()
    
    # 绘制径向平均轮廓
    plt.figure(figsize=(10, 6))
    plt.plot(radial_profile)
    plt.title('Radial Average Profile of 3D FFT Spectrum', fontweight='bold')
    plt.xlabel('Radial Bin')
    plt.ylabel('Average Magnitude')
    plt.grid(True, alpha=0.3)
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, "radial_profile.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"径向平均轮廓图已保存到: {os.path.join(output_dir, 'radial_profile.png')}")
    else:
        plt.show()


def analyze_and_plot_all_histograms(points, output_dir=None):
    """
    对点云数据进行全面的直方图分析并绘制图表
    
    Args:
        points (numpy.ndarray): 点云数据
        output_dir (str): 图表保存目录，如果为None则显示图表
    """
    # 创建保存目录
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 1. Z轴高度直方图
    h_hist, h_bin_edges = compute_h_histogram(points, bins=100, axis=2)
    fig1 = plot_histogram(h_hist, h_bin_edges, 
                         title="Z-Axis Height Histogram", 
                         xlabel="Height (Z-Axis)", 
                         ylabel="Frequency",
                         color='lightcoral')
    
    if output_dir:
        fig1.savefig(os.path.join(output_dir, "z_height_histogram.png"), dpi=300, bbox_inches='tight')
        plt.close(fig1)
        print(f"Z轴高度直方图已保存到: {os.path.join(output_dir, 'z_height_histogram.png')}")
    else:
        plt.show()
    
    # 2. 点投影直方图 (默认Z轴方向)
    ppf_hist, ppf_bin_edges = compute_point_projection_histogram(points, resolution=0.1)
    fig2 = plot_histogram(ppf_hist, ppf_bin_edges,
                         title="Point Projection Histogram (Z-Axis Direction)",
                         xlabel="Projection on Z-Axis",
                         ylabel="Frequency",
                         color='lightgreen')
    
    if output_dir:
        fig2.savefig(os.path.join(output_dir, "projection_histogram.png"), dpi=300, bbox_inches='tight')
        plt.close(fig2)
        print(f"点投影直方图已保存到: {os.path.join(output_dir, 'projection_histogram.png')}")
    else:
        plt.show()
    
    # 3. X轴直方图
    x_hist, x_bin_edges = compute_h_histogram(points, bins=100, axis=0)
    fig3 = plot_histogram(x_hist, x_bin_edges, 
                         title="X-Axis Distribution Histogram", 
                         xlabel="X Coordinate", 
                         ylabel="Frequency",
                         color='gold')
    
    if output_dir:
        fig3.savefig(os.path.join(output_dir, "x_histogram.png"), dpi=300, bbox_inches='tight')
        plt.close(fig3)
        print(f"X轴直方图已保存到: {os.path.join(output_dir, 'x_histogram.png')}")
    else:
        plt.show()
    
    # 4. Y轴直方图
    y_hist, y_bin_edges = compute_h_histogram(points, bins=100, axis=1)
    fig4 = plot_histogram(y_hist, y_bin_edges, 
                         title="Y-Axis Distribution Histogram", 
                         xlabel="Y Coordinate", 
                         ylabel="Frequency",
                         color='plum')
    
    if output_dir:
        fig4.savefig(os.path.join(output_dir, "y_histogram.png"), dpi=300, bbox_inches='tight')
        plt.close(fig4)
        print(f"Y轴直方图已保存到: {os.path.join(output_dir, 'y_histogram.png')}")
    else:
        plt.show()
    
    # 5. 3D FFT频谱分析
    plot_3d_fft_analysis(points, output_dir)


def print_statistical_summary(points):
    """
    打印点云统计摘要
    
    Args:
        points (numpy.ndarray): 点云数据
    """
    print(f"成功读取点云文件，共 {len(points)} 个点")
    print(f"点云数据维度: {points.shape}")
    
    # 显示前几个点的信息
    print("\n前5个点的坐标:")
    for i, point in enumerate(points[:5]):
        if len(point) >= 4:
            print(f"  点 {i+1}: [{point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f}, intensity={point[3]:.3f}]")
        else:
            print(f"  点 {i+1}: [{point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f}]")
    
    # 计算并显示各轴的统计信息
    axes = ['X', 'Y', 'Z']
    for i, axis in enumerate(axes):
        values = points[:, i]
        print(f"\n{axis}轴统计信息:")
        print(f"  最小值: {np.min(values):.3f}")
        print(f"  最大值: {np.max(values):.3f}")
        print(f"  平均值: {np.mean(values):.3f}")
        print(f"  中位数: {np.median(values):.3f}")
        print(f"  标准差: {np.std(values):.3f}")


def main():
    """主函数"""
    # 默认PCD文件路径和输出目录
    default_pcd_path =  "data/lidar/000000.pcd"
    default_output_dir = "data/analyze"
    
    # 检查输入文件是否存在
    if not os.path.exists(default_pcd_path):
        print(f"错误: 默认PCD文件 '{default_pcd_path}' 不存在")
        return
    
    try:
        print("正在读取点云文件...")
        points = read_pcd_file(default_pcd_path)
        
        # 打印统计摘要
        print_statistical_summary(points)
        
        # 创建输出目录
        if not os.path.exists(default_output_dir):
            os.makedirs(default_output_dir)
        
        print(f"\n分析结果将保存到: {default_output_dir}")
        analyze_and_plot_all_histograms(points, default_output_dir)
        print("\n分析完成!")
        
    except Exception as e:
        print(f"处理点云文件时出错: {e}")


if __name__ == "__main__":
    main()
