"""论文图表公共样式与工具函数

所有图表生成脚本统一从此模块导入配置和工具，
避免在各脚本中重复定义 rcParams、调色板、savefig 等。

用法:
    from scripts.chart_style import savefig, C_BLUE, C_RED, ...
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ═════════════════════════════════════════════════════════════════════════
# 全局 rcParams
# ═════════════════════════════════════════════════════════════════════════

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['SimHei', 'Microsoft YaHei', 'SimSun'],
    'axes.unicode_minus': False,
    'font.size': 9,
    'axes.labelsize': 10.5,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
})

# ═════════════════════════════════════════════════════════════════════════
# 调色板（学术风格，低饱和度）
# ═════════════════════════════════════════════════════════════════════════

C_BLUE   = '#457B9D'
C_RED    = '#E63946'
C_GREEN  = '#2A9D8F'
C_YELLOW = '#E9C46A'
C_ORANGE = '#F4A261'
C_GRAY   = '#6C757D'
C_DARK   = '#1D3557'
C_CYAN   = '#A8DADC'

COLORS_10 = [
    '#457B9D', '#E63946', '#2A9D8F', '#E9C46A', '#F4A261',
    '#6D597A', '#B56576', '#219EBC', '#023047', '#8ECAE6',
]

# ═════════════════════════════════════════════════════════════════════════
# 图表尺寸常量（inch）
# ═════════════════════════════════════════════════════════════════════════

# A4 文本区宽度 ≈ 15cm ≈ 5.9in
FIG_W = 5.9

SIZES = {
    'single_bar':      (4.2, 3.2),   # 单柱状图（2-3 组）
    'dual_axis':       (FIG_W, 2.8), # 双轴图
    'dual_panel':      (8.5, 3.8),   # 双栏子图（1×2）
    'triple_panel':    (9.5, 3.5),   # 三栏子图（1×3）
    'pr_curve':        (6.0, 5.0),   # PR 曲线
    'table':           (FIG_W, None),# 表格图（高度自适应）
}

# ═════════════════════════════════════════════════════════════════════════
# 工具函数
# ═════════════════════════════════════════════════════════════════════════

def savefig(fig, path):
    """保存图表至文件（统一参数）"""
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)


def style_ax(ax, grid_axis='y', grid_alpha=0.3):
    """应用统一的坐标轴样式：隐藏上/右边框 + 网格"""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis=grid_axis, alpha=grid_alpha, linewidth=0.4)


def bar_labels(ax, bars, values, suffix='', offset_ratio=0.03, fontsize=7.5):
    """在柱顶添加数值标注

    Args:
        ax: matplotlib Axes
        bars: bar 对象列表（ax.bar 返回值）
        values: 数值列表
        suffix: 后缀字符串（如 '%'）
        offset_ratio: 标注相对柱高的偏移比例
        fontsize: 字号
    """
    y_max = max(values) if values else 1
    offset = y_max * offset_ratio
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + offset,
                f'{v}{suffix}', ha='center', va='bottom',
                fontsize=fontsize, fontweight='bold')


def grouped_bar_labels(ax, x_positions, values_list, colors, suffixes=None,
                       offset_ratio=0.03, fontsize=7.5):
    """为分组柱状图添加数值标注

    Args:
        ax: matplotlib Axes
        x_positions: 每组柱子的 x 坐标（np.arange）
        values_list: [[strategy1 的 P,R,F1], [strategy2 的 P,R,F1], ...]
        colors: 各策略颜色列表
        suffixes: 各指标后缀列表，如 ['%', '%', '']
        offset_ratio: 标注相对柱高的偏移比例
    """
    bar_width = 0.3
    n_strategies = len(values_list)
    n_metrics = len(values_list[0]) if values_list else 0
    all_vals = [v for group in values_list for v in group]
    y_max = max(all_vals) if all_vals else 1
    offset = y_max * offset_ratio

    for i, vals in enumerate(values_list):
        suffixes = suffixes or [''] * n_metrics
        for j, v in enumerate(vals):
            ax.text(x_positions[j] + i * bar_width, v + offset,
                    f'{v:.1f}{suffixes[j] if j < len(suffixes) else ""}',
                    ha='center', va='bottom', fontsize=fontsize,
                    fontweight='bold', color=colors[i])


def auto_find_latest(pattern: str, key_file: str = None) -> list:
    """自动查找最新的输出目录

    Args:
        pattern: glob 模式，如 'eval_ablation_*'
        key_file: 目录内必须存在的文件名，如 'eval_result.json'

    Returns:
        匹配目录列表（按修改时间倒序）
    """
    from pathlib import Path
    dirs = sorted(Path('output').glob(pattern), reverse=True)
    if key_file:
        dirs = [d for d in dirs if (d / key_file).exists()]
    return dirs
