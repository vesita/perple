# 论文图表风格指南

统一所有 Python 图表生成脚本的 matplotlib 样式配置，确保论文所有图表视觉风格一致。

---

## 1. 全局 rcParams

所有脚本的开头应使用以下统一配置：

```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
```

**要点：**
- `font.sans-serif` 第一候选为 `SimHei`（黑体），回退 `Microsoft YaHei`、`SimSun`
- 不要使用 `savefig.bbox` 或 `savefig.bbox_inches` 作为 rcParams（无效键）
- `figure.dpi` 统一 300，不区分 figure 和 savefig 的 dpi

---

## 2. 保存函数

统一使用包装函数：

```python
def savefig(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
```

**要点：**
- `tight_layout()` 在 savefig 前调用
- `bbox_inches='tight'` 作为 savefig 参数而非 rcParams
- `plt.close(fig)` 紧跟释放内存

---

## 3. 调色板

```python
# 学术风格，低饱和度
C_BLUE   = '#457B9D'
C_RED    = '#E63946'
C_GREEN  = '#2A9D8F'
C_YELLOW = '#E9C46A'
C_ORANGE = '#F4A261'
C_GRAY   = '#6C757D'
C_DARK   = '#1D3557'
C_CYAN   = '#A8DADC'

# 多分类（最多 10 色）
COLORS_10 = [
    '#457B9D', '#E63946', '#2A9D8F', '#E9C46A', '#F4A261',
    '#6D597A', '#B56576', '#219EBC', '#023047', '#8ECAE6',
]
```

**要点：**
- 对比图用 `C_BLUE` vs `C_RED`，或 `C_BLUE` vs `C_ORANGE`
- 三指标（P/R/F1）分别用 `C_BLUE` / `C_GREEN` / `C_RED`

---

## 4. 通用图表样式

| 元素 | 规则 |
|------|------|
| **上/右边框** | `ax.spines['top'].set_visible(False)` + `ax.spines['right'].set_visible(False)` |
| **网格** | `ax.grid(axis='y', alpha=0.3, linewidth=0.4)` — 仅 Y 方向网格 |
| **图例** | `framealpha=0.85`，`edgecolor='#cccccc'`，位置自适应 |
| **数值标注** | 柱状图柱顶数字，`fontsize=7.5`，`fontweight='bold'` |
| **百分比** | 精确率/召回率标注加 `%` 后缀；F1 不加 |
| **坐标轴标签** | 中文，精确率/召回率用"精确率 P"、"召回率 R" |
| **图标题** | 无（不在图上显示标题），标注用 (a)(b) 子图标签 |
| **单图单图表** | 每个 Figure 只包含一个 Axes（`plt.subplots()` 不带行列参数），不使用多面板子图合并 |

---

## 5. 图片尺寸

| 图片类型 | 宽度 (inch) | 高度 (inch) |
|----------|------------|------------|
| 单柱状图（2-3 组） | 4.2 | 3.2 |
| 双轴图/折线图 | 5.9 | 2.8 |
| 表格图 | 5.9 | 按行数自适应 |
| PR 曲线 | 6.0 | 5.0 |

**原则：** 每个 Figure 仅包含一个图表，单图不超过 A4 文本区宽度（15cm ≈ 5.9in）。
**注意：** 不再使用 `dual_panel`（1×2）和 `triple_panel`（1×3）等多面板尺寸。

---

## 6. 输出目录

| 脚本 | 输出目录 |
|------|---------|
| `thesis_viz.py` | `output/thesis_figures/` |
| `bench_viz.py` | `output/bench_viz/{task}/` |
| `eval_viz.py` | `output/eval_viz/` |
| `ablation_charts.py` | `output/ablation_5run/charts/` |
| `edlines_compare_viz.py` | `output/edlines_bench/` |
| `edlines_labeled_viz.py` | `output/edlines_bench/` |
| `wall_strategy_analysis.py` | `output/wall_strategy_analysis/` |
| `viz_pr_curve.py` | `output/`（自动目录） |
| `viz_summary.py` | `output/`（自动目录） |
| `viz_trajectory.py` | `output/`（自动目录） |

---

## 7. 需修复的不一致问题

### 7.1 `thesis_viz.py`
- `savefig.bbox` rcParams 键无效 → 改用 `savefig(path, bbox_inches='tight')`
- `fig.tight_layout()` 调用正确，但应移入统一 savefig 函数
- 图中仍有图标题（如 `ax.set_title(...)`）→ 应移除，改为 (a)(b) 子图标注

### 7.2 `bench_viz.py`
- `axes.titlesize=14` 偏大 → 改为 12
- `figure.dpi=300` 已正确
- 图中仍有 `ax.set_title(...)` → 考虑移除或改为子图标签
- `plt.tight_layout()` 调用在 savefig 前，但未统一

### 7.3 `eval_viz.py`
- 字体定义使用独立常量（`FONT_TITLE=14`, `FONT_LABEL=10.5` 等）→ 应与 rcParams 统一
- `plt.tight_layout()` 调用正确
- 柱状图颜色使用 `#2ecc71` / `#e74c3c` → 改用调色板 `C_GREEN` / `C_RED`

### 7.4 `viz_pr_curve.py` / `viz_summary.py` / `viz_trajectory.py`
- `font.sans-serif` 只有 `Microsoft YaHei` → 改为 `['SimHei', 'Microsoft YaHei', 'SimSun']`
- `font.size=11` → 改为 9
- `savefig.bbox` rcParams 键无效 → 删除
- `figure.dpi=200` → 改为 300

### 7.5 `ablation_charts.py`
- `font.family='SimHei'` → 改为 `'sans-serif'` + `font.sans-serif` 列表
- `font.size=10.5` → 改为 9
- `savefig()` 包装函数已正确
- 无 `plt.tight_layout()` → 在 `savefig()` 内添加
- 无 `spines` 设置 → 添加移除上/右边框

### 7.6 `plot_accuracy_combined.py`
- 未使用 rcParams 统一配置 → 补全

---

## 8. 柱状图标注规则

| 指标 | 单位 | 标注格式 | 示例 |
|------|------|---------|------|
| 精确率 P | % | `f'{v:.1f}%'` | `84.6%` |
| 召回率 R | % | `f'{v:.1f}%'` | `68.5%` |
| F1 | % | `f'{v:.1f}%'` | `75.7%` |
| 数量（TP/FP/FN） | 整数 | `f'{v}'` | `964` |
| 耗时 | ms | `f'{v:.0f}ms'` | `29ms` |

---

## 9. 文件命名

```
{图号}_{描述}.png
```

例：`fig01_main_metrics.png`、`fig02_strategy_comparison.png`、`person_metrics.png`。

---

> 最后更新：2026-05-18
> 参考脚本：`scripts/thesis_viz.py`、`scripts/bench_viz.py`、`scripts/eval_viz.py`、`scripts/ablation_charts.py`
