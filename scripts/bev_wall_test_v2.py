"""BEV 投影 + OpenCV 直线检测墙体提取原型（改进版）。

改进点：
  - Z 范围编码：每个栅格存 Z 跨度（墙体点 Z 跨度大，地面附近噪点 Z 跨度小）
  - 多帧叠加：连续多帧 PCD 累积，提高 BEV 密度
  - 角度直方图：统计主方向，只保留墙体典型方向的线段
  - 后处理：合并共线线段，去除短线
"""

import sys
from pathlib import Path
import numpy as np
import cv2

# ── 参数 ──────────────────────────────────────────────
DATA_DIR = Path("data/cloud/lidar")
FRAME_COUNT = 5                 # 叠加帧数
START_FRAME = 101               # 起始帧号
GROUND_EXPAND = 0.10
BEV_RESOLUTION = 0.02
BEV_MAX_RANGE = 10.0
MIN_LINE_LENGTH = 30            # 像素
MAX_LINE_GAP = 20
HOUGH_THRESHOLD = 20

# 墙体角度容差（度）— 室内墙体通常是正交方向 ± 容差
WALL_ANGLES = [0, 90]
ANGLE_TOLERANCE = 10

# Z 跨度最小值（墙体垂直方向延伸）
MIN_Z_SPAN = 0.8


# ── 数据加载 ──────────────────────────────────────────
def load_pcd(path: str) -> np.ndarray:
    with open(path) as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if line.strip() == "DATA ascii":
                header_end = i + 1
                break
    data = np.loadtxt(lines[header_end:], dtype=np.float32)
    if data.ndim == 1:
        data = data.reshape(1, 3)
    return data


# ── 地面提取 ──────────────────────────────────────────
def extract_ground(points: np.ndarray, expand: float = 0.10) -> np.ndarray:
    """直方图地面提取，返回 ground_mask。"""
    z = points[:, 2]
    z_min, z_max = float(z.min()), float(z.max())
    if z_max - z_min < 1e-6:
        return np.zeros(len(points), dtype=bool)

    num_bins = 128
    bin_w = (z_max - z_min) / num_bins
    bins = np.zeros(num_bins, dtype=np.int32)
    for zi in z:
        b = min(int((zi - z_min) / bin_w), num_bins - 1)
        bins[b] += 1

    peak_bin = int(np.argmax(bins))
    peak_z = z_min + (peak_bin + 0.5) * bin_w

    z_low = peak_z - expand
    z_high = peak_z + expand
    return (z >= z_low) & (z <= z_high)


# ── BEV 编码：栅格存储 Z 跨度（区分墙体和噪点）────────────
def points_to_bev_zspan(points: np.ndarray,
                        resolution: float = 0.02,
                        max_range: float = 10.0) -> np.ndarray:
    """每个栅格统计 Z 跨度 → 归一化到 [0,255]。

    墙体点垂直延伸 → Z 跨度大 → 高亮。
    噪点/物体点分布集中 → Z 跨度小 → 暗淡。
    """
    xy = points[:, :2]
    z = points[:, 2]

    mask = (np.abs(xy[:, 0]) < max_range) & (np.abs(xy[:, 1]) < max_range)
    xy, z = xy[mask], z[mask]

    offset = max_range
    size = int(2 * max_range / resolution)

    z_min_grid = np.full((size, size), np.inf, dtype=np.float32)
    z_max_grid = np.full((size, size), -np.inf, dtype=np.float32)

    xs = ((xy[:, 0] + offset) / resolution).astype(np.int32)
    ys = ((xy[:, 1] + offset) / resolution).astype(np.int32)
    xs = np.clip(xs, 0, size - 1)
    ys = np.clip(ys, 0, size - 1)

    for i in range(len(xs)):
        x, y = xs[i], ys[i]
        if z[i] < z_min_grid[y, x]:
            z_min_grid[y, x] = z[i]
        if z[i] > z_max_grid[y, x]:
            z_max_grid[y, x] = z[i]

    z_span = z_max_grid - z_min_grid
    # 处理无点栅格和异常值
    z_span[~np.isfinite(z_span)] = 0.0
    z_span = np.maximum(z_span, 0.0)

    # 对数归一化，增强弱响应
    img = np.log1p(z_span)
    max_val = img.max()
    if max_val > 1e-6:
        img = (img / max_val * 255).astype(np.uint8)
    else:
        img = np.zeros((size, size), dtype=np.uint8)
    return img


# ── 辅助 BEV：密度编码（作为补充）────────────────────────
def points_to_bev_density(points: np.ndarray,
                          resolution: float = 0.02,
                          max_range: float = 10.0) -> np.ndarray:
    xy = points[:, :2]
    mask = (np.abs(xy[:, 0]) < max_range) & (np.abs(xy[:, 1]) < max_range)
    xy = xy[mask]

    offset = max_range
    size = int(2 * max_range / resolution)
    img = np.zeros((size, size), dtype=np.float32)

    xs = ((xy[:, 0] + offset) / resolution).astype(np.int32)
    ys = ((xy[:, 1] + offset) / resolution).astype(np.int32)
    xs = np.clip(xs, 0, size - 1)
    ys = np.clip(ys, 0, size - 1)
    np.add.at(img, (ys, xs), 1)

    img = np.log1p(img)
    if img.max() > 0:
        img = (img / img.max() * 255).astype(np.uint8)
    return img


# ── 直线检测 ──────────────────────────────────────────
def filter_wall_angles_adaptive(lines: np.ndarray,
                                angle_tol: float = 15.0) -> np.ndarray:
    """自适应角度过滤：找到所有线段的 2 个主方向，保留匹配的线段。

    室内墙体通常有 1-2 个主方向（正交布局）。
    """
    if len(lines) < 3:
        return lines

    # 计算每条线段的角度 [0, 180)
    x1, y1, x2, y2 = lines[:, 0], lines[:, 1], lines[:, 2], lines[:, 3]
    dx = x2 - x1
    dy = y2 - y1
    angles = np.degrees(np.arctan2(dy, dx)) % 180

    # 角度直方图聚类找到主方向
    hist, bins = np.histogram(angles, bins=36, range=(0, 180))  # 5度/箱
    peaks = []
    for i in range(len(hist)):
        if hist[i] > 0 and hist[i] == np.max(hist):
            peaks.append((bins[i] + bins[i + 1]) / 2)

    # 找 top-2 峰值（考虑 90 度折叠：如果两个方向接近正交则保留）
    sorted_idx = np.argsort(hist)[::-1]
    dominant = []
    for idx in sorted_idx:
        if len(dominant) >= 2:
            break
        angle = (bins[idx] + bins[idx + 1]) / 2
        # 检查是否与已有方向重复
        is_dup = any(
            min(abs(angle - d), 180 - abs(angle - d)) < angle_tol
            for d in dominant
        )
        if not is_dup:
            dominant.append(angle)

    if not dominant:
        dominant = [angles.mean()]

    # 保留匹配主方向的线段
    mask = np.zeros(len(lines), dtype=bool)
    for i, angle in enumerate(angles):
        for d in dominant:
            diff = min(abs(angle - d), 180 - abs(angle - d))
            if diff < angle_tol:
                mask[i] = True
                break

    return lines[mask]


def detect_lines_hough(img: np.ndarray) -> np.ndarray:
    """HoughLinesP 直线检测 + 形态学预处理。"""
    # 自适应直方图均衡化增强对比度
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img)

    blur = cv2.GaussianBlur(enhanced, (3, 3), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 轻量形态学闭运算连接墙体断线
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    lines = cv2.HoughLinesP(
        binary, rho=1, theta=np.pi / 360,
        threshold=HOUGH_THRESHOLD,
        minLineLength=MIN_LINE_LENGTH,
        maxLineGap=MAX_LINE_GAP,
    )
    if lines is None:
        return np.empty((0, 4))
    lines = lines.squeeze(1)
    return filter_wall_angles_adaptive(lines)


def detect_lines_lsd(img: np.ndarray) -> np.ndarray:
    """LSD 直线检测。"""
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    lines, _, _, _ = lsd.detect(img)
    if lines is None:
        return np.empty((0, 4))
    lines = lines.squeeze(1)

    # 过滤短线
    dx = lines[:, 2] - lines[:, 0]
    dy = lines[:, 3] - lines[:, 1]
    lengths = np.sqrt(dx ** 2 + dy ** 2)
    lines = lines[lengths > MIN_LINE_LENGTH]

    return filter_wall_angles_adaptive(lines)


# ── 合并近似线段 ──────────────────────────────────────
def merge_collinear(lines: np.ndarray,
                    angle_tol: float = 5.0,
                    dist_tol: float = 10.0) -> np.ndarray:
    """合并共线且相邻的线段。"""
    if len(lines) < 2:
        return lines

    # 每条线的方向和角度
    dx = lines[:, 2] - lines[:, 0]
    dy = lines[:, 3] - lines[:, 1]
    angles = np.degrees(np.arctan2(dy, dx)) % 180

    merged = []
    used = set()

    for i in range(len(lines)):
        if i in used:
            continue
        group = [i]
        used.add(i)
        for j in range(i + 1, len(lines)):
            if j in used:
                continue
            # 角度差
            d_angle = abs(angles[i] - angles[j])
            d_angle = min(d_angle, 180 - d_angle)
            if d_angle > angle_tol:
                continue
            # 端点距离
            ends = np.array([
                [lines[i][0], lines[i][1]],
                [lines[i][2], lines[i][3]],
                [lines[j][0], lines[j][1]],
                [lines[j][2], lines[j][3]],
            ])
            # 检查线段端点是否接近
            d_ends = np.linalg.norm(ends[:, None] - ends[None, :], axis=2)
            close = d_ends < dist_tol
            if close[:2, 2:].any() or close[2:, :2].any():
                group.append(j)
                used.add(j)

        if len(group) == 1:
            merged.append(lines[i])
        else:
            # 取整个组的最小/最大端点
            xs = np.concatenate([lines[g][[0, 2]] for g in group])
            ys = np.concatenate([lines[g][[1, 3]] for g in group])
            # 投影到主方向
            angle = np.mean([angles[g] for g in group])
            theta = np.radians(angle)
            proj = xs * np.cos(theta) + ys * np.sin(theta)
            i0, i1 = np.argmin(proj), np.argmax(proj)
            merged.append([xs[i0], ys[i0], xs[i1], ys[i1]])

    return np.array(merged)


# ── 可视化 ──────────────────────────────────────────
def draw_lines(img: np.ndarray, lines: np.ndarray,
               color: tuple = (0, 255, 0)) -> np.ndarray:
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for line in lines:
        x1, y1, x2, y2 = map(int, line[:4])
        cv2.line(vis, (x1, y1), (x2, y2), color, 2)
    return vis


# ── 主流程 ──────────────────────────────────────────
def main():
    print("=" * 60)
    print("BEV + OpenCV 墙体检测原型 v2")
    print("=" * 60)

    # 1. 加载多帧点云
    print(f"[1/4] 加载点云 ({FRAME_COUNT} 帧)")
    all_points = []
    for i in range(START_FRAME, START_FRAME + FRAME_COUNT):
        pcd_path = DATA_DIR / f"{i:06d}.pcd"
        if not pcd_path.exists():
            continue
        pts = load_pcd(str(pcd_path))
        # 剔除原点 (0,0,0) 无效点
        pts = pts[~(np.all(np.abs(pts) < 1e-6, axis=1))]
        all_points.append(pts)

    if not all_points:
        print("  错误：未找到 PCD 文件")
        return

    cloud = np.concatenate(all_points, axis=0)
    print(f"  总点云: {len(cloud)} 点")

    # 2. 地面提取
    print("[2/4] 地面提取 (histogram, expand={})".format(GROUND_EXPAND))
    ground_mask = extract_ground(cloud, GROUND_EXPAND)
    nonground = cloud[~ground_mask]
    print(f"  非地面点: {len(nonground)} / {len(cloud)}")

    # 3. BEV 投影（Z 跨度编码 + 密度编码融合）
    print("[3/4] BEV 投影")
    bev_zspan = points_to_bev_zspan(nonground, BEV_RESOLUTION, BEV_MAX_RANGE)
    bev_density = points_to_bev_density(nonground, BEV_RESOLUTION, BEV_MAX_RANGE)
    # 融合：Z 跨度图 + 密度图
    bev_fused = cv2.addWeighted(bev_zspan, 0.7, bev_density, 0.3, 0)
    print(f"  BEV: {bev_fused.shape}")

    # 4. 直线检测
    print("[4/4] 直线检测")
    lines_hough = detect_lines_hough(bev_fused)
    lines_hough = merge_collinear(lines_hough)
    print(f"  HoughLinesP: {len(lines_hough)} 条线段")

    lines_lsd = detect_lines_lsd(bev_fused)
    if len(lines_lsd) > 0:
        lines_lsd = merge_collinear(lines_lsd)
    print(f"  LSD: {len(lines_lsd)} 条线段")

    # 5. 保存结果
    out_dir = Path("output/bev_test")
    out_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(out_dir / "bev_zspan.png"), bev_zspan)
    cv2.imwrite(str(out_dir / "bev_density.png"), bev_density)
    cv2.imwrite(str(out_dir / "bev_fused.png"), bev_fused)

    vis_hough = draw_lines(bev_fused, lines_hough, (0, 255, 0))
    cv2.imwrite(str(out_dir / "bev_hough.png"), vis_hough)

    vis_lsd = draw_lines(bev_fused, lines_lsd, (0, 255, 255))
    cv2.imwrite(str(out_dir / "bev_lsd.png"), vis_lsd)

    # 并排对比
    h, w = bev_fused.shape
    gap = 10
    vis_w = w
    canvas = np.zeros((h, 4 * vis_w + 3 * gap, 3), dtype=np.uint8)
    panels = [
        (bev_zspan, "Z 跨度编码"),
        (bev_density, "密度编码"),
        (vis_hough, "HoughLinesP"),
        (vis_lsd, "LSD"),
    ]
    for i, (panel, label) in enumerate(panels):
        x = i * (vis_w + gap)
        if panel.ndim == 2:
            panel_color = cv2.cvtColor(panel, cv2.COLOR_GRAY2BGR)
        else:
            panel_color = panel
        canvas[:, x:x + vis_w] = panel_color
        cv2.putText(canvas, label, (x + 10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imwrite(str(out_dir / "bev_v2_comparison.png"), canvas)

    print(f"\n结果保存至: {out_dir}/")
    for f in ["bev_zspan.png", "bev_density.png", "bev_fused.png",
              "bev_hough.png", "bev_lsd.png", "bev_v2_comparison.png"]:
        print(f"  {f}")


if __name__ == "__main__":
    main()
