"""BEV (Bird's Eye View) 投影 + OpenCV 直线检测墙体提取原型。

流程：
  1. 读取 PCD → 直方图地面提取 → 去地面
  2. XY 投影 → 2D 占用栅格图像
  3. OpenCV HoughLinesP / LSD 直线检测
  4. 可视化
"""

import sys
from pathlib import Path
import numpy as np
import cv2
from dataclasses import dataclass
from typing import Optional

# ── BevEDLines 常量（匹配 Rust common.rs） ──────────
EDGE_VERTICAL = 1      # |gx| >= |gy|
EDGE_HORIZONTAL = 2    # |gy| > |gx|
LEFT = 3
RIGHT = 4
UP = 5
DOWN = 6

# ── 参数 ──────────────────────────────────────────────
PCD_PATH = Path("data/cloud/lidar/000101.pcd")
GROUND_EXPAND = 0.10        # 地面 Z 扩展量（米），匹配 Rust ground_expand
GROUND_THRESHOLD = 0.15     # 直方图峰值比例阈值，匹配 Rust PeakScan threshold
BEV_RESOLUTION = 0.02       # 栅格分辨率（米/像素）
BEV_MAX_RANGE = 10.0        # 最大显示范围（米）
MIN_LINE_LENGTH = 30        # 最短线段（像素）
MIN_LINE_GAP = 10           # 线段间隙容差（像素）
HOUGH_THRESHOLD = 30        # Hough 投票阈值

# ── 数据加载 ──────────────────────────────────────────
def load_pcd(path: str) -> np.ndarray:
    """加载 ASCII PCD，返回 (N, 3) float32 点云。"""
    with open(path) as f:
        header_end = 0
        lines = f.readlines()
        for i, line in enumerate(lines):
            if line.strip() == "DATA ascii":
                header_end = i + 1
                break
    data = np.loadtxt(lines[header_end:], dtype=np.float32)
    if data.ndim == 1:
        data = data.reshape(1, 3)
    print(f"  加载 {len(data)} 点")
    return data

# ── 地面提取（直方图法）─────────────────────────────────
def extract_ground(points: np.ndarray, expand: float = 0.10,
                   threshold: float = 0.15, upside_down: bool = True) -> tuple:
    """PeakScan 地面提取，完整匹配 Rust peak_scan.rs。

    算法：
      1. upside_down 时翻转 Z（LiDAR 倒装：p[2] = -p[2]）
      2. 按 Z 排序 → 128 箱直方图
      3. 找峰值（upside_down 时找首个 > 均值的 bin）
      4. 从峰值向下扫描，找 bin 计数 < threshold×peak_count → z_lower
      5. 上界 = peak_z + expand
      6. 地面 = [z_lower, z_upper]
    """
    z = points[:, 2].copy()
    if upside_down:
        z = -z  # 翻转 Z

    sort_idx = np.argsort(z)
    z_sorted = z[sort_idx]
    n = len(points)

    z_min, z_max = float(z_sorted[0]), float(z_sorted[-1])
    if z_max - z_min < 1e-6:
        return np.zeros(n, dtype=bool), 0.0

    num_bins = 128
    bin_w = (z_max - z_min) / num_bins
    bins = np.zeros(num_bins, dtype=np.int32)
    for zi in z:
        b = min(int((zi - z_min) / bin_w), num_bins - 1)
        bins[b] += 1

    # 找峰值（匹配 Rust find_peak_bin）
    if upside_down:
        avg = int(bins.sum() / num_bins)
        peak_bin = next((i for i, c in enumerate(bins) if c > avg), 0)
    else:
        peak_bin = int(np.argmax(bins))

    peak_count = bins[peak_bin]
    peak_z = z_min + (peak_bin + 0.5) * bin_w
    t = max(int(peak_count * threshold), 1)

    # 向下扫描找地面起始 bin
    ground_start_bin = 0
    for i in range(peak_bin - 1, -1, -1):
        if bins[i] < t:
            ground_start_bin = i + 1
            break

    z_lower = z_min + ground_start_bin * bin_w
    z_upper = peak_z + expand

    # 找到实际点索引范围
    ground_start = 0
    for i in range(n):
        if z_sorted[i] >= z_lower:
            ground_start = i
            break
    ground_end = n
    for i in range(n - 1, -1, -1):
        if z_sorted[i] <= z_upper:
            ground_end = i + 1
            break

    # 构建 mask（原始点顺序，注意翻转后需映射回原始 Z 空间）
    if upside_down:
        z_original = points[:, 2]
        z_lower_orig = -z_upper   # 翻转回去
        z_upper_orig = -z_lower
        mask = (z_original >= z_lower_orig) & (z_original <= z_upper_orig)
    else:
        mask = np.zeros(n, dtype=bool)
        for i in range(ground_start, ground_end):
            mask[sort_idx[i]] = True

    n_ground = mask.sum()
    peak_z_orig = -peak_z if upside_down else peak_z
    print(f"  地面: peak_z={peak_z_orig:.3f}, 翻转={upside_down}, "
          f"区间 [{z_lower_orig if upside_down else z_lower:.3f}, "
          f"{z_upper_orig if upside_down else z_upper:.3f}], "
          f"{n_ground} / {n} 点")
    return mask, peak_z_orig

# ── BEV 投影 ──────────────────────────────────────────
def points_to_bev(points: np.ndarray, resolution: float = 0.02,
                  max_range: float = 10.0) -> tuple:
    """点云 XY 投影到 BEV 图像，返回 (img, offset_x, offset_y)。

    每个栅格统计点密度，密度归一化到 [0,255]。
    """
    xy = points[:, :2]
    # 范围裁剪
    mask = (np.abs(xy[:, 0]) < max_range) & (np.abs(xy[:, 1]) < max_range)
    xy = xy[mask]
    print(f"  BEV 范围内: {len(xy)} / {len(points)} 点")

    offset = max_range
    size = int(2 * max_range / resolution)
    img = np.zeros((size, size), dtype=np.float32)

    xs = ((xy[:, 0] + offset) / resolution).astype(np.int32)
    ys = ((xy[:, 1] + offset) / resolution).astype(np.int32)
    xs = np.clip(xs, 0, size - 1)
    ys = np.clip(ys, 0, size - 1)
    np.add.at(img, (ys, xs), 1)

    # 对数归一化增强弱响应
    img = np.log1p(img)
    img = (img / img.max() * 255).astype(np.uint8) if img.max() > 0 else img
    return img, -offset, -offset

# ── 直线检测 ──────────────────────────────────────────
def detect_lines(img: np.ndarray) -> list:
    """HoughLinesP 直线检测。"""
    # 自适应二值化
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 形态学闭运算连接断线
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    lines = cv2.HoughLinesP(
        binary, rho=1, theta=np.pi / 180,
        threshold=HOUGH_THRESHOLD,
        minLineLength=MIN_LINE_LENGTH,
        maxLineGap=MIN_LINE_GAP,
    )
    return lines if lines is not None else []

def detect_lines_edlines(img: np.ndarray,
                         sigma: float = 1.0,
                         anchor_threshold: int = 8,
                         min_line_length: float = 30.0) -> np.ndarray:
    """EDLines 锚点检测 + 链式追踪直线提取。"""
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)

    h, w = img.shape
    mag = np.abs(gx) + np.abs(gy)
    direction = np.abs(gx) >= np.abs(gy)

    anchors = []
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if mag[y, x] < anchor_threshold:
                continue
            if direction[y, x]:
                if mag[y, x] > mag[y, x - 1] and mag[y, x] >= mag[y, x + 1]:
                    anchors.append((x, y))
            else:
                if mag[y, x] > mag[y - 1, x] and mag[y, x] >= mag[y + 1, x]:
                    anchors.append((x, y))

    if len(anchors) < 5:
        return np.empty((0, 4))

    visited = np.zeros((h, w), dtype=bool)
    chains = []
    edge_dir = [(-1, -1), (0, -1), (1, -1), (-1, 0),
                (1, 0), (-1, 1), (0, 1), (1, 1)]

    for ax, ay in anchors:
        if visited[ay, ax]:
            continue
        chain = [(ax, ay)]
        visited[ay, ax] = True
        for sign in [-1, 1]:
            cx, cy = ax, ay
            while True:
                best = None
                best_mag = -1
                for dx, dy in edge_dir:
                    nx, ny = cx + dx, cy + dy
                    if nx < 0 or nx >= w or ny < 0 or ny >= h:
                        continue
                    if visited[ny, nx]:
                        continue
                    if mag[ny, nx] < anchor_threshold * 0.5:
                        continue
                    if direction[ny, nx] != direction[ay, ax]:
                        continue
                    if mag[ny, nx] > best_mag:
                        best_mag = mag[ny, nx]
                        best = (nx, ny)
                if best is None:
                    break
                nx, ny = best
                visited[ny, nx] = True
                if sign == 1:
                    chain.append((nx, ny))
                else:
                    chain.insert(0, (nx, ny))
                cx, cy = nx, ny
        if len(chain) >= int(min_line_length * 0.3):
            chains.append(chain)

    if not chains:
        return np.empty((0, 4))

    lines = []
    for chain in chains:
        pts = np.array(chain, dtype=np.float32)
        xs, ys = pts[:, 0], pts[:, 1]
        n = len(pts)
        if n < 4:
            continue
        mean_x, mean_y = xs.mean(), ys.mean()
        dx = xs - mean_x
        dy = ys - mean_y
        cov = np.array([[np.sum(dx * dx), np.sum(dx * dy)],
                        [np.sum(dx * dy), np.sum(dy * dy)]]) / n
        eigvals, eigvecs = np.linalg.eigh(cov)
        main_dir = eigvecs[:, np.argmax(eigvals)]
        proj = xs * main_dir[0] + ys * main_dir[1]
        i0, i1 = np.argmin(proj), np.argmax(proj)
        x1, y1 = xs[i0], ys[i0]
        x2, y2 = xs[i1], ys[i1]
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if length >= min_line_length:
            lines.append([x1, y1, x2, y2])

    return np.array(lines, dtype=np.float32) if lines else np.empty((0, 4))


# ════════════════════════════════════════════════════════════
# BevEDLines 完整实现（移植自 Rust src/cloud/wall/bev_edlines.rs）
# ════════════════════════════════════════════════════════════

def bev_encode(cloud: np.ndarray, size: int, max_range: float, resolution: float) -> np.ndarray:
    """点云 XY 投影到 BEV 栅格，log1p 归一化到 [0,255] 一维数组。

    匹配 Rust common.rs bev_encode()。
    """
    bev = np.zeros(size * size, dtype=np.uint32)
    for p in cloud:
        if abs(p[0]) >= max_range or abs(p[1]) >= max_range:
            continue
        x = int((p[0] + max_range) / resolution)
        y = int((p[1] + max_range) / resolution)
        if 0 <= x < size and 0 <= y < size:
            bev[y * size + x] += 1

    img_f32 = np.log1p(bev.astype(np.float32))
    max_val = img_f32.max()
    if max_val > 1e-6:
        img = (img_f32 / max_val * 255).astype(np.uint8)
    else:
        img = np.zeros(size * size, dtype=np.uint8)
    return img


def gaussian_blur(src: np.ndarray, w: int, h: int, sigma: float) -> np.ndarray:
    """可分离 1D 高斯模糊，边界 clamp。

    匹配 Rust common.rs gaussian_blur()。
    """
    radius = int(np.ceil(sigma * 2.5))
    size = 2 * radius + 1
    kernel = np.zeros(size, dtype=np.float32)
    for i in range(size):
        x = i - radius
        kernel[i] = np.exp(-0.5 * x * x / (sigma * sigma))
    kernel /= kernel.sum()

    tmp = np.zeros(w * h, dtype=np.float32)
    for y in range(h):
        for x in range(w):
            val = 0.0
            for ki in range(size):
                sx = max(0, min(w - 1, x + ki - radius))
                val += src[y * w + sx] * kernel[ki]
            tmp[y * w + x] = val

    out = np.zeros(w * h, dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            val = 0.0
            for ki in range(size):
                sy = max(0, min(h - 1, y + ki - radius))
                val += tmp[sy * w + x] * kernel[ki]
            out[y * w + x] = np.clip(round(val), 0, 255)
    return out


def sobel_gradient(src: np.ndarray, w: int, h: int):
    """Sobel 3×3 梯度，返回幅值（|gx|+|gy|）和二进制方向。

    匹配 Rust bev_edlines.rs sobel_gradient()。
    """
    s = src.astype(np.int32)
    mag = np.zeros(w * h, dtype=np.float32)
    direction = np.zeros(w * h, dtype=np.uint8)
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            i = y * w + x
            gx = (-1 * s[i - w - 1] + 1 * s[i - w + 1]
                  -2 * s[i - 1]     + 2 * s[i + 1]
                  -1 * s[i + w - 1] + 1 * s[i + w + 1])
            gy = (-1 * s[i - w - 1] - 2 * s[i - w] - 1 * s[i - w + 1]
                  +1 * s[i + w - 1] + 2 * s[i + w] + 1 * s[i + w + 1])
            gx_abs = abs(gx)
            gy_abs = abs(gy)
            mag[i] = gx_abs + gy_abs
            direction[i] = EDGE_VERTICAL if gx_abs >= gy_abs else EDGE_HORIZONTAL
    return mag, direction


def walk_edge_chain(grad_mag: np.ndarray, grad_dir: np.ndarray,
                    w: int, h: int, mag_threshold: float,
                    sx: int, sy: int, direction: int,
                    edges: np.ndarray, chain: list):
    """从锚点沿指定方向追踪边缘链。

    匹配 Rust common.rs walk_edge_chain()。
    """
    if not chain:
        chain.append((sx, sy))
        edges[sy * w + sx] = -1.0

    x, y = sx, sy
    if direction == LEFT:
        step_x, step_y = -1, 0
    elif direction == RIGHT:
        step_x, step_y = 1, 0
    elif direction == UP:
        step_x, step_y = 0, -1
    elif direction == DOWN:
        step_x, step_y = 0, 1
    else:
        return

    while True:
        best_i = None
        best_mag = mag_threshold

        cands = [
            (x + step_x, y + step_y),
            (x + step_x + step_y, y + step_y - step_x),
            (x + step_x - step_y, y + step_y + step_x),
        ]
        for cx, cy in cands:
            if 1 <= cx < w - 1 and 1 <= cy < h - 1:
                ci = cy * w + cx
                if not np.isfinite(edges[ci]):
                    m = grad_mag[ci]
                    if m > best_mag:
                        best_mag = m
                        best_i = (cx, cy)

        if best_i is None:
            break
        nx, ny = best_i
        ni = ny * w + nx
        if direction in (LEFT, RIGHT) and grad_dir[ni] != EDGE_HORIZONTAL:
            break
        if direction in (UP, DOWN) and grad_dir[ni] != EDGE_VERTICAL:
            break
        chain.append((nx, ny))
        edges[ni] = -1.0
        x, y = nx, ny


def split_chain_by_curvature(chain, max_error: float):
    """对边缘链按曲率递归分裂。

    匹配 Rust common.rs split_chain_by_curvature()。
    """
    if len(chain) < 4:
        return [list(chain)]
    segments = []
    _split_recursive(chain, 0, len(chain) - 1, max_error, segments)
    return segments


def _split_recursive(chain, start: int, end: int, max_error: float, segments: list):
    if end - start < 3:
        segments.append(list(chain[start:end + 1]))
        return

    x1, y1 = chain[start]
    x2, y2 = chain[end]
    dx = x2 - x1
    dy = y2 - y1
    len2 = dx * dx + dy * dy
    if len2 < 1e-6:
        segments.append(list(chain[start:end + 1]))
        return

    max_dist = 0.0
    split_idx = start
    for i in range(start + 1, end):
        px, py = chain[i]
        dist = abs((py - y1) * dx - (px - x1) * dy) / np.sqrt(len2)
        if dist > max_dist:
            max_dist = dist
            split_idx = i

    if max_dist > max_error and split_idx > start and split_idx < end:
        _split_recursive(chain, start, split_idx, max_error, segments)
        _split_recursive(chain, split_idx, end, max_error, segments)
    else:
        segments.append(list(chain[start:end + 1]))


def fit_rectangle(region):
    """PCA 最小外接矩形拟合。

    匹配 Rust common.rs fit_rectangle()。
    返回 (cx, cy, length, width, angle)。
    """
    n = len(region)
    xs = np.array([p[0] for p in region], dtype=np.float32)
    ys = np.array([p[1] for p in region], dtype=np.float32)
    cx = xs.mean()
    cy = ys.mean()

    dx = xs - cx
    dy = ys - cy
    xx = np.dot(dx, dx)
    xy = np.dot(dx, dy)
    yy = np.dot(dy, dy)

    if abs(xy) > 1e-6:
        trace = xx + yy
        det = xx * yy - xy * xy
        sqrt_term = np.sqrt(max(trace * trace / 4.0 - det, 0.0))
        lambda1 = trace / 2.0 + sqrt_term
        angle = np.arctan2(lambda1 - xx, xy)
    else:
        angle = 0.0

    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    proj = dx * cos_a + dy * sin_a
    perp = -dx * sin_a + dy * cos_a

    length = proj.max() - proj.min()
    width = perp.max() - perp.min()
    return cx, cy, length, width, angle


def fit_rms_error(region, cx: float, cy: float, angle: float) -> float:
    """计算边缘链像素到拟合直线的垂直 RMS 距离。

    匹配 Rust bev_edlines.rs fit_rms_error()。
    """
    sin_a = np.sin(angle)
    cos_a = np.cos(angle)
    sum_sq = 0.0
    for x, y in region:
        dx = x - cx
        dy = y - cy
        perp = -(dx * sin_a) + (dy * cos_a)
        sum_sq += perp * perp
    return np.sqrt(sum_sq / len(region))


def classify_wall_points(cloud: np.ndarray, total_wall: int,
                         cxp: float, cyp: float, length: float, angle: float,
                         resolution: float, max_range: float,
                         distance: float, min_wall_pts: int,
                         min_z_span: float, min_extent: float):
    """墙体点分类与几何验证。

    匹配 Rust common.rs classify_wall_points()。
    返回 (inlier_indices_in_remaining, plane_eq) 或 None。
    """
    half = length / 2.0
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    px1 = cxp - half * cos_a
    py1 = cyp - half * sin_a
    px2 = cxp + half * cos_a
    py2 = cyp + half * sin_a

    x1 = px1 * resolution - max_range
    y1 = py1 * resolution - max_range
    x2 = px2 * resolution - max_range
    y2 = py2 * resolution - max_range

    dx = x2 - x1
    dy = y2 - y1
    len_m = np.sqrt(dx * dx + dy * dy)
    if len_m < 1e-6:
        return None

    rnx = -dy / len_m
    rny = dx / len_m
    rd = -(rnx * x1 + rny * y1)

    remaining = cloud[total_wall:]
    inlier_rel = []
    z_min = float('inf')
    z_max = float('-inf')

    for i, p in enumerate(remaining):
        dist = abs(rnx * p[0] + rny * p[1] + rd)
        if dist < distance:
            inlier_rel.append(i)
            if p[2] < z_min:
                z_min = p[2]
            if p[2] > z_max:
                z_max = p[2]

    if len(inlier_rel) < min_wall_pts:
        return None
    if z_max - z_min < min_z_span:
        return None

    line_dir_x = -rny
    line_dir_y = rnx
    t_min = float('inf')
    t_max = float('-inf')
    for rel_idx in inlier_rel:
        t = remaining[rel_idx][0] * line_dir_x + remaining[rel_idx][1] * line_dir_y
        if t < t_min:
            t_min = t
        if t > t_max:
            t_max = t
    if t_max - t_min < min_extent:
        return None

    return inlier_rel, (rnx, rny, 0.0, rd)


def detect_lines_edlines_full(
    img: np.ndarray,
    cloud_3d: Optional[np.ndarray] = None,
    resolution: float = 0.05,
    max_range: float = 10.0,
    distance: float = 0.10,
    min_wall_pts: int = 30,
    max_walls: int = 8,
    min_z_span: float = 1.0,
    min_extent: float = 0.7,
    grad_threshold: float = 0.05,
    anchor_threshold: float = 0.0,
    min_chain_len: int = 15,
    max_curvature_error: float = 2.0,
    min_length_ratio: float = 2.5,
    gaussian_sigma: float = 0.0,
    max_fit_error: float = 0.0,
):
    """完整 BevEDLines 管线（移植自 Rust）。

    返回:
        line_segments: list of (cx, cy, length, width, angle) 像素坐标
        wall_info: None 或 (line_segments_3d, wall_indices)，仅当 cloud_3d 提供时
    """
    h, w = img.shape[:2]
    size = w  # 方形 BEV 图

    # ── 准备一维数组 ──
    src = img.flatten()

    # ── 可选高斯模糊 ──
    if gaussian_sigma > 0:
        src = gaussian_blur(src, size, size, gaussian_sigma)

    # ── Sobel 梯度 ──
    grad_mag, grad_dir = sobel_gradient(src, size, size)

    max_mag_val = grad_mag.max()
    if max_mag_val < 1e-6:
        return [], None

    mag_threshold = max_mag_val * grad_threshold
    anchor_mag_threshold = max_mag_val * anchor_threshold

    # ── NMS 锚点检测 ──
    is_anchor = np.zeros(size * size, dtype=bool)
    for y in range(2, size - 2):
        for x in range(2, size - 2):
            i = y * size + x
            if grad_mag[i] < mag_threshold:
                continue
            if grad_dir[i] == EDGE_VERTICAL:
                if (grad_mag[i] >= grad_mag[i - 1] + anchor_mag_threshold and
                    grad_mag[i] >= grad_mag[i + 1] + anchor_mag_threshold):
                    is_anchor[i] = True
            elif grad_dir[i] == EDGE_HORIZONTAL:
                if (grad_mag[i] >= grad_mag[i - size] + anchor_mag_threshold and
                    grad_mag[i] >= grad_mag[i + size] + anchor_mag_threshold):
                    is_anchor[i] = True

    # ── 边缘绘制 ──
    edges = np.full(size * size, np.nan, dtype=np.float32)
    chains = []

    anchor_list = []
    for y in range(1, size - 1):
        for x in range(1, size - 1):
            if is_anchor[y * size + x]:
                anchor_list.append((x, y, grad_mag[y * size + x]))
    anchor_list.sort(key=lambda t: -t[2])

    for ax, ay, _ in anchor_list:
        if np.isfinite(edges[ay * size + ax]):
            continue

        d = grad_dir[ay * size + ax]
        if d == EDGE_VERTICAL:
            d1, d2 = UP, DOWN
        elif d == EDGE_HORIZONTAL:
            d1, d2 = LEFT, RIGHT
        else:
            continue

        chain = []
        walk_edge_chain(grad_mag, grad_dir, size, size, mag_threshold, ax, ay, d1, edges, chain)
        walk_edge_chain(grad_mag, grad_dir, size, size, mag_threshold, ax, ay, d2, edges, chain)

        if len(chain) < min_chain_len:
            for cx, cy in chain:
                edges[cy * size + cx] = np.nan
            continue

        chain_id = float(len(chains))
        for cx, cy in chain:
            edges[cy * size + cx] = chain_id
        chains.append(chain)

    # ── 线段拟合 + 曲率分裂 ──
    line_segments = []
    for chain in chains:
        sub_segs = split_chain_by_curvature(chain, max_curvature_error)
        for seg in sub_segs:
            if len(seg) < min_chain_len:
                continue
            cx, cy, length, width, angle = fit_rectangle(seg)
            if length < 3.0 or width < 0.5:
                continue
            if length / width < min_length_ratio:
                continue
            if max_fit_error > 0.0:
                rms = fit_rms_error(seg, cx, cy, angle)
                if rms > max_fit_error:
                    continue
            line_segments.append((cx, cy, length, width, angle))

    if not line_segments:
        return [], None

    line_segments.sort(key=lambda s: -s[2])  # 按长度降序

    # ── 3D 墙体点分类（对全点云独立检测每段）──
    wall_info = None
    if cloud_3d is not None:
        wall_planes = []
        all_wall_indices = set()
        for seg in line_segments[:max_walls * 2]:
            result = classify_wall_points(
                cloud_3d, 0,  # total_wall=0 → 对全量点云检测
                seg[0], seg[1], seg[2], seg[4],
                resolution, max_range,
                distance, min_wall_pts,
                min_z_span, min_extent,
            )
            if result is not None:
                inlier_rel, plane = result
                all_wall_indices.update(inlier_rel)
                wall_planes.append(plane)

        wall_info = {
            'indices': sorted(all_wall_indices),
            'planes': wall_planes,
            'total_wall': len(all_wall_indices),
        }

    line_segments_2d = [(s[0], s[1], s[2], s[3], s[4]) for s in line_segments]
    return line_segments_2d, wall_info


def draw_edlines_segments(img: np.ndarray, segments: list) -> np.ndarray:
    """在图上绘制 BevEDLines 检测到的线段。"""
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for cx, cy, length, width, angle in segments:
        half = length / 2.0
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        x1 = int(round(cx - half * cos_a))
        y1 = int(round(cy - half * sin_a))
        x2 = int(round(cx + half * cos_a))
        y2 = int(round(cy + half * sin_a))
        cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
    return vis


def detect_lines_lsd(img: np.ndarray) -> list:
    """LSD (Line Segment Detector) 直线检测 — OpenCV 4 推荐方法。"""
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    lsd = cv2.createLineSegmentDetector(0)
    lines, _, _, _ = lsd.detect(blur)
    if lines is None:
        return []
    # lines shape: (N, 1, 4) → (N, 4)
    lines = lines.squeeze(1)
    # 过滤短线
    lengths = np.sqrt((lines[:, 2] - lines[:, 0]) ** 2 + (lines[:, 3] - lines[:, 1]) ** 2)
    lines = lines[lengths > MIN_LINE_LENGTH]
    return lines

# ── 可视化 ──────────────────────────────────────────
def draw_lines(img: np.ndarray, lines: np.ndarray, color: tuple = (0, 255, 0)) -> np.ndarray:
    """在彩色图上绘制线段。"""
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for line in lines:
        line = line.flatten()
        x1, y1, x2, y2 = map(int, line[:4])
        cv2.line(vis, (x1, y1), (x2, y2), color, 1)
    return vis

# ── 主流程 ──────────────────────────────────────────
def main():
    print("=" * 50)
    print("BEV + OpenCV 墙体检测原型")
    print("=" * 50)

    # 1. 加载
    print("[1/4] 加载点云")
    cloud = load_pcd(str(PCD_PATH))

    # 2. 地面提取
    print("[2/4] 地面提取 (histogram, expand={})".format(GROUND_EXPAND))
    ground_mask, peak_z = extract_ground(cloud, GROUND_EXPAND, GROUND_THRESHOLD)
    nonground = cloud[~ground_mask]
    print(f"  非地面点: {len(nonground)}")

    # 3. BEV 投影
    print("[3/4] BEV 投影 (分辨率 {} m/px)".format(BEV_RESOLUTION))
    bev, ox, oy = points_to_bev(nonground, BEV_RESOLUTION, BEV_MAX_RANGE)

    # 4. 直线检测
    print("[4/4] 直线检测")
    lines_hough = detect_lines(bev)
    print(f"  HoughLinesP: {len(lines_hough)} 条线段")

    # LSD 检测
    lines_lsd = detect_lines_lsd(bev)
    print(f"  LSD: {len(lines_lsd)} 条线段")

    # EDLines 检测
    lines_edlines = detect_lines_edlines(bev)
    print(f"  EDLines: {len(lines_edlines)} 条线段")

    # BevEDLines 完整管线（匹配 Rust 实现）
    print("[4b/4] BevEDLines 完整管线")
    segments_full, wall_info = detect_lines_edlines_full(
        bev, cloud_3d=nonground,
        resolution=BEV_RESOLUTION, max_range=BEV_MAX_RANGE,
        distance=0.10, min_wall_pts=30, min_z_span=1.0, min_extent=0.7,
        grad_threshold=0.05, anchor_threshold=0.0,
        min_chain_len=15, max_curvature_error=2.0,
    )
    print(f"  BevEDLines 完整: {len(segments_full)} 条线段", end="")
    if wall_info:
        print(f", {wall_info['total_wall']} 个墙体点", end="")
    print()

    # 保存结果
    out_dir = Path("output/bev_test")
    out_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(out_dir / "bev_raw.png"), bev)

    vis_hough = draw_lines(bev, lines_hough, (0, 255, 0))
    cv2.imwrite(str(out_dir / "bev_hough.png"), vis_hough)

    vis_lsd = draw_lines(bev, lines_lsd, (0, 255, 255))
    cv2.imwrite(str(out_dir / "bev_lsd.png"), vis_lsd)

    vis_edlines = draw_lines(bev, lines_edlines, (255, 0, 255))
    cv2.imwrite(str(out_dir / "bev_edlines.png"), vis_edlines)

    vis_edlines_full = draw_edlines_segments(bev, segments_full)
    cv2.imwrite(str(out_dir / "bev_edlines_full.png"), vis_edlines_full)

    # 并排对比
    h, w = bev.shape
    gap = 8
    panels = [
        (bev, "BEV"),
        (vis_hough, "Hough"),
        (vis_lsd, "LSD"),
        (vis_edlines, "EDLines"),
        (vis_edlines_full, "EDLines_full"),
    ]
    n = len(panels)
    canvas = np.zeros((h, n * w + (n - 1) * gap, 3), dtype=np.uint8)
    for i, (panel, label) in enumerate(panels):
        x = i * (w + gap)
        if panel.ndim == 2:
            panel_color = cv2.cvtColor(panel, cv2.COLOR_GRAY2BGR)
        else:
            panel_color = panel
        canvas[:, x:x + w] = panel_color
        cv2.putText(canvas, label, (x + 10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imwrite(str(out_dir / "bev_comparison.png"), canvas)

    print(f"\n结果保存至: {out_dir}/")
    for f in ["bev_raw.png", "bev_hough.png", "bev_lsd.png", "bev_edlines.png",
              "bev_edlines_full.png", "bev_comparison.png"]:
        print(f"  {f}")


if __name__ == "__main__":
    main()
