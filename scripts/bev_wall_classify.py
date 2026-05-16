"""BEV 直线检测 → 邻域点分类 → 墙体提取原型。

流程：
  1. 地面提取（直方图法）
  2. BEV 投影（密度编码）
  3. HoughLinesP 直线检测
  4. 将直线投影回 3D，收集邻域点作为墙体点
  5. 与 RANSAC 结果对比验证
"""

import sys
import time
from pathlib import Path
import numpy as np
import cv2

# ── 参数 ──────────────────────────────────────────────
PCD_PATH = Path("data/cloud/lidar/000101.pcd")
GROUND_EXPAND = 0.10
BEV_RESOLUTION = 0.02       # 米/像素
BEV_MAX_RANGE = 10.0        # 米
WALL_DISTANCE = 0.10        # 点到直线的距离阈值（米）
MIN_Z_SPAN = 1.0            # 最小 Z 跨度（米）
MIN_WALL_PTS = 30           # 最少墙体点数
MIN_LINE_LENGTH = 30        # 最短线段（像素）
MIN_WALL_EXTENT = 0.7       # 沿墙面方向最小投影跨度（米）

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
    return (z >= peak_z - expand) & (z <= peak_z + expand)


# ── BEV 投影 ──────────────────────────────────────────
def points_to_bev(points: np.ndarray, resolution: float = 0.02,
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


# ── 直线检测（BEV 空间）─────────────────────────────────
def detect_wall_lines(bev: np.ndarray) -> np.ndarray:
    """返回 Nx4 线段阵列 [x1, y1, x2, y2]（像素坐标）。"""
    blur = cv2.GaussianBlur(bev, (3, 3), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    lines = cv2.HoughLinesP(binary, rho=1, theta=np.pi/360,
                            threshold=20, minLineLength=MIN_LINE_LENGTH,
                            maxLineGap=20)
    if lines is None:
        return np.empty((0, 4))
    lines = lines.squeeze(1)
    return lines


# ── 直线 → 3D 墙体点分类 ──────────────────────────────
def classify_wall_points(points: np.ndarray, lines_px: np.ndarray,
                         resolution: float, max_range: float,
                         distance: float = 0.10,
                         min_z_span: float = 1.0,
                         min_extent: float = 0.7,
                         min_pts: int = 30) -> tuple:
    """对每条 BEV 直线，收集附近 3D 点作为墙面。

    返回 (wall_mask, planes)，其中 planes 是 [nx, ny, 0, d] 列表。
    """
    if len(lines_px) == 0:
        return np.zeros(len(points), dtype=bool), []

    offset = max_range
    wall_mask = np.zeros(len(points), dtype=bool)
    planes = []
    unused = np.ones(len(points), dtype=bool)

    for line_px in lines_px:
        # 像素 → 米坐标
        x1_px, y1_px, x2_px, y2_px = line_px
        x1 = x1_px * resolution - offset
        y1 = y1_px * resolution - offset
        x2 = x2_px * resolution - offset
        y2 = y2_px * resolution - offset

        # 直线参数 (nx, ny, d) 使得 |nx*x + ny*y + d| = 点到直线距离
        dx = x2 - x1
        dy = y2 - y1
        length = np.sqrt(dx * dx + dy * dy)
        if length < 1e-6:
            continue
        nx = -dy / length
        ny = dx / length
        d = -(nx * x1 + ny * y1)

        # 提取未分类点到该直线的距离
        pts = points[unused]
        if len(pts) == 0:
            break
        dists = np.abs(pts[:, 0] * nx + pts[:, 1] * ny + d)

        # 距离阈值筛选
        near = dists < distance
        if near.sum() < min_pts:
            continue

        # Z 跨度验证
        z_vals = pts[near, 2]
        z_span = z_vals.max() - z_vals.min()
        if z_span < min_z_span:
            continue

        # 沿墙面方向投影跨度（min_extent 过滤）
        line_dir_x = -ny
        line_dir_y = nx
        t_vals = pts[near, 0] * line_dir_x + pts[near, 1] * line_dir_y
        t_span = t_vals.max() - t_vals.min()
        if t_span < min_extent:
            continue

        # TLS 精化墙面直线（协方差特征分解）
        inlier_xy = pts[near, :2]
        cx = inlier_xy[:, 0].mean()
        cy = inlier_xy[:, 1].mean()
        dxy = inlier_xy - np.array([cx, cy])
        cov = dxy.T @ dxy / len(inlier_xy)
        trace = cov[0, 0] + cov[1, 1]
        det = cov[0, 0] * cov[1, 1] - cov[0, 1] * cov[1, 0]
        disc = max(trace * trace - 4 * det, 0)
        lambda_min = (trace - np.sqrt(disc)) / 2

        refined_nx = cov[0, 1]
        refined_ny = lambda_min - cov[0, 0]
        refined_len = np.sqrt(refined_nx**2 + refined_ny**2)
        if refined_len > 1e-8:
            refined_nx /= refined_len
            refined_ny /= refined_len
        else:
            refined_nx, refined_ny = nx, ny
        refined_d = -(refined_nx * cx + refined_ny * cy)

        # 标记墙体点（在原始点云中的索引）
        # 注意：unused 中的 near 对应的是原始点云中的索引
        orig_indices = np.where(unused)[0][near]
        wall_mask[orig_indices] = True
        unused[orig_indices] = False
        planes.append([refined_nx, refined_ny, 0.0, refined_d])

        print(f"  墙体 {len(planes)}: pts={near.sum()}, "
              f"z_span={z_span:.2f}m, extent={t_span:.2f}m")

    return wall_mask, planes


# ── 主流程 ──────────────────────────────────────────
def main():
    print("=" * 60)
    print("BEV 直线检测 → 墙体点分类 原型")
    print("=" * 60)

    # 1. 加载
    print("[1/5] 加载点云")
    cloud = load_pcd(str(PCD_PATH))
    # 剔除原点
    cloud = cloud[~(np.all(np.abs(cloud) < 1e-6, axis=1))]
    print(f"  有效点: {len(cloud)}")

    # 2. 地面提取
    print("[2/5] 地面提取 (histogram, expand={})".format(GROUND_EXPAND))
    t0 = time.perf_counter()
    ground_mask = extract_ground(cloud, GROUND_EXPAND)
    nonground = cloud[~ground_mask]
    print(f"  非地面点: {len(nonground)} ({time.perf_counter()-t0:.1f}s)")

    # 3. BEV 投影
    print("[3/5] BEV 投影")
    t0 = time.perf_counter()
    bev = points_to_bev(nonground, BEV_RESOLUTION, BEV_MAX_RANGE)
    print(f"  BEV: {bev.shape[0]}x{bev.shape[1]} ({time.perf_counter()-t0:.1f}s)")

    # 4. 直线检测
    print("[4/5] 直线检测 (HoughLinesP)")
    t0 = time.perf_counter()
    lines = detect_wall_lines(bev)
    print(f"  检测到 {len(lines)} 条线段 ({time.perf_counter()-t0:.3f}s)")

    # 5. 直线 → 墙体点分类
    print("[5/5] 墙体点分类")
    t0 = time.perf_counter()
    wall_mask, planes = classify_wall_points(
        cloud, lines, BEV_RESOLUTION, BEV_MAX_RANGE,
        distance=WALL_DISTANCE, min_z_span=MIN_Z_SPAN,
        min_extent=MIN_WALL_EXTENT, min_pts=MIN_WALL_PTS,
    )
    t_elapsed = time.perf_counter() - t0
    n_wall = wall_mask.sum()

    print(f"\n  结果: {n_wall} / {len(cloud)} 墙体点 ({t_elapsed:.3f}s)")
    print(f"         {len(planes)} 面墙体")

    # 6. 可视化
    out_dir = Path("output/bev_test")
    out_dir.mkdir(parents=True, exist_ok=True)

    # BEV + 直线 + 墙体点标记
    vis = cv2.cvtColor(bev, cv2.COLOR_GRAY2BGR)
    for line in lines:
        x1, y1, x2, y2 = map(int, line[:4])
        cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 1)

    # 标记墙体点（在 BEV 上画红点）
    wall_xy = cloud[wall_mask, :2]
    offset = BEV_MAX_RANGE
    if len(wall_xy) > 0:
        wx = ((wall_xy[:, 0] + offset) / BEV_RESOLUTION).astype(np.int32)
        wy = ((wall_xy[:, 1] + offset) / BEV_RESOLUTION).astype(np.int32)
        wx = np.clip(wx, 0, vis.shape[1] - 1)
        wy = np.clip(wy, 0, vis.shape[0] - 1)
        for i in range(0, len(wx), 3):  # 隔点采样，避免太密
            cv2.circle(vis, (wx[i], wy[i]), 1, (0, 0, 255), -1)

    # 非墙体非地面点（灰）
    non_wall_nonground = cloud[~ground_mask & ~wall_mask]
    if len(non_wall_nonground) > 0:
        nwx = ((non_wall_nonground[:, 0] + offset) / BEV_RESOLUTION).astype(np.int32)
        nwy = ((non_wall_nonground[:, 1] + offset) / BEV_RESOLUTION).astype(np.int32)
        nwx = np.clip(nwx, 0, vis.shape[1] - 1)
        nwy = np.clip(nwy, 0, vis.shape[0] - 1)
        for i in range(0, len(nwx), 5):
            cv2.circle(vis, (nwx[i], nwy[i]), 1, (100, 100, 100), -1)

    cv2.putText(vis, f"Wall pts: {n_wall}  Walls: {len(planes)}  Lines: {len(lines)}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imwrite(str(out_dir / "bev_classify_result.png"), vis)
    print(f"\n可视化结果: output/bev_test/bev_classify_result.png")


if __name__ == "__main__":
    main()
