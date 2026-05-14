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

# ── 参数 ──────────────────────────────────────────────
PCD_PATH = Path("data/cloud/lidar/000101.pcd")
GROUND_EXPAND = 0.10        # 地面 Z 扩展量（米）
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
def extract_ground(points: np.ndarray, expand: float = 0.10) -> tuple:
    """直方图地面提取，返回 (ground_mask, z_peak)。

    算法同 Rust histogram.rs：按 Z 排序 → 128 箱直方图 → 找峰值 → peak_z ± expand。
    """
    z = points[:, 2]
    z_min, z_max = float(z.min()), float(z.max())
    if z_max - z_min < 1e-6:
        return np.zeros(len(points), dtype=bool), 0.0

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
    mask = (z >= z_low) & (z <= z_high)
    print(f"  地面: peak_z={peak_z:.3f}, 范围 [{z_low:.3f}, {z_high:.3f}], "
          f"{mask.sum()} / {len(points)} 点")
    return mask, peak_z

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
    ground_mask, peak_z = extract_ground(cloud, GROUND_EXPAND)
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

    # 保存结果
    out_dir = Path("output/bev_test")
    out_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(out_dir / "bev_raw.png"), bev)

    vis_hough = draw_lines(bev, lines_hough, (0, 255, 0))
    cv2.imwrite(str(out_dir / "bev_hough.png"), vis_hough)

    vis_lsd = draw_lines(bev, lines_lsd, (0, 255, 255))
    cv2.imwrite(str(out_dir / "bev_lsd.png"), vis_lsd)

    # 并排对比
    h, w = bev.shape
    canvas = np.zeros((h, w * 3 + 20, 3), dtype=np.uint8)
    canvas[:, :w] = cv2.cvtColor(bev, cv2.COLOR_GRAY2BGR)
    canvas[:, w + 10:2 * w + 10] = vis_hough
    canvas[:, 2 * w + 20:] = vis_lsd
    cv2.line(canvas, (w + 5, 0), (w + 5, h), (128, 128, 128), 2)
    cv2.line(canvas, (2 * w + 15, 0), (2 * w + 15, h), (128, 128, 128), 2)
    labels = ["BEV 占用栅格", "HoughLinesP", "LSD"]
    for i, label in enumerate(labels):
        x = i * (w + 10) + 10
        cv2.putText(canvas, label, (x, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imwrite(str(out_dir / "bev_comparison.png"), canvas)

    print(f"\n结果保存至: {out_dir}/")
    print(f"  bev_raw.png — BEV 占用栅格")
    print(f"  bev_hough.png — HoughLinesP 检测结果")
    print(f"  bev_lsd.png — LSD 检测结果")
    print(f"  bev_comparison.png — 并排对比")


if __name__ == "__main__":
    main()
