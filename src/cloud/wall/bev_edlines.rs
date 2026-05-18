/// BEV 鸟瞰图 + EDLines 线段检测墙体检测（bev_edlines）。
///
/// # 设计思路
///
/// 实现 EDLines (Akinlar & Topal, 2011) 算法的核心步骤：
/// 1. BEV 栅格化 → 梯度计算（Sobel）
/// 2. **锚点检测**：沿梯度方向（垂直于边缘）的局部极大值点
/// 3. **边缘绘制（Edge Drawing）**：从锚点出发沿边缘方向链式追踪，连接成连续边缘链
/// 4. **线段拟合**：对每条边缘链做最小二乘拟合 + 曲率分裂
/// 5. 墙壁验证（同 bev_lsd）
///
/// # 与 bev_lsd 的区别
///
/// - bev_lsd：全图种子排序 + BFS 区域生长，对弱边缘更鲁棒
/// - bev_edlines：锚点检测 + 链式边缘追踪，速度更快、边缘定位更精确
///
/// # 流程
///
/// 1. **BEV 栅格化**：XY 投影 → 密度编码 → log 归一化
/// 2. **梯度计算**：Sobel 3×3 → 幅值 + 梯度方向
/// 3. **锚点检测**：沿梯度方向找局部幅值极大值
/// 4. **边缘绘制**：从锚点双向沿边缘方向追踪，连接高梯度像素
/// 5. **线段拟合**：PCA 或最小二乘 → 曲率分裂
/// 6. **墙壁验证**：长宽比 + 3D 点 Z 跨度 + 沿墙跨度
/// 7. **墙体点分类**：对每条候选直线，收集距离 < `distance` 的 3D 点
///
/// 聚类策略：edline（EDLines 线段检测）— 图像域链式边缘追踪
/// 几何检测：l2（2D 线拟合）— 点到直线距离分类
/// 空间索引：bev（鸟瞰图栅格）— XY 平面栅格化
use super::common::{
    bev_encode, classify_wall_points, fit_rectangle, gaussian_blur,
    split_chain_by_curvature, walk_edge_chain,
    EDGE_HORIZONTAL, EDGE_VERTICAL, LEFT, RIGHT, UP, DOWN,
};
use super::WallPickStrategy;

pub struct BevEdLines {
    /// BEV 分辨率（米/像素）
    resolution: f32,
    /// BEV 范围（米），绘制 [-max_range, max_range]
    max_range: f32,
    /// 点到直线距离阈值（米）
    distance: f32,
    /// 每面墙最少点数
    min_wall_pts: usize,
    /// 最多墙面板数
    max_walls: usize,
    /// 最小 Z 跨度（米）
    min_z_span: f32,
    /// 沿墙面方向最小投影跨度（米）
    min_extent: f32,
    /// 梯度幅值阈值比例 [0, 1]，相对最大梯度，默认 0.05
    grad_threshold: f32,
    /// 锚点检测阈值比例（相对最大梯度），低于此不产生锚点。
    /// NMS 检查：`grad_mag[i] >= grad_mag[neighbor] + max_mag * anchor_threshold`
    anchor_threshold: f32,
    /// 边缘链最少像素数
    min_chain_len: usize,
    /// 线段拟合最大曲率误差（像素），超过则分裂线段
    max_curvature_error: f32,
    /// 矩形最小长宽比
    min_length_ratio: f32,
    /// BEV 图高斯模糊 σ（像素），0 = 不模糊，推荐 0.8~1.5
    gaussian_sigma: f32,
    /// 线段拟合最大 RMS 误差（像素），0 = 不校验，推荐 0.5~1.0
    max_fit_error: f32,
}

impl BevEdLines {
    pub fn new() -> Self {
        Self {
            resolution: 0.05,
            max_range: 10.0,
            distance: 0.10,
            min_wall_pts: 30,
            max_walls: 8,
            min_z_span: 1.0,
            min_extent: 0.7,
            grad_threshold: 0.05,
            anchor_threshold: 0.0,
            min_chain_len: 15,
            max_curvature_error: 2.0,
            min_length_ratio: 2.5,
            gaussian_sigma: 0.0,
            max_fit_error: 0.0,
        }
    }

    pub fn with_params(distance: f32, min_wall_pts: usize) -> Self {
        Self { distance, min_wall_pts, ..Self::new() }
    }

    pub fn with_min_extent(mut self, extent: f32) -> Self {
        self.min_extent = extent;
        self
    }

    pub fn with_grad_threshold(mut self, t: f32) -> Self {
        self.grad_threshold = t;
        self
    }

    pub fn with_anchor_threshold(mut self, t: f32) -> Self {
        self.anchor_threshold = t;
        self
    }

    pub fn with_chain_min_length(mut self, len: usize) -> Self {
        self.min_chain_len = len;
        self
    }

    pub fn with_gaussian_blur(mut self, sigma: f32) -> Self {
        self.gaussian_sigma = sigma;
        self
    }

    pub fn with_fit_error_threshold(mut self, err: f32) -> Self {
        self.max_fit_error = err;
        self
    }

    /// 启用所有优化：高斯模糊 0.8 + 锚点阈值 0.04 + 拟合误差校验 0.5
    pub fn with_optimizations(mut self) -> Self {
        self.gaussian_sigma = 0.8;
        self.anchor_threshold = 0.04;
        self.max_fit_error = 0.5;
        self
    }
}

impl WallPickStrategy for BevEdLines {
    fn strategy_name(&self) -> &'static str { "bev_edlines" }

    fn pick(&mut self, cloud: &mut [[f32; 3]]) -> (usize, Vec<[f32; 4]>) {
        let n = cloud.len();
        if n < self.min_wall_pts { return (0, Vec::new()); }

        let size = (2.0 * self.max_range / self.resolution) as usize;

        // ── 1. BEV 密度编码 ──
        let img = bev_encode(cloud, size, self.max_range, self.resolution);

        // ── 可选高斯模糊 ──
        let img = if self.gaussian_sigma > 0.0 {
            gaussian_blur(&img, size, size, self.gaussian_sigma)
        } else {
            img
        };

        // ── 2. Sobel 梯度 + 二进制方向（无三角函数）──
        let (grad_mag, grad_dir) = sobel_gradient(&img, size, size);

        let max_mag = grad_mag.iter().fold(0.0f32, |a, &b| a.max(b));
        if max_mag < 1e-6 { return (0, Vec::new()); }
        let mag_threshold = max_mag * self.grad_threshold;
        let anchor_mag_threshold = max_mag * self.anchor_threshold;

        // ── 3. 锚点检测（NMS + 二进制方向）──
        // 沿梯度方向检查是否为局部极大值
        // 当 anchor_threshold > 0 时增加阈值偏移，抑制弱边缘锚点
        let mut is_anchor = vec![false; size * size];
        for y in 2..size - 2 {
            for x in 2..size - 2 {
                let i = y * size + x;
                if grad_mag[i] < mag_threshold { continue; }

                match grad_dir[i] {
                    EDGE_VERTICAL => {
                        // |gx|>=|gy|: 梯度水平 → 检查左右邻域极大值
                        if grad_mag[i] >= grad_mag[i - 1] + anchor_mag_threshold
                            && grad_mag[i] >= grad_mag[i + 1] + anchor_mag_threshold
                        {
                            is_anchor[i] = true;
                        }
                    }
                    EDGE_HORIZONTAL => {
                        // |gy|>|gx|: 梯度垂直 → 检查上下邻域极大值
                        if grad_mag[i] >= grad_mag[i - size] + anchor_mag_threshold
                            && grad_mag[i] >= grad_mag[i + size] + anchor_mag_threshold
                        {
                            is_anchor[i] = true;
                        }
                    }
                    _ => {}
                }
            }
        }

        // ── 4. 边缘绘制（Edge Drawing）──
        let mut edges = vec![f32::NEG_INFINITY; size * size];
        let mut chains: Vec<Vec<(usize, usize)>> = Vec::new();

        // 所有锚点按梯度幅值降序
        let mut anchor_list: Vec<(usize, usize, f32)> = Vec::new();
        for y in 1..size - 1 {
            for x in 1..size - 1 {
                if is_anchor[y * size + x] {
                    anchor_list.push((x, y, grad_mag[y * size + x]));
                }
            }
        }
        anchor_list.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

        for &(ax, ay, _) in &anchor_list {
            if edges[ay * size + ax].is_finite() { continue; }

            let mut chain = Vec::new();
            // 从二进制方向确定边缘追踪方向
            // EDGE_VERTICAL(|gx|>=|gy|): 梯度水平 → 边缘垂直 → 上下追踪
            // EDGE_HORIZONTAL(|gy|>|gx|): 梯度垂直 → 边缘水平 → 左右追踪
            let (d1, d2) = match grad_dir[ay * size + ax] {
                EDGE_VERTICAL => (UP, DOWN),
                EDGE_HORIZONTAL => (LEFT, RIGHT),
                _ => continue,
            };

            walk_edge_chain(&grad_mag, &grad_dir, size, size, mag_threshold, ax, ay, d1, &mut edges, &mut chain);
            walk_edge_chain(&grad_mag, &grad_dir, size, size, mag_threshold, ax, ay, d2, &mut edges, &mut chain);

            if chain.len() < self.min_chain_len {
                for &(cx, cy) in &chain {
                    edges[cy * size + cx] = f32::NEG_INFINITY;
                }
                continue;
            }

            let chain_id = chains.len() as f32;
            for &(cx, cy) in &chain {
                edges[cy * size + cx] = chain_id;
            }
            chains.push(chain);
        }

        // ── 5. 线段拟合 + 曲率分裂 ──
        let mut line_segments: Vec<(f32, f32, f32, f32, f32)> = Vec::new();
        for chain in &chains {
            let sub_segments = split_chain_by_curvature(chain, self.max_curvature_error);
            for seg in &sub_segments {
                if seg.len() < self.min_chain_len { continue; }
                let (cx, cy, length, width, angle) = fit_rectangle(seg);
                if length < 3.0 || width < 0.5 { continue; }
                if length / width < self.min_length_ratio { continue; }
                // 可选拟合误差校验：过滤曲线段或锯齿链
                if self.max_fit_error > 0.0 {
                    let rms = fit_rms_error(seg, cx, cy, angle);
                    if rms > self.max_fit_error { continue; }
                }
                line_segments.push((cx, cy, length, width, angle));
            }
        }

        if line_segments.is_empty() { return (0, Vec::new()); }

        // 按长度降序
        line_segments.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

        log::debug!("bev_edlines: size={} anchors={} chains={} segments={}",
            size, anchor_list.len(), chains.len(), line_segments.len());

        // ── 6. 墙体点分类 ──
        let mut total_wall = 0usize;
        let mut planes = Vec::new();

        for &(cxp, cyp, length, _width, angle) in line_segments.iter().take(self.max_walls * 2) {
            if let Some(plane) = classify_wall_points(
                cloud, &mut total_wall,
                cxp, cyp, length, angle,
                self.resolution, self.max_range,
                self.distance, self.min_wall_pts,
                self.min_z_span, self.min_extent,
            ) {
                planes.push(plane);
            }
        }

        (total_wall, planes)
    }
}

/// 单次 Sobel 计算，返回梯度幅值（|gx|+|gy|）和二进制方向（避免 atan2/cos/sin）
fn sobel_gradient(src: &[u8], w: usize, h: usize) -> (Vec<f32>, Vec<u8>) {
    let n = src.len();
    let mut mag = vec![0.0f32; n];
    let mut dir = vec![0u8; n];
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let i = y * w + x;
            let gx = -1i32 * src[i - w - 1] as i32 + 1 * src[i - w + 1] as i32
                     -2 * src[i - 1] as i32     + 2 * src[i + 1] as i32
                     -1 * src[i + w - 1] as i32 + 1 * src[i + w + 1] as i32;
            let gy = -1i32 * src[i - w - 1] as i32 - 2 * src[i - w] as i32 - 1 * src[i - w + 1] as i32
                     +1 * src[i + w - 1] as i32 + 2 * src[i + w] as i32 + 1 * src[i + w + 1] as i32;

            let gx_abs = gx.abs() as f32;
            let gy_abs = gy.abs() as f32;
            mag[i] = gx_abs + gy_abs;

            if gx_abs >= gy_abs {
                dir[i] = EDGE_VERTICAL;
            } else {
                dir[i] = EDGE_HORIZONTAL;
            }
        }
    }
    (mag, dir)
}

/// 计算边缘链像素到拟合直线的垂直 RMS 距离（像素）
///
/// line 由 (cx, cy, angle) 定义，方向向量为 (cos_a, sin_a)。
/// 垂直距离 = |-(x-cx)*sin_a + (y-cy)*cos_a|
fn fit_rms_error(region: &[(usize, usize)], cx: f32, cy: f32, angle: f32) -> f32 {
    let sin_a = angle.sin();
    let cos_a = angle.cos();
    let mut sum_sq = 0.0f32;
    for &(x, y) in region {
        let dx = x as f32 - cx;
        let dy = y as f32 - cy;
        let perp = -(dx * sin_a) + (dy * cos_a);
        sum_sq += perp * perp;
    }
    (sum_sq / region.len() as f32).sqrt()
}
