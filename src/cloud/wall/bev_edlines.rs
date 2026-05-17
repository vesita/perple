/// BEV 鸟瞰图 + EDLines 线段检测墙体提取（bev_edlines）。
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
    /// 锚点检测阈值比例，低于此不产生锚点
    anchor_threshold: f32,
    /// 边缘链最少像素数
    min_chain_len: usize,
    /// 线段拟合最大曲率误差（像素），超过则分裂线段
    max_curvature_error: f32,
    /// 矩形最小长宽比
    min_length_ratio: f32,
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
            anchor_threshold: 0.08,
            min_chain_len: 15,
            max_curvature_error: 2.0,
            min_length_ratio: 2.5,
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
}

impl WallPickStrategy for BevEdLines {
    fn strategy_name(&self) -> &'static str { "bev_edlines" }

    fn pick(&mut self, cloud: &mut [[f32; 3]]) -> (usize, Vec<[f32; 4]>) {
        let n = cloud.len();
        if n < self.min_wall_pts { return (0, Vec::new()); }

        let size = (2.0 * self.max_range / self.resolution) as usize;

        // ── 1. BEV 密度编码 ──
        let mut bev = vec![0u32; size * size];
        for p in cloud.iter() {
            if p[0].abs() >= self.max_range || p[1].abs() >= self.max_range { continue; }
            let x = ((p[0] + self.max_range) / self.resolution) as isize;
            let y = ((p[1] + self.max_range) / self.resolution) as isize;
            if x >= 0 && (x as usize) < size && y >= 0 && (y as usize) < size {
                bev[y as usize * size + x as usize] += 1;
            }
        }

        let mut img_f32 = vec![0.0f32; size * size];
        let mut max_val = 0.0f32;
        for i in 0..bev.len() {
            let l = (bev[i] as f32 + 1.0).ln();
            img_f32[i] = l;
            if l > max_val { max_val = l; }
        }
        let mut img = vec![0u8; size * size];
        if max_val > 1e-6 {
            let scale = 255.0 / max_val;
            for i in 0..img_f32.len() {
                img[i] = (img_f32[i] * scale) as u8;
            }
        }

        // ── 2. Sobel 梯度 ──
        let grad_mag = sobel_magnitude(&img, size, size);
        let grad_angle = sobel_angle(&img, size, size);

        let max_mag = grad_mag.iter().fold(0.0f32, |a, &b| a.max(b));
        if max_mag < 1e-6 { return (0, Vec::new()); }
        let mag_threshold = max_mag * self.grad_threshold;
        let anchor_mag_threshold = max_mag * self.anchor_threshold;

        // ── 3. 锚点检测（ED 核心）──
        // 沿梯度方向检查是否局部极大值
        let mut is_anchor = vec![false; size * size];
        for y in 1..size - 1 {
            for x in 1..size - 1 {
                let i = y * size + x;
                if grad_mag[i] < anchor_mag_threshold { continue; }

                let angle = grad_angle[i];
                // 梯度方向单位向量（指向梯度幅值增大方向）
                let gx = angle.cos();
                let gy = angle.sin();

                // 取梯度方向的两个邻域像素（最近邻近似）
                let (nx1, ny1) = neighbor_step(gx, gy);
                let nx1_clamp = (x as i32 + nx1).clamp(0, size as i32 - 1) as usize;
                let ny1_clamp = (y as i32 + ny1).clamp(0, size as i32 - 1) as usize;
                // 反方向
                let nx2_clamp = (x as i32 - nx1).clamp(0, size as i32 - 1) as usize;
                let ny2_clamp = (y as i32 - ny1).clamp(0, size as i32 - 1) as usize;

                let mag = grad_mag[i];
                let mag1 = grad_mag[ny1_clamp * size + nx1_clamp];
                let mag2 = grad_mag[ny2_clamp * size + nx2_clamp];

                if mag >= mag1 && mag >= mag2 {
                    is_anchor[i] = true;
                }
            }
        }

        // ── 4. 边缘绘制（Edge Drawing）──
        // 从锚点出发双向追踪
        let mut edges = vec![f32::NEG_INFINITY; size * size]; // 存储链 ID（用 float 方便与其他逻辑互操作）
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
            if edges[ay * size + ax].is_finite() { continue; } // 已被其他链覆盖

            let mut chain = Vec::new();
            // 从锚点的双侧方向追踪，先走一个方向再走另一个方向
            // 边缘方向 = 梯度方向 + 90°
            let edge_angle = grad_angle[ay * size + ax] + std::f32::consts::PI / 2.0;
            let (ed_x, ed_y) = (edge_angle.cos(), edge_angle.sin());

            // 正向：沿 (ed_x, ed_y) 方向
            walk_edge_chain(
                &grad_mag, &grad_angle, size, size, mag_threshold,
                ax, ay, ed_x, ed_y, &mut edges, &mut chain
            );

            // 反向：沿 (-ed_x, -ed_y) 方向
            walk_edge_chain(
                &grad_mag, &grad_angle, size, size, mag_threshold,
                ax, ay, -ed_x, -ed_y, &mut edges, &mut chain
            );

            // 如果链太短则丢弃
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
        let wall_end = n;

        for &(cxp, cyp, length, _width, angle) in line_segments.iter().take(self.max_walls * 2) {
            if total_wall >= wall_end { break; }

            let half = length / 2.0;
            let cos_a = angle.cos();
            let sin_a = angle.sin();
            let px1 = cxp - half * cos_a;
            let py1 = cyp - half * sin_a;
            let px2 = cxp + half * cos_a;
            let py2 = cyp + half * sin_a;

            let x1 = px1 * self.resolution - self.max_range;
            let y1 = py1 * self.resolution - self.max_range;
            let x2 = px2 * self.resolution - self.max_range;
            let y2 = py2 * self.resolution - self.max_range;

            let dx = x2 - x1;
            let dy = y2 - y1;
            let len_m = (dx * dx + dy * dy).sqrt();
            if len_m < 1e-6 { continue; }

            let rnx = -dy / len_m;
            let rny = dx / len_m;
            let rd = -(rnx * x1 + rny * y1);

            let remaining = &cloud[total_wall..wall_end];
            let mut inlier_rel = Vec::new();
            let mut z_min = f32::MAX;
            let mut z_max = f32::MIN;

            for (i, p) in remaining.iter().enumerate() {
                let dist = (rnx * p[0] + rny * p[1] + rd).abs();
                if dist < self.distance {
                    inlier_rel.push(i);
                    if p[2] < z_min { z_min = p[2]; }
                    if p[2] > z_max { z_max = p[2]; }
                }
            }

            if inlier_rel.len() < self.min_wall_pts { continue; }
            if z_max - z_min < self.min_z_span { continue; }

            let line_dir_x = -rny;
            let line_dir_y = rnx;
            let (mut t_min, mut t_max) = (f32::MAX, f32::MIN);
            for &rel_idx in &inlier_rel {
                let t = remaining[rel_idx][0] * line_dir_x + remaining[rel_idx][1] * line_dir_y;
                if t < t_min { t_min = t; }
                if t > t_max { t_max = t; }
            }
            if t_max - t_min < self.min_extent { continue; }

            let mut write = total_wall;
            for &rel_idx in &inlier_rel {
                cloud.swap(total_wall + rel_idx, write);
                write += 1;
            }
            total_wall = write;
            planes.push([rnx, rny, 0.0, rd]);
        }

        (total_wall, planes)
    }
}

// ─── 边缘绘制（Edge Drawing）核心 ──────────────────────

/// 沿指定方向从起点追踪边缘链。
///
/// `walk_edge_chain` 实现 EDLines 的链式追踪：
/// 从起点 (sx, sy) 出发，沿边缘方向 (ed_x, ed_y) 前进，
/// 每一步从 3 个候选像素中选梯度幅值最高的作为下一步。
fn walk_edge_chain(
    grad_mag: &[f32],
    grad_angle: &[f32],
    w: usize,
    h: usize,
    mag_threshold: f32,
    sx: usize,
    sy: usize,
    ed_x: f32,
    ed_y: f32,
    edges: &mut [f32],
    chain: &mut Vec<(usize, usize)>,
) {
    // 先把锚点加入链（仅在首次调用时）
    if chain.is_empty() {
        chain.push((sx, sy));
        edges[sy * w + sx] = -1.0; // 临时标记
    }

    let mut x = sx as i32;
    let mut y = sy as i32;

    loop {
        // 边缘方向确定主前进方向
        let (step_x, step_y) = ed_step(ed_x, ed_y);

        // 3 个候选像素：正前方 + 侧偏
        let candidates = [
            (x + step_x, y + step_y),                     // 正前方
            (x + step_x + step_y, y + step_y - step_x),   // 右偏
            (x + step_x - step_y, y + step_y + step_x),   // 左偏
        ];

        // 过滤有效候选，选梯度幅值最高的
        let mut best_i = None;
        let mut best_mag = mag_threshold; // 必须超过阈值

        // 优先选择方向与当前边缘方向最一致且未使用的高梯度像素
        for (_idx, &(cx, cy)) in candidates.iter().enumerate() {
            if cx < 1 || cx >= w as i32 - 1 || cy < 1 || cy >= h as i32 - 1 { continue; }
            let ci = cy as usize * w + cx as usize;
            if edges[ci].is_finite() { continue; } // 已属于其它链

            let mag = grad_mag[ci];
            if mag > best_mag {
                // 检查角度一致性（方向差 < 90°）
                let angle_diff = edge_angle_diff(grad_angle[ci], grad_angle[y as usize * w + x as usize]);
                if angle_diff < std::f32::consts::PI / 2.0 {
                    best_mag = mag;
                    best_i = Some((cx, cy));
                }
            }
        }

        match best_i {
            Some((nx, ny)) => {
                chain.push((nx as usize, ny as usize));
                edges[ny as usize * w + nx as usize] = -1.0;
                x = nx;
                y = ny;
                // 更新边缘方向为当前点的梯度方向
                let new_edge_angle = grad_angle[y as usize * w + x as usize] + std::f32::consts::PI / 2.0;
                // 取与当前方向一致的朝向（不反向）
                let dot = new_edge_angle.cos() * ed_x + new_edge_angle.sin() * ed_y;
                if dot >= 0.0 {
                    // 保持原有 ed_x, ed_y，或者说保留原参考方向
                }
                // 继续
            }
            None => break, // 无法继续
        }
    }
}

/// 根据梯度方向向量确定主像素步进方向
fn ed_step(ed_x: f32, ed_y: f32) -> (i32, i32) {
    let abs_x = ed_x.abs();
    let abs_y = ed_y.abs();
    if abs_x >= abs_y {
        // 水平主导
        (if ed_x > 0.0 { 1 } else { -1 }, 0)
    } else {
        // 垂直主导
        (0, if ed_y > 0.0 { 1 } else { -1 })
    }
}

/// 沿梯度方向取邻域步进
fn neighbor_step(gx: f32, gy: f32) -> (i32, i32) {
    let abs_x = gx.abs();
    let abs_y = gy.abs();
    if abs_x >= abs_y {
        (if gx > 0.0 { 1 } else { -1 }, 0)
    } else {
        (0, if gy > 0.0 { 1 } else { -1 })
    }
}

/// 两条边缘方向之间的夹角（忽略朝向）
fn edge_angle_diff(a: f32, b: f32) -> f32 {
    // 梯度角度模 π
    let diff = ((a + std::f32::consts::PI / 2.0) - (b + std::f32::consts::PI / 2.0)).abs() % std::f32::consts::PI;
    diff.min(std::f32::consts::PI - diff)
}

// ─── 曲率分裂 ──────────────────────────────────────────

/// 对边缘链按曲率分裂为多个近似直线的线段。
///
/// 算法：递归分裂
/// 1. 拟合链端点到直线
/// 2. 找距直线最远的点
/// 3. 若距离 > max_error，在改点处分裂为两段，递归处理
fn split_chain_by_curvature(chain: &[(usize, usize)], max_error: f32) -> Vec<Vec<(usize, usize)>> {
    if chain.len() < 4 {
        return vec![chain.to_vec()];
    }

    let mut segments = Vec::new();
    split_recursive(chain, 0, chain.len() - 1, max_error, &mut segments);
    segments
}

fn split_recursive(
    chain: &[(usize, usize)],
    start: usize,
    end: usize,
    max_error: f32,
    segments: &mut Vec<Vec<(usize, usize)>>,
) {
    if end - start < 3 {
        segments.push(chain[start..=end].to_vec());
        return;
    }

    // 端点直线
    let (x1, y1) = (chain[start].0 as f32, chain[start].1 as f32);
    let (x2, y2) = (chain[end].0 as f32, chain[end].1 as f32);
    let dx = x2 - x1;
    let dy = y2 - y1;
    let len2 = dx * dx + dy * dy;
    if len2 < 1e-6 {
        segments.push(chain[start..=end].to_vec());
        return;
    }

    // 找距直线最远的点
    let mut max_dist = 0.0f32;
    let mut split_idx = start;
    for i in (start + 1)..end {
        let (px, py) = (chain[i].0 as f32, chain[i].1 as f32);
        let dist = ((py - y1) * dx - (px - x1) * dy).abs() / len2.sqrt();
        if dist > max_dist {
            max_dist = dist;
            split_idx = i;
        }
    }

    if max_dist > max_error && split_idx > start && split_idx < end {
        split_recursive(chain, start, split_idx, max_error, segments);
        split_recursive(chain, split_idx, end, max_error, segments);
    } else {
        segments.push(chain[start..=end].to_vec());
    }
}

// ─── 图像处理辅助 ──────────────────────────────────────

fn sobel_magnitude(src: &[u8], w: usize, h: usize) -> Vec<f32> {
    let mut mag = vec![0.0f32; src.len()];
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let i = y * w + x;
            let gx = -1.0 * src[i - w - 1] as f32 + 1.0 * src[i - w + 1] as f32
                     -2.0 * src[i - 1] as f32     + 2.0 * src[i + 1] as f32
                     -1.0 * src[i + w - 1] as f32 + 1.0 * src[i + w + 1] as f32;
            let gy = -1.0 * src[i - w - 1] as f32 - 2.0 * src[i - w] as f32 - 1.0 * src[i - w + 1] as f32
                     +1.0 * src[i + w - 1] as f32 + 2.0 * src[i + w] as f32 + 1.0 * src[i + w + 1] as f32;
            mag[i] = (gx * gx + gy * gy).sqrt();
        }
    }
    mag
}

fn sobel_angle(src: &[u8], w: usize, h: usize) -> Vec<f32> {
    let mut angle = vec![0.0f32; src.len()];
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let i = y * w + x;
            let gx = -1.0 * src[i - w - 1] as f32 + 1.0 * src[i - w + 1] as f32
                     -2.0 * src[i - 1] as f32     + 2.0 * src[i + 1] as f32
                     -1.0 * src[i + w - 1] as f32 + 1.0 * src[i + w + 1] as f32;
            let gy = -1.0 * src[i - w - 1] as f32 - 2.0 * src[i - w] as f32 - 1.0 * src[i - w + 1] as f32
                     +1.0 * src[i + w - 1] as f32 + 2.0 * src[i + w] as f32 + 1.0 * src[i + w + 1] as f32;
            let a = gy.atan2(gx);
            angle[i] = if a < 0.0 { a + std::f32::consts::PI } else { a };
        }
    }
    angle
}

/// PCA 矩形拟合（同 bev_lsd）
fn fit_rectangle(region: &[(usize, usize)]) -> (f32, f32, f32, f32, f32) {
    let n = region.len() as f32;
    let mut cx = 0.0f32;
    let mut cy = 0.0f32;
    for &(x, y) in region {
        cx += x as f32;
        cy += y as f32;
    }
    cx /= n;
    cy /= n;

    let mut xx = 0.0f32;
    let mut xy = 0.0f32;
    let mut yy = 0.0f32;
    for &(x, y) in region {
        let dx = x as f32 - cx;
        let dy = y as f32 - cy;
        xx += dx * dx;
        xy += dx * dy;
        yy += dy * dy;
    }

    let angle = if xy.abs() > 1e-6 {
        let trace = xx + yy;
        let det = xx * yy - xy * xy;
        let sqrt_term = ((trace * trace / 4.0 - det).max(0.0)).sqrt();
        let lambda1 = trace / 2.0 + sqrt_term;
        (lambda1 - xx).atan2(xy)
    } else {
        0.0
    };

    let cos_a = angle.cos();
    let sin_a = angle.sin();
    let (mut min_proj, mut max_proj) = (f32::MAX, f32::MIN);
    let (mut min_perp, mut max_perp) = (f32::MAX, f32::MIN);
    for &(x, y) in region {
        let dx = x as f32 - cx;
        let dy = y as f32 - cy;
        let proj = dx * cos_a + dy * sin_a;
        let perp = -dx * sin_a + dy * cos_a;
        min_proj = min_proj.min(proj);
        max_proj = max_proj.max(proj);
        min_perp = min_perp.min(perp);
        max_perp = max_perp.max(perp);
    }

    (cx, cy, max_proj - min_proj, max_perp - min_perp, angle)
}
