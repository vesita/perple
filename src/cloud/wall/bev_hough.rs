/// BEV 鸟瞰图 + Hough 直线检测墙体提取（bev_hough）。
///
/// # 设计思路
///
/// 将非地面点云投影到 XY 平面生成 BEV 占用栅格图像，在图像域做 Hough 直线检测，
/// 再将 2D 直线映射回 3D 空间，收集邻域点作为墙体点。
///
/// # 流程
///
/// 1. **BEV 栅格化**：XY 投影 → 密度编码 → log 归一化
/// 2. **图像预处理**：高斯模糊 → Otsu 二值化 → 形态学闭运算
/// 3. **密集 Hough 变换**：所有占用栅格（非仅边缘）对 (θ, ρ) 累加器投票
///    — 全像素投票可捕捉稀疏墙体区域的微弱信号
/// 4. **峰值检测**：绝对阈值 + 相对阈值取较小值，适应不同密度场景
/// 5. **墙体点分类**：对每条候选直线，收集距离 < `distance` 的 3D 点
///
/// 聚类策略：hough（Hough 变换）— 图像域全局直线检测
/// 几何检测：l2（2D 线拟合）— 点到直线距离分类
/// 空间索引：bev（鸟瞰图栅格）— XY 平面栅格化
use super::WallPickStrategy;

pub struct BevHough {
    /// BEV 分辨率（米/像素），默认 0.02
    resolution: f32,
    /// BEV 范围（米），默认 10.0，即绘制 [-max_range, max_range] 范围
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
    /// Hough 累加器阈值比例 [0, 1]，值越小越敏感，默认 0.10
    hough_threshold: f32,
}

impl BevHough {
    pub fn new() -> Self {
        Self {
            resolution: 0.05,
            max_range: 10.0,
            distance: 0.10,
            min_wall_pts: 30,
            max_walls: 8,
            min_z_span: 1.0,
            min_extent: 0.7,
            hough_threshold: 0.10,
        }
    }

    pub fn with_params(distance: f32, min_wall_pts: usize) -> Self {
        Self { distance, min_wall_pts, ..Self::new() }
    }

    pub fn with_min_extent(mut self, extent: f32) -> Self {
        self.min_extent = extent;
        self
    }

    pub fn with_hough_threshold(mut self, t: f32) -> Self {
        self.hough_threshold = t;
        self
    }
}

impl WallPickStrategy for BevHough {
    fn strategy_name(&self) -> &'static str { "bev_hough" }

    fn pick(&mut self, cloud: &mut [[f32; 3]]) -> (usize, Vec<[f32; 4]>) {
        let n = cloud.len();
        if n < self.min_wall_pts { return (0, Vec::new()); }

        let size = (2.0 * self.max_range / self.resolution) as usize; // e.g., 1000

        // 1. BEV 密度编码
        let mut bev = vec![0u32; size * size];
        for p in cloud.iter() {
            if p[0].abs() >= self.max_range || p[1].abs() >= self.max_range { continue; }
            let x = ((p[0] + self.max_range) / self.resolution) as isize;
            let y = ((p[1] + self.max_range) / self.resolution) as isize;
            if x >= 0 && (x as usize) < size && y >= 0 && (y as usize) < size {
                bev[y as usize * size + x as usize] += 1;
            }
        }

        // log1p 归一化到 [0, 255]
        let mut img_f32 = vec![0.0f32; size * size];
        let mut max_val = 0.0f32;
        for i in 0..bev.len() {
            let l = (bev[i] as f32 + 1.0).ln();
            img_f32[i] = l;
            if l > max_val { max_val = l; }
        }

        let mut img_u8 = vec![0u8; size * size];
        if max_val > 1e-6 {
            let scale = 255.0 / max_val;
            for i in 0..img_f32.len() {
                img_u8[i] = (img_f32[i] * scale) as u8;
            }
        }

        // 2. 图像预处理：高斯模糊 + Otsu 二值化 + 形态学闭运算
        let blur_radius = 1; // 3x3
        let blurred = gaussian_blur(&img_u8, size, size, blur_radius);
        let threshold = otsu_threshold(&blurred);
        let mut binary = vec![0u8; size * size];
        for i in 0..blurred.len() {
            binary[i] = if blurred[i] > threshold { 255u8 } else { 0u8 };
        }
        // 形态学闭运算（3x3 矩形核）
        morph_close(&mut binary, size, size, 3);

        // 3. 收集投票像素：所有二值化后占用的栅格（非仅边缘）
        //    — 相比边缘检测，能捕捉稀疏墙体区域的微弱信号
        let vote_cells: Vec<(usize, usize)> = {
            let mut cells = Vec::new();
            for y in 0..size {
                for x in 0..size {
                    if binary[y * size + x] > 0 {
                        cells.push((x, y));
                    }
                }
            }
            cells
        };

        if vote_cells.is_empty() { return (0, Vec::new()); }

        log::debug!("bev_hough: BEV={}x{} occupied={}", size, size, vote_cells.len());

        // 4. Hough 变换
        let theta_steps = 180; // 1°/步
        let rho_max = ((size as f64) * 1.415).ceil() as usize; // ~1414
        let rho_bins = 2 * rho_max + 1;
        let mut accumulator = vec![0u32; theta_steps * rho_bins];

        let sin_table: Vec<f64> = (0..theta_steps)
            .map(|t| ((t as f64) * std::f64::consts::PI / 180.0).sin())
            .collect();
        let cos_table: Vec<f64> = (0..theta_steps)
            .map(|t| ((t as f64) * std::f64::consts::PI / 180.0).cos())
            .collect();

        for &(vx, vy) in &vote_cells {
            for t in 0..theta_steps {
                let r = (vx as f64) * cos_table[t] + (vy as f64) * sin_table[t];
                let ri = (r.round() as isize) + rho_max as isize;
                if ri >= 0 && (ri as usize) < rho_bins {
                    accumulator[t * rho_bins + ri as usize] += 1;
                }
            }
        }

        // 5. 峰值检测：绝对阈值 + 相对阈值取较小值
        let max_votes = accumulator.iter().max().copied().unwrap_or(0);
        let abs_min_votes = (10usize.max(vote_cells.len() / 200)) as u32;
        let rel_threshold = (max_votes as f32 * self.hough_threshold) as u32;
        let threshold_votes = rel_threshold.max(abs_min_votes);
        if threshold_votes < 2 { return (0, Vec::new()); }

        // 非极大值抑制（3x3 邻域）
        let mut peaks: Vec<(usize, usize, u32)> = Vec::new();
        let nbr_radius = 2;

        for t in 0..theta_steps {
            for ri in 0..rho_bins {
                let v = accumulator[t * rho_bins + ri];
                if v < threshold_votes { continue; }

                let mut is_peak = true;
                for dt in -(nbr_radius as isize)..=nbr_radius as isize {
                    for dr in -(nbr_radius as isize)..=nbr_radius as isize {
                        if dt == 0 && dr == 0 { continue; }
                        let nt = (t as isize + dt).rem_euclid(theta_steps as isize) as usize;
                        let nr = ri as isize + dr;
                        if nr < 0 || nr >= rho_bins as isize { continue; }
                        if accumulator[nt * rho_bins + nr as usize] > v {
                            is_peak = false;
                        }
                    }
                }
                if is_peak {
                    peaks.push((t, ri, v));
                }
            }
        }

        // 按投票数降序排序
        peaks.sort_by(|a, b| b.2.cmp(&a.2));

        // 角度去重：如果两个峰的角度差 < 10°，保留投票数更高的
        let mut deduped: Vec<(usize, usize, u32)> = Vec::new();
        for &p in &peaks {
            let ang_deg = p.0 as f64 * 1.0; // 1°/step
            let is_dup = deduped.iter().any(|d| {
                let d_ang = d.0 as f64 * 1.0;
                let diff = (ang_deg - d_ang).abs().min(180.0 - (ang_deg - d_ang).abs());
                diff < 15.0
            });
            if !is_dup {
                deduped.push(p);
            }
        }

        log::debug!("bev_hough: edge={} peaks={} deduped={}",
            vote_cells.len(), peaks.len(), deduped.len());

        // 6. 直线 → 墙体点分类
        let mut total_wall = 0usize;
        let mut planes = Vec::new();
        let wall_end = n;

        for &(t, ri, _votes) in deduped.iter().take(self.max_walls * 2) {
            if total_wall >= wall_end { break; }

            // BEV 像素坐标 → 射线参数：θ 是法线方向，ρ 是原点到直线距离
            let theta_rad = (t as f64) * std::f64::consts::PI / 180.0;
            let nx = theta_rad.cos() as f32;
            let ny = theta_rad.sin() as f32;
            let rho_px = ri as f32 - rho_max as f32; // BEV 像素坐标
            // ρ_px = nx * x_px + ny * y_px，其中 (x_px, y_px) 是 BEV 像素坐标
            // 需要两个点来确定 BEV 像素空间中的直线
            // 在 BEV 像素坐标系中：nx*x + ny*y = rho_px
            // 选两个基准点：x=0 → y=rho_px/ny，y=0 → x=rho_px/nx
            let (px1, py1, px2, py2) = if ny.abs() > nx.abs() {
                // 以 y=0 和 y=size 两个端点确定线段
                let y1_f = 0.0f32;
                let x1_f = if nx.abs() > 1e-6 { (rho_px - ny * y1_f) / nx } else { 0.0 };
                let y2_f = size as f32;
                let x2_f = if nx.abs() > 1e-6 { (rho_px - ny * y2_f) / nx } else { 0.0 };
                (x1_f, y1_f, x2_f, y2_f)
            } else {
                let x1_f = 0.0f32;
                let y1_f = if ny.abs() > 1e-6 { (rho_px - nx * x1_f) / ny } else { 0.0 };
                let x2_f = size as f32;
                let y2_f = if ny.abs() > 1e-6 { (rho_px - nx * x2_f) / ny } else { 0.0 };
                (x1_f, y1_f, x2_f, y2_f)
            };

            // 像素 → 米坐标
            let x1 = px1 * self.resolution - self.max_range;
            let y1 = py1 * self.resolution - self.max_range;
            let x2 = px2 * self.resolution - self.max_range;
            let y2 = py2 * self.resolution - self.max_range;

            let dx = x2 - x1;
            let dy = y2 - y1;
            let len = (dx * dx + dy * dy).sqrt();
            if len < 1e-6 { continue; }

            // 重新计算精确的法线方向
            let rnx = -dy / len;
            let rny = dx / len;
            let rd = -(rnx * x1 + rny * y1);

            // 收集剩余点中靠近此直线的点
            let remaining = &cloud[total_wall..wall_end];
            let mut inlier_rel: Vec<usize> = Vec::new();
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

            // 沿墙面方向投影跨度检查
            let line_dir_x = -rny;
            let line_dir_y = rnx;
            let (mut t_min, mut t_max) = (f32::MAX, f32::MIN);
            for &rel_idx in &inlier_rel {
                let t = remaining[rel_idx][0] * line_dir_x + remaining[rel_idx][1] * line_dir_y;
                if t < t_min { t_min = t; }
                if t > t_max { t_max = t; }
            }
            if t_max - t_min < self.min_extent { continue; }

            // 标记：将墙体点移到前部
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

// ─── 图像处理辅助函数 ──────────────────────────────────

/// 3x3 高斯模糊
fn gaussian_blur(src: &[u8], w: usize, h: usize, _radius: usize) -> Vec<u8> {
    let kernel: [f32; 9] = [
        1.0 / 16.0, 2.0 / 16.0, 1.0 / 16.0,
        2.0 / 16.0, 4.0 / 16.0, 2.0 / 16.0,
        1.0 / 16.0, 2.0 / 16.0, 1.0 / 16.0,
    ];
    let mut dst = vec![0u8; src.len()];
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let mut sum = 0.0f32;
            for ky in 0..3 {
                for kx in 0..3 {
                    let px = src[(y + ky - 1) * w + (x + kx - 1)];
                    sum += px as f32 * kernel[ky * 3 + kx];
                }
            }
            dst[y * w + x] = sum as u8;
        }
    }
    // 复制边界
    for y in 0..h {
        dst[y * w] = src[y * w];
        dst[y * w + w - 1] = src[y * w + w - 1];
    }
    for x in 0..w {
        dst[x] = src[x];
        dst[(h - 1) * w + x] = src[(h - 1) * w + x];
    }
    dst
}

/// Otsu 二值化阈值
fn otsu_threshold(src: &[u8]) -> u8 {
    let mut hist = [0u32; 256];
    for &v in src {
        hist[v as usize] += 1;
    }
    let total = src.len() as f64;
    let mut sum_all = 0.0f64;
    for i in 0..256 {
        sum_all += i as f64 * hist[i] as f64;
    }

    let mut sum_bg = 0.0f64;
    let mut w_bg = 0.0f64;
    let mut max_var = 0.0f64;
    let mut threshold = 0u8;

    for i in 0..256 {
        w_bg += hist[i] as f64;
        if w_bg < 1.0 { continue; }
        let w_fg = total - w_bg;
        if w_fg < 1.0 { break; }

        sum_bg += i as f64 * hist[i] as f64;
        let mean_bg = sum_bg / w_bg;
        let mean_fg = (sum_all - sum_bg) / w_fg;
        let var = w_bg * w_fg * (mean_bg - mean_fg).powi(2);
        if var > max_var {
            max_var = var;
            threshold = i as u8;
        }
    }
    threshold
}

/// 3x3 矩形核形态学闭运算（膨胀 → 腐蚀）
fn morph_close(binary: &mut [u8], w: usize, h: usize, _kernel_size: usize) {
    // 膨胀
    let mut dilated = vec![0u8; binary.len()];
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let mut max_val = 0u8;
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let px = binary[(y as isize + dy) as usize * w + (x as isize + dx) as usize];
                    if px > max_val { max_val = px; }
                }
            }
            dilated[y * w + x] = max_val;
        }
    }
    // 腐蚀
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let mut min_val = 255u8;
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let px = dilated[(y as isize + dy) as usize * w + (x as isize + dx) as usize];
                    if px < min_val { min_val = px; }
                }
            }
            binary[y * w + x] = min_val;
        }
    }
}



