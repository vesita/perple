//! 特征级关联（LV-DOT 风格）
//!
//! 用加权特征向量替代纯马氏距离 + 尺寸惩罚的关联代价：
//!
//! | 特征 | 权重 | 说明 |
//! |------|------|------|
//! | 位置 | 1.0 (马氏距离) | KF 协方差加权的 (x,y) 距离 |
//! | 尺寸 | 0.1 | 归一化 (l,w,h) L1 距离 |
//! | 质心偏移 | 0.5 | 点云质心与 box 中心偏移的一致性 |
//! | 语义 | 3.0 | 类名相同=0, person/非person=1, 其他=0.5 |
//! | 速度余弦 | 0.5 | 轨迹预测速度与位移方向的夹角 |

/// LV-DOT 启发特征权重（乘到各项距离上）
pub const W_SIZE: f64 = 0.1;
pub const W_CENTROID: f64 = 0.5;
pub const W_SEMANTIC: f64 = 3.0;
pub const W_VELOCITY: f64 = 0.5;
pub const W_OVERLAP: f64 = 2.0; // BEV 重叠比权重

/// 语义距离：同类 0，person↔非person 1，其他异类 0.5
pub fn semantic_distance(a: &str, b: &str) -> f64 {
    if a == b {
        0.0
    } else if a == "person" || b == "person" {
        1.0
    } else {
        0.5
    }
}

/// BEV 重叠比：intersection_area / min(area_a, area_b)
///
/// 将两个 3D box 的 8 顶点投影到 x-y 平面，计算 AABB 重叠。
/// 返回 [0, 1]，1 = 一个完全包含另一个的投影。
pub fn bev_overlap_ratio(verts_a: &[nalgebra::Point3<f32>; 8], verts_b: &[nalgebra::Point3<f32>; 8]) -> f64 {
    fn aabb_xy(verts: &[nalgebra::Point3<f32>; 8]) -> (f64, f64, f64, f64) {
        let (mut x1, mut x2) = (verts[0].x as f64, verts[0].x as f64);
        let (mut y1, mut y2) = (verts[0].y as f64, verts[0].y as f64);
        for v in verts {
            let vx = v.x as f64;
            let vy = v.y as f64;
            if vx < x1 { x1 = vx; }
            if vx > x2 { x2 = vx; }
            if vy < y1 { y1 = vy; }
            if vy > y2 { y2 = vy; }
        }
        (x1, x2, y1, y2)
    }

    let (ax1, ax2, ay1, ay2) = aabb_xy(verts_a);
    let (bx1, bx2, by1, by2) = aabb_xy(verts_b);

    let ix1 = ax1.max(bx1);
    let ix2 = ax2.min(bx2);
    let iy1 = ay1.max(by1);
    let iy2 = ay2.min(by2);

    if ix1 >= ix2 || iy1 >= iy2 {
        return 0.0;
    }

    let inter = (ix2 - ix1) * (iy2 - iy1);
    let area_a = (ax2 - ax1) * (ay2 - ay1);
    let area_b = (bx2 - bx1) * (by2 - by1);
    let min_area = area_a.min(area_b);
    if min_area <= 0.0 {
        return 0.0;
    }
    inter / min_area
}

/// 计算特征级关联代价
///
/// # Parameters
///
/// * `mahalanobis` — KF 马氏距离（xy 位置，协方差加权）
/// * `track_size` — 轨迹预测尺寸 [l, w, h]（米）
/// * `det_size`   — 检测尺寸 [l, w, h]（米）
/// * `track_centroid` — 轨迹最后点云质心 (x, y)
/// * `det_centroid`   — 检测点云质心 (x, y)
/// * `track_pos` — KF 预测位置 (x, y)
/// * `det_pos`   — 检测 box 中心 (x, y)
/// * `track_class` — 轨迹类别标签
/// * `det_class`   — 检测类别标签
/// * `track_vel` — KF 预测速度 (vx, vy)
/// * `overlap_ratio` — BEV 投影重叠比 [0, 1]
///
/// # Returns
///
/// 加权总代价，越小表示越可能是同一目标。
/// 门控由调用方用 `mahalanobis` 独立做 χ² 检验。
pub fn feature_association_cost(
    mahalanobis: f64,
    track_size: [f64; 3],
    det_size: [f64; 3],
    track_centroid: [f64; 2],
    det_centroid: [f64; 2],
    track_pos: [f64; 2],
    det_pos: [f64; 2],
    track_class: &str,
    det_class: &str,
    track_vel: [f64; 2],
    overlap_ratio: f64,
) -> f64 {
    // ── 1. 尺寸距离（归一化 L1） ──────────────────────────────────────────
    let size_dist = {
        let denom = [
            track_size[0].abs() + det_size[0].abs(),
            track_size[1].abs() + det_size[1].abs(),
            track_size[2].abs() + det_size[2].abs(),
        ];
        let d0 = (track_size[0] - det_size[0]).abs() / denom[0].max(1e-6);
        let d1 = (track_size[1] - det_size[1]).abs() / denom[1].max(1e-6);
        let d2 = (track_size[2] - det_size[2]).abs() / denom[2].max(1e-6);
        (d0 + d1 + d2) / 3.0
    };

    // ── 2. 质心偏移一致性 ──────────────────────────────────────────────────
    // 点云质心相对于 box 中心的偏移 → 编码形状信息
    let off_track = [track_centroid[0] - track_pos[0], track_centroid[1] - track_pos[1]];
    let off_det   = [det_centroid[0] - det_pos[0],     det_centroid[1] - det_pos[1]];
    let centroid_dist = ((off_track[0] - off_det[0]).powi(2)
                       + (off_track[1] - off_det[1]).powi(2))
        .sqrt();

    // ── 3. 语义距离 ────────────────────────────────────────────────────────
    let sem_dist = semantic_distance(track_class, det_class);

    // ── 4. 速度余弦距离 ────────────────────────────────────────────────────
    // 轨迹速度方向 vs. 预测位置→检测位置的位移方向
    let vel_cos_dist = {
        let dx = det_pos[0] - track_pos[0];
        let dy = det_pos[1] - track_pos[1];
        let disp_mag = (dx * dx + dy * dy).sqrt();
        let vel_mag = (track_vel[0].powi(2) + track_vel[1].powi(2)).sqrt();
        if disp_mag > 0.05 && vel_mag > 0.05 {
            let dot = track_vel[0] * dx + track_vel[1] * dy;
            let cos_sim = (dot / (vel_mag * disp_mag)).clamp(-1.0, 1.0);
            (1.0 - cos_sim) * 0.5 // → [0, 1]
        } else {
            0.0
        }
    };

    // ── 加权和 ─────────────────────────────────────────────────────────────
    mahalanobis
        + W_SIZE * size_dist
        + W_CENTROID * centroid_dist
        + W_SEMANTIC * sem_dist
        + W_VELOCITY * vel_cos_dist
        + W_OVERLAP * (1.0 - overlap_ratio)
}
