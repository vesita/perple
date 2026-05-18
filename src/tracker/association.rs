use std::collections::BTreeMap;

use nalgebra::{Point3, Vector2};

use crate::{
    cloud::CldBud,
    tracker::{
        features,
        hungarian::hungarian,
        object::TrackedObject,
    },
    utils::boxes::Box3D,
};

/// 马氏距离门控阈值
///
/// χ²(3) 在 α=0.05 时阈值为 7.815
/// sqrt(7.815) ≈ 2.795 用于距离比较
const MAHALANOBIS_THRESHOLD: f64 = 2.796;

/// 马氏距离关联（匈牙利算法最优指派）
///
/// `cost_buf` / `sq_buf` 复用缓冲区，避免每帧堆分配。
pub(crate) fn associate(
    objects: &BTreeMap<usize, TrackedObject>,
    detections: &[CldBud],
    cost_buf: &mut Vec<Vec<f64>>,
    sq_buf: &mut Vec<Vec<f64>>,
) -> (Vec<(usize, usize)>, Vec<usize>) {
    let n_objects = objects.len();
    let n_detections = detections.len();

    if n_objects == 0 || n_detections == 0 {
        return (Vec::new(), (0..n_detections).collect());
    }

    let mut obj_ids: Vec<usize> = objects.keys().copied().collect();
    obj_ids.sort(); // BTreeMap 保证 key 有序，sort 仅做双重保险

    // 构建代价矩阵（复用缓冲区）
    cost_buf.clear();
    cost_buf.resize(n_objects, Vec::with_capacity(n_detections));
    for row in cost_buf.iter_mut() {
        row.clear();
        row.resize(n_detections, f64::MAX);
    }
    for (obj_idx, &obj_id) in obj_ids.iter().enumerate() {
        let obj = &objects[&obj_id];
        // 特征级关联：轨迹特征（用 predicted_box 而非 last_box）
        let track_pos = {
            let p = obj.kalman_filter.get_position();
            [p.x, p.y]
        };
        let track_vel = {
            let v = obj.kalman_filter.get_velocity();
            [v.x, v.y]
        };
        // 用 predicted_box 的尺寸（KF 平滑后）
        let track_size = obj.predicted_box.as_ref().map(|b| {
            [b.length as f64, b.width as f64, b.height as f64]
        }).or_else(|| obj.last_box.as_ref().map(|b| {
            [b.length as f64, b.width as f64, b.height as f64]
        }));
        let track_centroid = [obj.last_centroid[0] as f64, obj.last_centroid[1] as f64];
        let track_class = &obj.class_type;
        // 预计算轨迹的 BEV AABB 顶点（用于重叠比）
        let track_verts = obj.predicted_box.as_ref()
            .or(obj.last_box.as_ref())
            .map(|b| b.vertices());

        for (det_idx, det) in detections.iter().enumerate() {
            let center = det.the_box.center();
            let meas = Vector2::new(center.x as f64, center.y as f64);
            let dist = obj.kalman_filter.mahalanobis_distance(meas);
            if dist < MAHALANOBIS_THRESHOLD {
                let det_center = [center.x as f64, center.y as f64];
                let det_size = [
                    det.the_box.length as f64,
                    det.the_box.width as f64,
                    det.the_box.height as f64,
                ];
                let det_centroid = [det.centroid[0] as f64, det.centroid[1] as f64];

                // BEV 重叠比
                let overlap = track_verts.as_ref()
                    .and_then(|tv| {
                        let dv = det.the_box.vertices();
                        Some(features::bev_overlap_ratio(tv, &dv))
                    })
                    .unwrap_or(0.0);

                let feature_cost = if let Some(ref tsize) = track_size {
                    features::feature_association_cost(
                        dist,
                        *tsize,
                        det_size,
                        track_centroid,
                        det_centroid,
                        track_pos,
                        det_center,
                        track_class,
                        &det.class_name,
                        track_vel,
                        overlap,
                    )
                } else {
                    dist // 无历史时退化为纯马氏距离
                };
                cost_buf[obj_idx][det_idx] = feature_cost;
            }
        }
    }

    // 匈牙利最优指派（复用 sq_buf）
    let assignment = hungarian(cost_buf, sq_buf);

    // 提取匹配结果（返回实际 obj_id 而非索引，避免顺序不确定性）
    let mut used_det = vec![false; n_detections];
    let mut matches = Vec::new();

    for (obj_idx, &det_idx) in assignment.iter().enumerate() {
        if det_idx < n_detections && cost_buf[obj_idx][det_idx] < f64::MAX / 2.0 {
            matches.push((obj_ids[obj_idx], det_idx));
            used_det[det_idx] = true;
        }
    }

    let unmatched: Vec<usize> = (0..n_detections)
        .filter(|&i| !used_det[i])
        .collect();

    (matches, unmatched)
}

/// 提取包围盒内的点（AABB 快速过滤），最多取 `max_out` 个点
/// 超出的部分步长抽样，保证 O(N²) 投票可控
fn extract_points_in_box(points: &[[f32; 3]], box3d: &Box3D, max_out: usize) -> Vec<[f32; 3]> {
    // 先用 AABB 粗略过滤
    let verts = box3d.vertices();
    let (mut x_min, mut x_max) = (verts[0].x, verts[0].x);
    let (mut y_min, mut y_max) = (verts[0].y, verts[0].y);
    let (mut z_min, mut z_max) = (verts[0].z, verts[0].z);
    for v in &verts {
        x_min = x_min.min(v.x);
        x_max = x_max.max(v.x);
        y_min = y_min.min(v.y);
        y_max = y_max.max(v.y);
        z_min = z_min.min(v.z);
        z_max = z_max.max(v.z);
    }

    // 预计算逆矩阵（避免每点重复求逆）
    let inv_pose = box3d.pose.try_inverse().unwrap_or_else(|| panic!("矩阵不可求逆: {}", box3d.pose));
    let hl = box3d.length / 2.0;
    let hw = box3d.width / 2.0;
    let hh = box3d.height / 2.0;

    let candidates: Vec<[f32; 3]> = points.iter()
        .filter(|p| {
            p[0] >= x_min && p[0] <= x_max
                && p[1] >= y_min && p[1] <= y_max
                && p[2] >= z_min && p[2] <= z_max
                && {
                    let local = inv_pose.transform_point(&Point3::new(p[0], p[1], p[2]));
                    local.x >= -hl && local.x <= hl
                        && local.y >= -hw && local.y <= hw
                        && local.z >= -hh && local.z <= hh
                }
        })
        .copied()
        .collect();

    if candidates.len() <= max_out {
        candidates
    } else {
        // 均匀步长下采样到 max_out 个点
        let step = candidates.len() / max_out;
        candidates.into_iter().step_by(step).take(max_out).collect()
    }
}

/// 更新所有活跃轨迹的点云历史
///
/// 先计算所有目标 AABB 的并集，快速过滤非目标区域点云，
/// 再逐目标精确过滤，避免对每个目标扫描整个点云。
pub(crate) fn update_object_point_clouds(
    objects: &mut BTreeMap<usize, TrackedObject>,
    filter_points: &[[f32; 3]],
    max_history: usize,
    max_points_per_obj: usize,
) {
    // Step 1: 收集所有目标 box AABB 并集
    let boxes: Vec<&Box3D> = objects.values()
        .filter_map(|obj| obj.last_box.as_ref())
        .collect();

    if boxes.is_empty() {
        return;
    }

    let (mut ax_min, mut ax_max) = (f32::MAX, f32::NEG_INFINITY);
    let (mut ay_min, mut ay_max) = (f32::MAX, f32::NEG_INFINITY);
    let (mut az_min, mut az_max) = (f32::MAX, f32::NEG_INFINITY);
    for b in &boxes {
        let v = b.vertices();
        for p in &v {
            ax_min = ax_min.min(p.x); ax_max = ax_max.max(p.x);
            ay_min = ay_min.min(p.y); ay_max = ay_max.max(p.y);
            az_min = az_min.min(p.z); az_max = az_max.max(p.z);
        }
    }

    // Step 2: 一次扫描，得到落在联合 AABB 内的候选点
    let candidates: Vec<[f32; 3]> = filter_points.iter()
        .filter(|p| {
            p[0] >= ax_min && p[0] <= ax_max
                && p[1] >= ay_min && p[1] <= ay_max
                && p[2] >= az_min && p[2] <= az_max
        })
        .copied()
        .collect();

    // Step 3: 逐目标精确过滤（仅对候选点操作）
    for obj in objects.values_mut() {
        if let Some(ref last_box) = obj.last_box {
            let pts = if candidates.is_empty() {
                extract_points_in_box(filter_points, last_box, max_points_per_obj)
            } else {
                extract_points_in_box(&candidates, last_box, max_points_per_obj)
            };
            if pts.is_empty() {
                continue;
            }
            if obj.point_cloud_history.len() >= max_history {
                obj.point_cloud_history.pop_front();
            }
            obj.point_cloud_history.push_back(pts);
        }
    }
}
