use std::collections::BTreeMap;

use crate::tracker::object::TrackedObject;

/// 基于 (id, vel) 快照的 DBSCAN 速度聚类
pub(crate) fn analyze_velocity_clusters_from_snapshot(snapshot: &[(usize, [f32; 3])]) -> Vec<usize> {
    let n = snapshot.len();
    if n < 2 {
        return Vec::new();
    }

    let ids: Vec<usize> = snapshot.iter().map(|(id, _)| *id).collect();
    let velocities: Vec<[f32; 3]> = snapshot.iter().map(|(_, v)| *v).collect();

    let eps = 0.3f32;
    let min_pts = 2;
    let mut neighbor_counts = vec![0; n];

    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let d = ((velocities[i][0] - velocities[j][0]).powi(2)
                + (velocities[i][1] - velocities[j][1]).powi(2)
                + (velocities[i][2] - velocities[j][2]).powi(2))
            .sqrt();
            if d < eps {
                neighbor_counts[i] += 1;
            }
        }
    }

    let mut clusters: Vec<Vec<usize>> = Vec::new();
    let mut assigned = vec![false; n];

    for i in 0..n {
        if assigned[i] || neighbor_counts[i] < min_pts {
            continue;
        }
        let mut cluster = Vec::new();
        let mut stack = vec![i];
        assigned[i] = true;
        while let Some(idx) = stack.pop() {
            cluster.push(idx);
            for j in 0..n {
                if assigned[j] {
                    continue;
                }
                let d = ((velocities[idx][0] - velocities[j][0]).powi(2)
                    + (velocities[idx][1] - velocities[j][1]).powi(2)
                    + (velocities[idx][2] - velocities[j][2]).powi(2))
                .sqrt();
                if d < eps {
                    assigned[j] = true;
                    stack.push(j);
                }
            }
        }
        clusters.push(cluster);
    }

    if let Some(largest) = clusters.iter().max_by_key(|c| c.len()) {
        largest.iter().map(|&pos| ids[pos]).collect()
    } else {
        Vec::new()
    }
}

/// 点云投票分析（直接引用 tracked_objects，无拷贝）
///
/// 动态投票：将所有历史帧的点云按时间分为两组（前半段 vs 后半段），
/// 累积组内所有点后比较运动方向与 Kalman 速度的一致性。
/// 相比原实现（只比两帧），累积后点更多，投票更稳定，减少目标因稀疏点云漏判。
///
/// 速度方向过滤（参考 LV-DOT 方法）：
/// - 位移方向与 Kalman 速度相反的点的排除出投票基数（不是简单地不给票）
/// - 位移幅度小于 `point_vel_threshold` 的不计为运动票
pub(crate) fn analyze_point_cloud_voting_direct(
    objects: &BTreeMap<usize, TrackedObject>,
    vote_threshold: f32,
    skip_frames: usize,
    point_vel_threshold: f32,
) -> Vec<usize> {
    let mut pass_ids = Vec::new();
    let ids: Vec<usize> = objects.keys().copied().collect();

    for id in &ids {
        let obj = match objects.get(id) {
            Some(obj) => obj,
            None => continue,
        };

        let hist_len = obj.point_cloud_history.len();
        // 至少需要 skip_frames+1 帧才有足够的前后对比
        if hist_len < skip_frames + 2 {
            continue;
        }

        let speed = obj.speed();
        if speed < 0.2 {
            continue;
        }

        // ─── 累积投票：把所有历史帧分为前后两组 ─────────────────
        let mid = hist_len - 1 - skip_frames; // 分割点
        let mut old_all: Vec<[f32; 3]> = Vec::new();
        let mut new_all: Vec<[f32; 3]> = Vec::new();

        for i in 0..hist_len {
            if i < mid {
                old_all.extend_from_slice(&obj.point_cloud_history[i]);
            } else {
                new_all.extend_from_slice(&obj.point_cloud_history[i]);
            }
        }

        if old_all.is_empty() || new_all.is_empty() {
            continue;
        }

        let vel = obj.kalman_filter.get_velocity();

        let mut votes = 0usize;
        let mut valid_count = 0usize; // 排除方向相反的点后的计数
        let total = new_all.len().min(old_all.len());
        for i in 0..total {
            let np = new_all[i];
            let best_old = old_all.iter().min_by(|a, b| {
                let da = (np[0] - a[0]).powi(2) + (np[1] - a[1]).powi(2) + (np[2] - a[2]).powi(2);
                let db = (np[0] - b[0]).powi(2) + (np[1] - b[1]).powi(2) + (np[2] - b[2]).powi(2);
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            });
            if let Some(op) = best_old {
                let dx = np[0] - op[0];
                let dy = np[1] - op[1];
                let dz = np[2] - op[2];
                let dot = dx * vel.x as f32 + dy * vel.y as f32 + dz * vel.z as f32;
                if dot <= 0.0 {
                    // 方向相反 → 排除此点（LV-DOT: 方向不一致的点不应参与投票）
                    continue;
                }
                valid_count += 1;
                // 位移幅度 > 阈值才计为运动票
                let disp_mag = (dx * dx + dy * dy + dz * dz).sqrt();
                if disp_mag > point_vel_threshold {
                    votes += 1;
                }
            }
        }

        if valid_count == 0 {
            continue;
        }
        let ratio = votes as f32 / valid_count as f32;
        if ratio >= vote_threshold {
            pass_ids.push(*id);
        }
    }

    pass_ids
}
