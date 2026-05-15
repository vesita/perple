use std::collections::HashMap;

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
pub(crate) fn analyze_point_cloud_voting_direct(
    objects: &HashMap<usize, TrackedObject>,
    vote_threshold: f32,
    skip_frames: usize,
) -> Vec<usize> {
    let mut pass_ids = Vec::new();
    let ids: Vec<usize> = objects.keys().copied().collect();

    for id in &ids {
        let obj = match objects.get(id) {
            Some(obj) => obj,
            None => continue,
        };

        let hist_len = obj.point_cloud_history.len();
        if hist_len <= skip_frames {
            continue;
        }

        let old_pts = &obj.point_cloud_history[hist_len - 1 - skip_frames];
        let new_pts = &obj.point_cloud_history[hist_len - 1];

        if old_pts.is_empty() || new_pts.is_empty() {
            continue;
        }

        let speed = obj.speed();
        if speed < 0.2 {
            continue;
        }

        let vel = obj.kalman_filter.get_velocity();

        let mut votes = 0usize;
        let total = new_pts.len().min(old_pts.len());
        if total == 0 {
            continue;
        }

        for i in 0..total {
            let np = new_pts[i];
            let best_old = old_pts.iter().min_by(|a, b| {
                let da = (np[0] - a[0]).powi(2) + (np[1] - a[1]).powi(2) + (np[2] - a[2]).powi(2);
                let db = (np[0] - b[0]).powi(2) + (np[1] - b[1]).powi(2) + (np[2] - b[2]).powi(2);
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            });
            if let Some(op) = best_old {
                let dx = np[0] - op[0];
                let dy = np[1] - op[1];
                let dz = np[2] - op[2];
                let dot = dx * vel.x as f32 + dy * vel.y as f32 + dz * vel.z as f32;
                if dot > 0.0 {
                    votes += 1;
                }
            }
        }

        let ratio = votes as f32 / total as f32;
        if ratio >= vote_threshold {
            pass_ids.push(*id);
        }
    }

    pass_ids
}
