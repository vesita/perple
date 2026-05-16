use rand::{rng, seq::IndexedRandom};

pub fn select_some(start: usize, end: usize, count: usize) -> Vec<usize> {
    let ans = (start..end).collect::<Vec<_>>();
    let mut rng = rng();
    ans.sample(&mut rng, count).cloned().collect::<Vec<_>>()
}

/// 下采样点云，限制最多 `max_count` 个点（均匀抽取）
pub fn limit_points(points: &[[f32; 3]], max_count: usize) -> Vec<[f32; 3]> {
    if points.len() <= max_count {
        return points.to_vec();
    }
    let step = (points.len() / max_count).max(1);
    points.iter()
        .enumerate()
        .filter(|(i, _)| i % step == 0)
        .map(|(_, p)| *p)
        .collect()
}
