use crate::utils::boxes::Box3D;

use super::xy_grid::{CellKey, XYGrid};

/// 对非墙面点做网格连通域聚类，返回每个簇的 AABB。
///
/// 纯 XY 连通，Z 仅用于最终包围盒高度。
pub fn cluster_obstacles(points: &[[f32; 3]], cell_size: f32, min_pts: usize, min_edge: f32, _max_range: f32) -> Vec<Box3D> {
    let (boxes, _) = cluster_obstacles_with_indices(points, cell_size, min_pts, min_edge, _max_range);
    boxes
}

/// 网格连通域聚类，返回 (AABB 列表, 各簇原始点索引)。
///
/// 与 `cluster_obstacles` 相同算法，额外返回每个 box 对应的点索引，
/// 供下游 DBSCAN 精化使用。
#[allow(unused_variables)]
pub fn cluster_obstacles_with_indices(
    points: &[[f32; 3]], cell_size: f32, min_pts: usize, min_edge: f32, max_range: f32,
) -> (Vec<Box3D>, Vec<Vec<usize>>) {
    let n = points.len();
    if n == 0 { return (Vec::new(), Vec::new()); }

    let grid = XYGrid::new(points, cell_size);
    let valid: std::collections::HashSet<CellKey> = grid.dense_cells(min_pts)
        .into_iter().collect();

    let mut visited = std::collections::HashSet::new();
    let mut cell_clusters: Vec<Vec<CellKey>> = Vec::new();

    for &key in &valid {
        if visited.contains(&key) { continue; }
        let mut component = Vec::new();
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(key);
        visited.insert(key);
        while let Some(cur) = queue.pop_front() {
            component.push(cur);
            for dx in -1i32..=1 {
                for dy in -1i32..=1 {
                    if dx == 0 && dy == 0 { continue; }
                    let nbr = (cur.0 + dx, cur.1 + dy);
                    if valid.contains(&nbr) && !visited.contains(&nbr) {
                        visited.insert(nbr);
                        queue.push_back(nbr);
                    }
                }
            }
        }
        cell_clusters.push(component);
    }

    let mut boxes = Vec::new();
    let mut all_indices = Vec::new();
    for component in &cell_clusters {
        let mut indices = Vec::new();
        for key in component {
            if let Some(cell) = grid.cells.get(key) {
                indices.extend_from_slice(cell);
            }
        }
        if indices.is_empty() { continue; }
        let box3d = Box3D::from_cloud_aabb(
            &indices.iter().map(|&i| points[i]).collect::<Vec<_>>(),
            min_edge,
        );
        // 距离硬过滤（已禁用，用于对比实验）
        // if max_range > 0.0 && !box3d.is_in_xy_range([0.0; 3], max_range) {
        //     continue;
        // }
        boxes.push(box3d);
        all_indices.push(indices);
    }

    (boxes, all_indices)
}

/// XY 平面 DBSCAN，使用 XYGrid 空间索引。
///
/// 返回簇索引列表，每个簇是采样点集中的索引。
pub fn xy_dbscan(points: &[[f32; 3]], eps: f32, min_pts: usize) -> Vec<Vec<usize>> {
    let n = points.len();
    if n == 0 { return Vec::new(); }

    let grid = XYGrid::new(points, eps);
    let mut visited = vec![false; n];
    let mut clusters = Vec::new();
    let mut nbr_buf = Vec::new();

    for i in 0..n {
        if visited[i] { continue; }
        visited[i] = true;

        nbr_buf.clear();
        grid.query_neighbors(points, points[i][0], points[i][1], eps, &mut nbr_buf);
        if nbr_buf.len() < min_pts { continue; }

        let mut cluster = vec![i];
        let mut queue = std::collections::VecDeque::new();
        let seed_nbrs: Vec<usize> = nbr_buf.drain(..).collect();
        for &j in &seed_nbrs {
            if !visited[j] {
                visited[j] = true;
                queue.push_back(j);
            }
        }

        while let Some(cur) = queue.pop_front() {
            cluster.push(cur);
            nbr_buf.clear();
            grid.query_neighbors(points, points[cur][0], points[cur][1], eps, &mut nbr_buf);
            if nbr_buf.len() >= min_pts {
                for &j in &nbr_buf {
                    if !visited[j] {
                        visited[j] = true;
                        queue.push_back(j);
                    }
                }
            }
        }
        clusters.push(cluster);
    }

    clusters
}
