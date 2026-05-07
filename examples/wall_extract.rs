use std::collections::VecDeque;
use std::time::Instant;

use perple::bench::BenchRecorder;
use perple::cloud::ground::create_ground_strategy;
use perple::cloud::wall::{WallPickStrategy, TopDownCluster};
use perple::optional::data_loader::DataLoader;
use perple::swapl::global_swapl;
use perple::utils::boxes::Box3D;

const CLUSTER_EPS: f32 = 0.3;
const CLUSTER_MIN: usize = 5;
const FRAME_LIMIT: usize = 10;

// redra 语义材质短名
const MAT_RAW: &str = "point_cloud";      // 暖白
const MAT_GROUND: &str = "ground";         // 暗橄榄绿
const MAT_WALL: &str = "red";              // 暖色警示
const MAT_REMAIN: &str = "yellow";         // 黄色剩余点
const MAT_BOX: &str = "disabled";          // 暗灰半透明包围盒

// ─── 墙面点分配 ───

fn assign_wall_counts(wall_points: &[[f32; 3]], planes: &[[f32; 4]]) -> Vec<usize> {
    if planes.is_empty() { return Vec::new(); }
    let mut counts = vec![0usize; planes.len()];
    for p in wall_points {
        let mut best = 0;
        let mut best_dist = f32::MAX;
        for (i, plane) in planes.iter().enumerate() {
            let dist = (plane[0]*p[0] + plane[1]*p[1] + plane[2]*p[2] + plane[3]).abs();
            if dist < best_dist {
                best_dist = dist;
                best = i;
            }
        }
        counts[best] += 1;
    }
    counts
}

// ─── 欧式聚类 ───

fn euclidean_cluster(points: &[[f32; 3]], eps: f32, min_pts: usize) -> Vec<Vec<usize>> {
    let n = points.len();
    let eps2 = eps * eps;
    let mut visited = vec![false; n];
    let mut clusters = Vec::new();

    for i in 0..n {
        if visited[i] { continue; }
        visited[i] = true;

        let mut queue = VecDeque::new();
        queue.push_back(i);
        let mut cluster = vec![i];

        while let Some(cur) = queue.pop_front() {
            for j in 0..n {
                if visited[j] { continue; }
                let dx = points[j][0] - points[cur][0];
                let dy = points[j][1] - points[cur][1];
                let dz = points[j][2] - points[cur][2];
                if dx*dx + dy*dy + dz*dz <= eps2 {
                    visited[j] = true;
                    queue.push_back(j);
                    cluster.push(j);
                }
            }
        }

        if cluster.len() >= min_pts {
            clusters.push(cluster);
        }
    }
    clusters
}

// ─── main ───

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();

    println!("=== 墙体提取测试：RANSAC 墙面 → 欧式聚类 ===\n");

    let mut data_loader = DataLoader::new("./data/test".into());
    data_loader.set_frame_limit(FRAME_LIMIT);
    data_loader.load().await?;

    let mut recorder = BenchRecorder::new();
    let mut wall_strategy = TopDownCluster::new();
    let mut frame_idx = 0usize;

    while data_loader.load_next().await.unwrap_or(false) {
        let cloud = {
            let swapl = global_swapl();
            let mut stream = swapl.clouds.lock().await;
            match stream.read() { Some(d) => d, None => continue }
        };

        // 1. 地面提取
        let mut ground_cloud = cloud.clone();
        let mut ground_strategy = create_ground_strategy();
        let (n_ground, _, _) = ground_strategy.pick(&mut ground_cloud);
        let non_ground = &ground_cloud[n_ground..];

        if non_ground.len() < 10 {
            frame_idx += 1;
            continue;
        }

        // 2. 墙体提取（使用 TopDownCluster 策略）
        let mut wall_points: Vec<[f32; 3]> = non_ground.to_vec();

        let t1 = Instant::now();
        let (n_walls, wall_planes) = wall_strategy.pick(&mut wall_points);
        let wall_ms = t1.elapsed().as_secs_f64() * 1000.0;

        // 3. 欧式聚类（剔除墙面后的剩余点）
        let remaining = &wall_points[n_walls..];
        let t2 = Instant::now();
        let clusters = euclidean_cluster(remaining, CLUSTER_EPS, CLUSTER_MIN);
        let cluster_ms = t2.elapsed().as_secs_f64() * 1000.0;

        println!("帧 {}: 地面={} 非地面={} 墙面={} 剩余={} 簇={} | 墙面 {:.1}ms 聚类 {:.1}ms",
            frame_idx, n_ground, non_ground.len(), n_walls, remaining.len(), clusters.len(),
            wall_ms, cluster_ms);

        // 5. 输出 .rdra
        recorder.begin_frame(frame_idx);

        // 原始点云背景（受 write_raw 开关控制，默认关闭避免与分类点云重复）
        let raw_step = (cloud.len() / 5000).max(1);
        for i in (0..cloud.len()).step_by(raw_step) {
            recorder.write_raw_cloud(&[cloud[i]], MAT_RAW, 1);
        }

        // 地面（暗橄榄绿，语义层）
        let ground_step = (n_ground / 3000).max(1);
        for i in (0..n_ground).step_by(ground_step) {
            recorder.write_point_cloud(&[ground_cloud[i]], MAT_GROUND, 1);
        }

        // 墙面（红色，暖色警示）
        let wall_step = (n_walls / 2000).max(1);
        for i in (0..n_walls).step_by(wall_step) {
            recorder.write_point_cloud(&[wall_points[i]], MAT_WALL, 1);
        }
        // 墙面 OBB 包围盒（按平面方程分配点到各墙面）
        let wall_counts = assign_wall_counts(&wall_points[..n_walls], &wall_planes);
        let mut wall_offset = 0usize;
        for (wi, (plane, &count)) in wall_planes.iter().zip(wall_counts.iter()).enumerate() {
            if count == 0 { continue; }
            let wall_pts: Vec<[f32; 3]> = wall_points[wall_offset..wall_offset + count].to_vec();
            let wall_box = Box3D::from_cloud_aabb(&wall_pts, 0.05);
            let tag = format!("wall{} n=({:.2},{:.2},{:.2}) d={:.2} {:.1}x{:.1}x{:.1} {}pts",
                wi, plane[0], plane[1], plane[2], plane[3],
                wall_box.length, wall_box.width, wall_box.height, count);
            recorder.write_boxes(&[(wall_box, tag)], MAT_BOX);
            wall_offset += count;
        }

        // 剩余点（黄色）
        let remain_step = (remaining.len() / 3000).max(1);
        for i in (0..remaining.len()).step_by(remain_step) {
            recorder.write_point_cloud(&[remaining[i]], MAT_REMAIN, 1);
        }
        // 聚类包围盒（半透明，最小边长 0.05m）
        for (ci, cluster) in clusters.iter().enumerate() {
            let pts: Vec<[f32; 3]> = cluster.iter().map(|&i| remaining[i]).collect();
            let box3d = Box3D::from_cloud_aabb(&pts, 0.05);
            let tag = format!("cluster{} {}pts", ci, pts.len());
            recorder.write_boxes(&[(box3d, tag)], MAT_BOX);
        }

        recorder.end_frame();
        frame_idx += 1;
    }

    let out_path = "output/wall_extract";
    std::fs::create_dir_all("output")?;
    recorder.save(&format!("{}.rdra", out_path))?;
    println!("\n已保存到 {}.rdra", out_path);

    Ok(())
}
