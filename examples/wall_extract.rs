use std::collections::VecDeque;
use std::time::Instant;

use perple::bench::BenchRecorder;
use perple::cloud::ground::create_ground_strategy;
use perple::optional::data_loader::DataLoader;
use perple::swapl::global_swapl;
use perple::utils::boxes::Box3D;
use perple::utils::random::select_some;

const WALL_DISTANCE: f32 = 0.15;
const WALL_ITERS: usize = 200;
const NORMAL_THRESH: f32 = 0.3;
const MAX_WALLS: usize = 5;
const CLUSTER_EPS: f32 = 0.3;
const CLUSTER_MIN: usize = 5;
const FRAME_LIMIT: usize = 10;

// redra 语义材质短名
const MAT_RAW: &str = "point_cloud";      // 暖白
const MAT_GROUND: &str = "ground";         // 暗橄榄绿
const MAT_WALL: &str = "red";              // 暖色警示
const MAT_REMAIN: &str = "yellow";         // 暖色，与红/绿/青均区分
const MAT_BOX: &str = "glass";             // 半透明包围盒，可透视内部点

// ─── 墙体 RANSAC ───

fn extract_walls(
    points: &mut [[f32; 3]],
    distance: f32,
    iterations: usize,
    normal_thresh: f32,
    max_walls: usize,
) -> (usize, Vec<[f32; 4]>, Vec<usize>) {
    let mut total_wall = 0usize;
    let mut walls = Vec::new();
    let mut wall_counts = Vec::new();

    for _ in 0..max_walls {
        let remaining = &mut points[total_wall..];
        if remaining.len() < 10 { break; }

        let (n_inliers, plane) = ransac_vertical_plane(remaining, distance, iterations, normal_thresh);
        if n_inliers < 10 { break; }

        walls.push(plane);
        wall_counts.push(n_inliers);
        total_wall += n_inliers;
    }

    (total_wall, walls, wall_counts)
}

fn ransac_vertical_plane(
    points: &mut [[f32; 3]],
    distance: f32,
    iterations: usize,
    normal_thresh: f32,
) -> (usize, [f32; 4]) {
    let n = points.len();
    if n < 3 { return (0, [0.0, 0.0, 1.0, 0.0]); }

    let mut best_count = 0usize;
    let mut best_plane = [0.0f32; 4];

    for _ in 0..iterations {
        let idxs = select_some(0, n, 3);
        let p1 = points[idxs[0]];
        let p2 = points[idxs[1]];
        let p3 = points[idxs[2]];

        let v1 = [p2[0]-p1[0], p2[1]-p1[1], p2[2]-p1[2]];
        let v2 = [p3[0]-p1[0], p3[1]-p1[1], p3[2]-p1[2]];

        let nx = v1[1]*v2[2] - v1[2]*v2[1];
        let ny = v1[2]*v2[0] - v1[0]*v2[2];
        let nz = v1[0]*v2[1] - v1[1]*v2[0];
        let len = (nx*nx + ny*ny + nz*nz).sqrt();
        if len < 1e-6 { continue; }
        let (nx, ny, nz) = (nx/len, ny/len, nz/len);

        // 竖直性约束：法线接近水平（|nz| 小）→ 竖直平面 → 墙
        if nz.abs() > normal_thresh { continue; }

        let d = -(nx*p1[0] + ny*p1[1] + nz*p1[2]);

        let count = points.iter().filter(|p| {
            (nx*p[0] + ny*p[1] + nz*p[2] + d).abs() < distance
        }).count();

        if count > best_count {
            best_count = count;
            best_plane = [nx, ny, nz, d];
        }
    }

    if best_count == 0 { return (0, [0.0, 0.0, 1.0, 0.0]); }

    // Partition: 内点移到前面
    let [nx, ny, nz, d] = best_plane;
    let mut write = 0usize;
    for read in 0..n {
        if (nx*points[read][0] + ny*points[read][1] + nz*points[read][2] + d).abs() < distance {
            points.swap(read, write);
            write += 1;
        }
    }

    (write, best_plane)
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

    println!("=== 墙体提取测试：法线 → RANSAC 墙面 → 欧式聚类 ===\n");

    let mut data_loader = DataLoader::new("./data/test".into());
    data_loader.set_frame_limit(FRAME_LIMIT);
    data_loader.load().await?;

    let mut recorder = BenchRecorder::new();
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

        // 2. 墙体 RANSAC
        let mut wall_points: Vec<[f32; 3]> = non_ground.to_vec();

        let t1 = Instant::now();
        let (n_walls, wall_planes, wall_counts) = extract_walls(
            &mut wall_points, WALL_DISTANCE, WALL_ITERS, NORMAL_THRESH, MAX_WALLS,
        );
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
        // 墙面平面方程 tag（半透明包围盒）
        let mut wall_offset = 0usize;
        for (wi, (plane, &count)) in wall_planes.iter().zip(wall_counts.iter()).enumerate() {
            let tag = format!("wall{} n=({:.2},{:.2},{:.2}) d={:.2} {}pts",
                wi, plane[0], plane[1], plane[2], plane[3], count);
            let mut wall_box = Box3D::empty_box();
            wall_box.cloud2box(&wall_points[wall_offset..wall_offset + count].to_vec());
            recorder.write_boxes(&[(wall_box, tag)], MAT_BOX);
            wall_offset += count;
        }

        // 剩余点（黄色）
        let remain_step = (remaining.len() / 3000).max(1);
        for i in (0..remaining.len()).step_by(remain_step) {
            recorder.write_point_cloud(&[remaining[i]], MAT_REMAIN, 1);
        }
        // 聚类包围盒（半透明，可透视内部点）
        for (ci, cluster) in clusters.iter().enumerate() {
            let pts: Vec<[f32; 3]> = cluster.iter().map(|&i| remaining[i]).collect();
            let mut box3d = Box3D::empty_box();
            box3d.cloud2box(&pts);
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
