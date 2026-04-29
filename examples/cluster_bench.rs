use std::time::Instant;

use perple::optional::data_loader::DataLoader;
use perple::swapl::global_swapl;
use perple::utils::boxes::Box3D;
use perple::cloud::classify::environment::single_pick_ground;

use expto::rdmp::auto::unit::generate_unit;
use expto::rdmp::proto::command::{CommandType, ExCommand};
use expto::rdmp::*;
use redra_client::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();

    println!("=== 室内聚类策略对比测试 ===");

    // ── 加载数据 ──
    let mut data_loader = DataLoader::new("./data/test".to_string());
    data_loader.set_frame_limit(3);
    let _ = data_loader.load().await;

    let swapl = global_swapl();
    let mut cloud = {
        let mut stream = swapl.clouds.lock().await;
        stream.read().unwrap_or_default()
    };

    if cloud.is_empty() {
        eprintln!("无数据");
        return Ok(());
    }
    println!("点云总数: {}", cloud.len());

    // ── 提走地面，只对非地面点做聚类对比 ──
    let (n_ground, _) = single_pick_ground(&mut cloud);
    let non_ground = &cloud[n_ground..];
    println!("地面点: {}, 非地面点: {}\n", n_ground, non_ground.len());

    // ── 测试参数组合 ──
    let patience_values = [0.15, 0.25, 0.35, 0.50, 0.80];
    let min_pts_values = [3, 5, 8, 15];
    let voxel_values = [0.05, 0.10, 0.20];

    let mut results: Vec<BenchResult> = Vec::new();

    // ── 策略 1: 当前默认参数 ──
    {
        let points = non_ground.to_vec();
        let start = Instant::now();
        let (clusters, noise) = dbscan_2d(&points, 0.20, 10, 0.10);
        let elapsed = start.elapsed();
        let n_humans = count_human_like(&clusters);
        println!(
            "  [默认] eps=0.20 min=10 vox=0.10 → {}簇 {}噪声 {}大目标  {:.1}ms",
            clusters.len(), noise, n_humans, elapsed.as_secs_f64() * 1000.0
        );
        results.push(BenchResult {
            label: "默认 eps0.20".to_string(),
            n_clusters: clusters.len(),
            n_noise: noise,
            n_humans,
            clusters,
            elapsed_ms: elapsed.as_secs_f64() * 1000.0,
        });
    }

    // ── 策略 2: 遍历参数组合 ──
    for &voxel in &voxel_values {
        for &eps in &patience_values {
            for &min_pts in &min_pts_values {
                let points = non_ground.to_vec();
                let start = Instant::now();
                let (clusters, noise) = dbscan_2d(&points, eps, min_pts, voxel);
                let elapsed = start.elapsed();
                let n_humans = count_human_like(&clusters);

                if clusters.len() > 0 || noise > 0 {
                    println!(
                        "  eps={:.2} min={} vox={:.2} → {}簇 {}噪声 {}大目标  {:.1}ms",
                        eps, min_pts, voxel, clusters.len(), noise, n_humans,
                        elapsed.as_secs_f64() * 1000.0
                    );
                    results.push(BenchResult {
                        label: format!("eps{:.2}_m{}_v{:.2}", eps, min_pts, voxel),
                        n_clusters: clusters.len(),
                        n_noise: noise,
                        n_humans,
                        clusters,
                        elapsed_ms: elapsed.as_secs_f64() * 1000.0,
                    });
                }
            }
        }
    }

    // ── 策略 3: 自适应 eps DBSCAN（近密小 eps，远疏大 eps） ──
    let eps0_values = [0.05, 0.10, 0.15];
    let slope_values = [0.02, 0.05, 0.10];
    for &voxel in &voxel_values {
        for &eps_0 in &eps0_values {
            for &slope in &slope_values {
                for &min_pts in &min_pts_values {
                    let points = non_ground.to_vec();
                    let start = Instant::now();
                    let (clusters, noise) = dbscan_2d_adaptive(&points, eps_0, slope, min_pts, voxel);
                    let elapsed = start.elapsed();
                    let n_humans = count_human_like(&clusters);

                    if clusters.len() > 0 || noise > 0 {
                        println!(
                            "  adapt eps0={:.2} s={:.2} m{} v{:.2} → {}簇 {}噪声 {}大目标  {:.1}ms",
                            eps_0, slope, min_pts, voxel, clusters.len(), noise, n_humans,
                            elapsed.as_secs_f64() * 1000.0
                        );
                        results.push(BenchResult {
                            label: format!("adapt_e{:.2}_s{:.2}_m{}_v{:.2}", eps_0, slope, min_pts, voxel),
                            n_clusters: clusters.len(),
                            n_noise: noise,
                            n_humans,
                            clusters,
                            elapsed_ms: elapsed.as_secs_f64() * 1000.0,
                        });
                    }
                }
            }
        }
    }

    // ── 汇总表 ──
    println!("\n");
    println!("{:-<100}", "");
    println!("| {:<32} | {:>6} | {:>5} | {:>5} | {:>7} | {:>7} |", "策略", "簇数", "噪声", "大目标", "耗时ms", "大目标占比");
    println!("{:-<100}", "");

    // 排序：大目标数降序（尺寸过滤，供跟踪器参考）
    results.sort_by(|a, b| b.n_humans.cmp(&a.n_humans));

    for r in &results {
        let human_ratio = if r.n_clusters > 0 {
            r.n_humans as f64 / r.n_clusters as f64 * 100.0
        } else {
            0.0
        };
        println!("| {:<32} | {:>6} | {:>5} | {:>5} | {:>7.1} | {:>6.0}% |",
            r.label, r.n_clusters, r.n_noise, r.n_humans, r.elapsed_ms, human_ratio);
    }
    println!("{:-<100}", "");

    // ── 发送最优几个到 redra 可视化 ──
    println!("\n发送到 redra 可视化...");
    let best = if results.len() > 6 { &results[..6] } else { &results[..] };
    for (idx, r) in best.iter().enumerate() {
        send_cluster_result(idx, r).await?;
        // 每组结果单独一帧，方便在 redra 中翻页对比
        let mut unit = generate_unit();
        unit.command = Some(ExCommand { u_command: CommandType::Frameend as i32 });
        unit.send().await?;
    }

    println!("\n建议：关注 大目标多 + 噪声少 的组合");
    println!("查看 redra 可视化。");
    Ok(())
}

struct BenchResult {
    label: String,
    n_clusters: usize,
    n_noise: usize,
    n_humans: usize,
    clusters: Vec<Vec<[f32; 3]>>,
    elapsed_ms: f64,
}

// ── 2D DBSCAN（XY 平面 + Z 范围过滤） ──
fn dbscan_2d(
    points: &[[f32; 3]],
    eps: f32,
    min_pts: usize,
    voxel_size: f32,
) -> (Vec<Vec<[f32; 3]>>, usize) {
    if points.is_empty() {
        return (Vec::new(), 0);
    }

    // 1. 体素下采样
    let sampled = voxel_sample(points, voxel_size);
    let n = sampled.len();
    if n == 0 { return (Vec::new(), 0); }

    // 2. 固定 eps 邻域统计
    let neighbor_counts = count_neighbors_fixed(&sampled, eps);
    // 3. DBSCAN 扩张
    expand_clusters(&sampled, eps, min_pts, &neighbor_counts)
}

/// 自适应 eps DBSCAN：近密小 eps，远疏大 eps
fn dbscan_2d_adaptive(
    points: &[[f32; 3]],
    eps_0: f32,
    slope: f32,
    min_pts: usize,
    voxel_size: f32,
) -> (Vec<Vec<[f32; 3]>>, usize) {
    if points.is_empty() {
        return (Vec::new(), 0);
    }

    let sampled = voxel_sample(points, voxel_size);
    let n = sampled.len();
    if n == 0 { return (Vec::new(), 0); }

    // 预计算每个点到 LiDAR（原点）的 XY 距离
    let ranges: Vec<f32> = sampled.iter()
        .map(|p| (p[0] * p[0] + p[1] * p[1]).sqrt())
        .collect();

    // 自适应邻域：eps = eps_0 + slope * max(r_i, r_j)
    let mut neighbor_counts = vec![0usize; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let dx = sampled[i][0] - sampled[j][0];
            let dy = sampled[i][1] - sampled[j][1];
            let dz = (sampled[i][2] - sampled[j][2]).abs();
            let d2 = dx * dx + dy * dy;
            let eps_ij = eps_0 + slope * ranges[i].max(ranges[j]);
            if d2 < eps_ij * eps_ij && dz < eps_ij {
                neighbor_counts[i] += 1;
                neighbor_counts[j] += 1;
            }
        }
    }

    // DBSCAN 扩张也用自适应 eps
    let mut labels = vec![-1i32; n];
    let mut cluster_id = 0i32;
    let mut clusters: Vec<Vec<[f32; 3]>> = Vec::new();

    for i in 0..n {
        if labels[i] >= 0 { continue; }
        if neighbor_counts[i] < min_pts { continue; }

        let mut stack = vec![i];
        labels[i] = cluster_id;
        let mut members = vec![i];

        while let Some(seed) = stack.pop() {
            let eps_seed = eps_0 + slope * ranges[seed];
            for j in 0..n {
                if labels[j] >= 0 { continue; }
                let dx = sampled[seed][0] - sampled[j][0];
                let dy = sampled[seed][1] - sampled[j][1];
                let dz = (sampled[seed][2] - sampled[j][2]).abs();
                if dx * dx + dy * dy < eps_seed * eps_seed && dz < eps_seed {
                    labels[j] = cluster_id;
                    members.push(j);
                    if neighbor_counts[j] >= min_pts {
                        stack.push(j);
                    }
                }
            }
        }
        clusters.push(members.iter().map(|&idx| sampled[idx]).collect());
        cluster_id += 1;
    }

    let noise_count = labels.iter().filter(|&&l| l == -1).count();
    (clusters, noise_count)
}

/// 体素下采样
fn voxel_sample(points: &[[f32; 3]], voxel_size: f32) -> Vec<[f32; 3]> {
    let mut seen = std::collections::HashSet::new();
    points.iter().filter(|p| {
        let key = [
            (p[0] / voxel_size).floor() as i32,
            (p[1] / voxel_size).floor() as i32,
            (p[2] / voxel_size).floor() as i32,
        ];
        seen.insert(key)
    }).copied().collect()
}

/// 固定 eps 邻域计数
fn count_neighbors_fixed(sampled: &[[f32; 3]], eps: f32) -> Vec<usize> {
    let n = sampled.len();
    let mut counts = vec![0usize; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let dx = sampled[i][0] - sampled[j][0];
            let dy = sampled[i][1] - sampled[j][1];
            let dz = (sampled[i][2] - sampled[j][2]).abs();
            if dx * dx + dy * dy < eps * eps && dz < eps {
                counts[i] += 1;
                counts[j] += 1;
            }
        }
    }
    counts
}

/// DBSCAN 簇扩张（固定 eps）
fn expand_clusters(
    sampled: &[[f32; 3]],
    eps: f32,
    min_pts: usize,
    neighbor_counts: &[usize],
) -> (Vec<Vec<[f32; 3]>>, usize) {
    let n = sampled.len();
    let mut labels = vec![-1i32; n];
    let mut cluster_id = 0i32;
    let mut clusters: Vec<Vec<[f32; 3]>> = Vec::new();

    for i in 0..n {
        if labels[i] >= 0 { continue; }
        if neighbor_counts[i] < min_pts { continue; }

        let mut stack = vec![i];
        labels[i] = cluster_id;
        let mut members = vec![i];

        while let Some(seed) = stack.pop() {
            for j in 0..n {
                if labels[j] >= 0 { continue; }
                let dx = sampled[seed][0] - sampled[j][0];
                let dy = sampled[seed][1] - sampled[j][1];
                let dz = (sampled[seed][2] - sampled[j][2]).abs();
                if dx * dx + dy * dy < eps * eps && dz < eps {
                    labels[j] = cluster_id;
                    members.push(j);
                    if neighbor_counts[j] >= min_pts {
                        stack.push(j);
                    }
                }
            }
        }
        clusters.push(members.iter().map(|&idx| sampled[idx]).collect());
        cluster_id += 1;
    }

    let noise_count = labels.iter().filter(|&&l| l == -1).count();
    (clusters, noise_count)
}

/// 按目标尺寸过滤（剔除过小/过大的簇，适合跟踪器的初始筛选）
fn count_human_like(clusters: &[Vec<[f32; 3]>]) -> usize {
    let mut count = 0;
    for cluster in clusters {
        if cluster.len() < 3 { continue; }
        let mut box3d = Box3D::empty_box();
        box3d.cloud2box(&cluster);
        let w = box3d.length.max(box3d.width);
        let h = box3d.height;
        let xy_ratio = if h > 0.0 { w / h } else { 999.0 };

        // 典型站立目标：高>宽，高1.0~2.5m，宽<1.2m
        if h > w * 0.5 && w < 1.2 && h > 1.0 && h < 2.5 && xy_ratio < 1.5 {
            count += 1;
        }
    }
    count
}

/// 发送聚类结果到 redra（每帧独立，不同策略用不同颜色）
async fn send_cluster_result(
    idx: usize,
    result: &BenchResult,
) -> Result<(), Box<dyn std::error::Error>> {
    let colors = ["red", "green", "blue", "yellow", "magenta", "cyan"];
    let color = colors[idx % colors.len()];

    // 发标签
    {
        let mut tag = generate_unit();
        tag.objects.extend(vec![
            ExObject::from(100_000u64 + idx as u64),
            ExObject::from(Tag::new(format!(
                "{} | {}簇 {}人 {:.0}ms",
                result.label, result.n_clusters, result.n_humans, result.elapsed_ms,
            )).with_offset(ExTransform {
                x: 0.0, y: 10.0, z: 0.0,
                rx: 0.0, ry: 0.0, rz: 0.0,
                sx: 1.0, sy: 1.0, sz: 1.0,
            })),
        ]);
        tag.send().await?;
    }

    // 发每个簇的包围盒
    for (ci, cluster) in result.clusters.iter().enumerate() {
        if cluster.is_empty() { continue; }

        let mut box3d = Box3D::empty_box();
        box3d.cloud2box(cluster);
        let verts = box3d.vertices();
        let points: Vec<Point> = verts.iter()
            .map(|v| Point { x: v.x, y: v.y, z: v.z })
            .collect();

        let mut min = [f32::MAX, f32::MAX, f32::MAX];
        let mut max = [f32::MIN, f32::MIN, f32::MIN];
        for p in &points {
            min[0] = min[0].min(p.x); min[1] = min[1].min(p.y); min[2] = min[2].min(p.z);
            max[0] = max[0].max(p.x); max[1] = max[1].max(p.y); max[2] = max[2].max(p.z);
        }
        let cx = (min[0] + max[0]) / 2.0;
        let cy = (min[1] + max[1]) / 2.0;
        let cz = (min[2] + max[2]) / 2.0;

        let eid = 200_000 + (idx as u64) * 1000 + (ci as u64);

        let mut unit = generate_unit();
        unit.objects.extend(vec![
            ExObject::from(eid),
            ExObject::from(ExMesh::from(Cube { vertices: points })),
            ExObject::from(ExTransform {
                x: cx, y: cy, z: cz,
                rx: 0.0, ry: 0.0, rz: 0.0,
                sx: 1.0, sy: 1.0, sz: 1.0,
            }),
            ExObject { u_object: Some(ex_object::UObject::MaterialId(color.to_string())) },
            ExObject::from(Tag::new(format!(
                "{}pts h={:.1}m",
                cluster.len(),
                box3d.height,
            )).with_offset(ExTransform {
                x: cx, y: cy + box3d.height / 2.0 + 0.3, z: cz,
                rx: 0.0, ry: 0.0, rz: 0.0,
                sx: 1.0, sy: 1.0, sz: 1.0,
            })),
        ]);
        unit.send().await?;
    }

    Ok(())
}
