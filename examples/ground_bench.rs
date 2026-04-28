use std::time::Instant;

use perple::config::fixif;
use perple::optional::data_loader::DataLoader;
use perple::swapl::global_swapl;
use perple::utils::random::select_some;
use perple::cloud::CldBud;
use perple::utils::boxes::Box3D;

use expto::rdmp::auto::unit::generate_unit;
use expto::rdmp::proto::command::{CommandType, ExCommand};
use expto::rdmp::*;
use redra_client::*;

use nalgebra::{Matrix3, Vector3, SVD};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();

    println!("=== 地面检测策略对比测试 ===");

    // ── 加载数据 ──
    let mut data_loader = DataLoader::new("./data/test".to_string());
    data_loader.set_frame_limit(5);
    let _ = data_loader.load().await;

    let swapl = global_swapl();
    let cloud = {
        let mut stream = swapl.clouds.lock().await;
        stream.read().unwrap_or_default()
    };
    let _ = swapl;

    let upside_down = fixif().upside_down;

    if cloud.is_empty() {
        eprintln!("无数据");
        return Ok(());
    }

    println!("点云总数: {}\n", cloud.len());

    // ── 测试所有策略，收集结果 ──
    let mut results: Vec<(String, Vec<[f32; 3]>, f32)> = Vec::new();

    // 策略 1：Z-直方图 + expand
    for &expand in &[0.05, 0.10, 0.15, 0.20, 0.30] {
        let mut c = cloud.clone();
        let start = Instant::now();
        let (n_ground, _) = histogram_expand(&mut c, expand, upside_down);
        let elapsed = start.elapsed();
        let ground_pts: Vec<[f32; 3]> = c[..n_ground].to_vec();
        print_result("直方图+expand", elapsed, expand, n_ground, cloud.len());
        results.push((format!("expand={:.2}", expand), ground_pts, expand));
    }

    // 策略 2：从峰值向下扫（threshold）+ 向上扩（expand）
    for &threshold in &[0.05, 0.10, 0.15, 0.20] {
        for &expand in &[0.05, 0.10, 0.20] {
            let mut c = cloud.clone();
            let start = Instant::now();
            let (n_ground, _n_ceil) = peak_down_expand_up(&mut c, threshold, expand, 128, upside_down);
            let elapsed = start.elapsed();
            if n_ground > 0 {
                let ground_pts: Vec<[f32; 3]> = c[..n_ground].to_vec();
                print_result("峰下扫+上扩", elapsed, threshold, n_ground, cloud.len());
                results.push((
                    format!("sd={:.2}_ex={:.2}", threshold, expand),
                    ground_pts, threshold + expand,
                ));
            }
        }
    }

    // 策略 3：RANSAC
    for &distance in &[0.3, 0.5] {
        let mut c = cloud.clone();
        let start = Instant::now();
        let (n_ground, _) = ransac_ground(&mut c, distance, 200, 0.25);
        let elapsed = start.elapsed();
        let ground_pts: Vec<[f32; 3]> = c[..n_ground].to_vec();
        print_result("RANSAC", elapsed, distance, n_ground, cloud.len());
        results.push((format!("ransac={:.1}", distance), ground_pts, distance + 2.0));
    }

    // 策略 4：直方图种子 → 平面拟合生长
    for &distance in &[0.3, 0.5] {
        for &expand in &[0.10, 0.20] {
            let mut c = cloud.clone();
            let start = Instant::now();
            let (n_ground, _) = histoseed_grow(&mut c, expand, distance, 100, upside_down);
            let elapsed = start.elapsed();
            if n_ground > 0 {
                let ground_pts: Vec<[f32; 3]> = c[..n_ground].to_vec();
                print_result("种子+生长", elapsed, distance, n_ground, cloud.len());
                results.push((
                    format!("seed_ex={:.2}_d={:.1}", expand, distance),
                    ground_pts, distance + expand,
                ));
            }
        }
    }

    // 策略 5：GPF（Ground Plane Fitting）— SVD 迭代平面拟合
    for &n_lpr in &[100, 200] {
        for &th_dist in &[0.2, 0.3, 0.5] {
            let mut c = cloud.clone();
            let start = Instant::now();
            let (n_ground, _) = gpf_ground(&mut c, n_lpr, 0.5, th_dist, 3, upside_down);
            let elapsed = start.elapsed();
            if n_ground > 0 {
                let ground_pts: Vec<[f32; 3]> = c[..n_ground].to_vec();
                print_result("GPF", elapsed, th_dist, n_ground, cloud.len());
                results.push((
                    format!("gpf_nlpr={}_d={:.1}", n_lpr, th_dist),
                    ground_pts, th_dist,
                ));
            }
        }
    }

    // ── 发送到 redra 可视化 ──
    println!("\n发送地面点到 redra ...");

    let colors = ["red", "green", "blue", "yellow", "cyan", "magenta"];

    // 每个策略单独一帧
    for (idx, (label, points, _)) in results.iter().enumerate() {
        let color = colors[idx % colors.len()];
        
        // 下采样到最多 3000 点
        let step = (points.len() / 3000).max(1);
        let sampled: Vec<[f32; 3]> = points.iter()
            .enumerate()
            .filter(|(i, _)| i % step == 0)
            .map(|(_, p)| *p)
            .collect();

        println!("  [Frame {}] {} ({}): {} 点 / 采样 {} → color={}",
            idx + 1, label, points.len(), sampled.len(), step, color);

        // 每帧只发送该策略的地面点
        send_colored_cloud(&sampled, color, 1_000_000u64).await?;

        // 发送标签
        send_label(&format!("{} ({})", label, color), -5.0, 5.0, 2.0, 1_100_000u64).await?;

        // 发送帧结束命令，分隔不同策略的帧
        {
            let mut unit = generate_unit();
            unit.command = Some(ExCommand { u_command: CommandType::Frameend as i32 });
            unit.send().await?;
        }

        // 短暂延迟以便观察
        tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
    }

    println!("\n可视化完成，请查看 redra。共 {} 帧，每帧一个策略。", results.len());
    Ok(())
}

async fn send_colored_cloud(points: &[[f32; 3]], color: &str, base_id: u64) -> Result<(), Box<dyn std::error::Error>> {
    if points.is_empty() { return Ok(()); }
    let mut unit = generate_unit();
    for (i, p) in points.iter().enumerate() {
        let eid = base_id + (i as u64) * 4;
        unit.objects.extend(vec![
            ExObject::from(eid),
            ExObject::from(ExMesh::from(Point { x: 0.0, y: 0.0, z: 0.0 })),
            ExObject::from(ExTransform { x: p[0], y: p[1], z: p[2], rx: 0.0, ry: 0.0, rz: 0.0, sx: 1.0, sy: 1.0, sz: 1.0 }),
            ExObject { u_object: Some(ex_object::UObject::MaterialId(color.to_string())) },
        ]);
    }
    unit.send().await?;
    Ok(())
}

async fn send_label(text: &str, x: f32, y: f32, z: f32, base_id: u64) -> Result<(), Box<dyn std::error::Error>> {
    let mut unit = generate_unit();
    unit.objects.extend(vec![
        ExObject::from(base_id),
        ExObject::from(Tag::new(text).with_offset(ExTransform { x, y, z, rx: 0.0, ry: 0.0, rz: 0.0, sx: 1.0, sy: 1.0, sz: 1.0 })),
    ]);
    unit.send().await?;
    Ok(())
}

fn print_result(name: &str, elapsed: std::time::Duration, param: f32, ground_count: usize, total: usize) {
    let ground_count = ground_count.min(total);
    let non_ground = total - ground_count;
    let us = elapsed.as_micros();
    println!(
        "  {:<20} param={:<5.2}  {:>5}μs  ground={:<6}  non-ground={:<6}  {:.1}%",
        name, param, us, ground_count, non_ground,
        ground_count as f64 / total as f64 * 100.0,
    );
}

// ═══════════════════════════════════════════════════════════════
// 策略 1：Z-直方图 + expand
// ═══════════════════════════════════════════════════════════════
/// 原地交换：调用后 ground 在 [0, n_ground)，剩余在 [n_ground..)
fn histogram_expand(cloud: &mut [[f32; 3]], expand: f32, upside_down: bool) -> (usize, Vec<CldBud>) {
    if cloud.len() < 10 { return (0, vec![]); }
    if upside_down { for p in cloud.iter_mut() { p[2] = -p[2]; } }
    cloud.sort_by(|a, b| a[2].partial_cmp(&b[2]).unwrap());
    let n = cloud.len();
    let z_min = cloud[0][2];
    let z_max = cloud[n - 1][2];
    let z_range = z_max - z_min;
    if z_range < 1e-6 {
        if upside_down { for p in cloud.iter_mut() { p[2] = -p[2]; } }
        return (0, vec![]);
    }

    let num_bins = 128;
    let bin_w = z_range / num_bins as f32;
    let mut bins = vec![0usize; num_bins];
    for p in cloud.iter() {
        let mut b = ((p[2] - z_min) / bin_w) as usize;
        b = b.min(num_bins - 1);
        bins[b] += 1;
    }

    let peak = find_peak_bin(&bins, upside_down);
    let peak_z = z_min + (peak as f32 + 0.5) * bin_w;
    let z_low = peak_z - expand;
    let z_high = peak_z + expand;

    // 找出地面在排序数组中的范围
    let mut start = 0;
    for (i, p) in cloud.iter().enumerate() {
        if p[2] >= z_low { start = i; break; }
    }
    let mut end = n;
    for (i, p) in cloud.iter().enumerate().rev() {
        if p[2] <= z_high { end = i + 1; break; }
    }

    let n_ground = end - start;
    eprintln!(
        "  [histogram_expand] expand={:.3} z=[{:.3}, {:.3}] peak_z={:.3} range=[{:.3}, {:.3}] idx=[{}, {}) n_ground={}",
        expand, z_min, z_max, peak_z, z_low, z_high, start, end, n_ground
    );

    // 原地交换：把 ground [start..end) 换到 [0..n_ground)
    for i in 0..n_ground {
        cloud.swap(start + i, i);
    }

    let buds = make_buds(&cloud[..n_ground]);
    if upside_down { for p in cloud.iter_mut() { p[2] = -p[2]; } }
    (n_ground, buds)
}

// ═══════════════════════════════════════════════════════════════
// 策略 2：从峰值向下扫（threshold）+ 向上扩（expand）
// ═══════════════════════════════════════════════════════════════
/// 从直方图峰值向下扫描，直到 bins 低于 threshold × peak
/// 地面范围 = [下边界, peak_z + expand]
/// ceiling 从顶部下扫（单独找顶部的密集层）
fn peak_down_expand_up(cloud: &mut [[f32; 3]], threshold: f32, expand: f32, num_bins: usize, upside_down: bool) -> (usize, usize) {
    if cloud.len() < 10 { return (0, 0); }
    if upside_down { for p in cloud.iter_mut() { p[2] = -p[2]; } }
    cloud.sort_by(|a, b| a[2].partial_cmp(&b[2]).unwrap());
    let n = cloud.len();
    let z_min = cloud[0][2];
    let z_max = cloud[n - 1][2];
    let z_range = z_max - z_min;
    if z_range < 1e-6 {
        if upside_down { for p in cloud.iter_mut() { p[2] = -p[2]; } }
        return (0, 0);
    }

    let bin_w = z_range / num_bins as f32;
    let mut bins = vec![0usize; num_bins];
    for p in cloud.iter() {
        let mut b = ((p[2] - z_min) / bin_w) as usize;
        b = b.min(num_bins - 1);
        bins[b] += 1;
    }

    let peak = find_peak_bin(&bins, upside_down);
    let peak_count = bins[peak];
    let peak_z = z_min + (peak as f32 + 0.5) * bin_w;
    let t = (peak_count as f32 * threshold).max(1.0) as usize;

    // scan DOWN from peak: first bin below threshold = ground lower bound
    let mut ground_start_bin = 0;
    for i in (0..peak).rev() {
        if bins[i] < t {
            ground_start_bin = i + 1;
            break;
        }
    }
    let z_lower = z_min + ground_start_bin as f32 * bin_w;
    let z_upper = peak_z + expand;

    // find ground indices in sorted array
    let mut ground_start = 0;
    for (i, p) in cloud.iter().enumerate() {
        if p[2] >= z_lower { ground_start = i; break; }
    }
    let mut ground_end = n;
    for (i, p) in cloud.iter().enumerate().rev() {
        if p[2] <= z_upper { ground_end = i + 1; break; }
    }

    let n_ground = if ground_end > ground_start { ground_end - ground_start } else { 0 };
    // swap ground [ground_start..ground_end) to front
    for i in 0..n_ground {
        cloud.swap(ground_start + i, i);
    }

    // ceiling: scan from top bins
    let mut n_ceil = 0;
    if peak + 1 < num_bins {
        let mut ceil_start_bin = num_bins;
        for i in (peak + 1..num_bins).rev() {
            if bins[i] < t {
                ceil_start_bin = i + 1;
                break;
            }
            ceil_start_bin = i;
        }
        if ceil_start_bin < num_bins {
            let z_ceil = z_min + ceil_start_bin as f32 * bin_w;
            for p in cloud[n_ground..].iter().rev() {
                if p[2] >= z_ceil { n_ceil += 1; } else { break; }
            }
        }
    }

    if upside_down { for p in cloud.iter_mut() { p[2] = -p[2]; } }
    (n_ground, n_ceil)
}

// ═══════════════════════════════════════════════════════════════
// 策略 3：RANSAC
// ═══════════════════════════════════════════════════════════════
fn ransac_ground(cloud: &mut [[f32; 3]], distance_threshold: f32, iterations: usize, min_ratio: f32) -> (usize, Vec<CldBud>) {
    let n = cloud.len();
    if n < 10 { return (0, vec![]); }

    let mut best_count = 0usize;
    let mut best_plane = ([0.0f32; 3], [0.0f32; 3]);

    for _ in 0..iterations {
        let idx = select_some(0, n, 3);
        if idx.len() < 3 { continue; }
        let (p1, p2, p3) = (&cloud[idx[0]], &cloud[idx[1]], &cloud[idx[2]]);

        let v1 = [p2[0]-p1[0], p2[1]-p1[1], p2[2]-p1[2]];
        let v2 = [p3[0]-p1[0], p3[1]-p1[1], p3[2]-p1[2]];
        let nx = v1[1]*v2[2] - v1[2]*v2[1];
        let ny = v1[2]*v2[0] - v1[0]*v2[2];
        let nz = v1[0]*v2[1] - v1[1]*v2[0];

        let len = (nx*nx + ny*ny + nz*nz).sqrt();
        if len < 1e-6 { continue; }
        let (nx, ny, nz) = (nx/len, ny/len, nz/len);
        if nz.abs() < 0.7 { continue; }

        let count = cloud.iter().filter(|p| {
            let dx = p[0] - p1[0];
            let dy = p[1] - p1[1];
            let dz = p[2] - p1[2];
            (nx*dx + ny*dy + nz*dz).abs() < distance_threshold
        }).count();

        if count > best_count {
            best_count = count;
            best_plane = ([p1[0], p1[1], p1[2]], [nx, ny, nz]);
        }
    }

    let min_inliers = (n as f32 * min_ratio) as usize;
    if best_count < min_inliers { return (0, vec![]); }

    let (pp, norm) = &best_plane;
    let inlier_mask: Vec<bool> = cloud.iter().map(|p| {
        let dx = p[0] - pp[0];
        let dy = p[1] - pp[1];
        let dz = p[2] - pp[2];
        (norm[0]*dx + norm[1]*dy + norm[2]*dz).abs() < distance_threshold
    }).collect();

    let mut write = 0;
    for read in 0..n {
        if inlier_mask[read] {
            cloud.swap(read, write);
            write += 1;
        }
    }

    let buds = make_buds(&cloud[..write]);
    (write, buds)
}

// ═══════════════════════════════════════════════════════════════
// 策略 4：直方图种子 → 平面拟合生长
// ═══════════════════════════════════════════════════════════════
fn histoseed_grow(cloud: &mut [[f32; 3]], expand: f32, distance: f32, iterations: usize, upside_down: bool) -> (usize, Vec<CldBud>) {
    let n = cloud.len();
    if n < 10 { return (0, vec![]); }
    if upside_down { for p in cloud.iter_mut() { p[2] = -p[2]; } }

    // Step 1: Z-histogram → seed region
    cloud.sort_by(|a, b| a[2].partial_cmp(&b[2]).unwrap());
    let z_min = cloud[0][2];
    let z_max = cloud[n - 1][2];
    let z_range = z_max - z_min;
    if z_range < 1e-6 {
        if upside_down { for p in cloud.iter_mut() { p[2] = -p[2]; } }
        return (0, vec![]);
    }

    let num_bins = 128;
    let bin_w = z_range / num_bins as f32;
    let mut bins = vec![0usize; num_bins];
    for p in cloud.iter() {
        let mut b = ((p[2] - z_min) / bin_w) as usize;
        b = b.min(num_bins - 1);
        bins[b] += 1;
    }
    let peak = find_peak_bin(&bins, upside_down);
    let peak_z = z_min + (peak as f32 + 0.5) * bin_w;
    let z_low = peak_z - expand;
    let z_high = peak_z + expand;

    let mut seed_start = 0;
    for (i, p) in cloud.iter().enumerate() { if p[2] >= z_low { seed_start = i; break; } }
    let mut seed_end = n;
    for (i, p) in cloud.iter().enumerate().rev() { if p[2] <= z_high { seed_end = i + 1; break; } }

    let n_seed = seed_end - seed_start;
    if n_seed < 10 {
        if upside_down { for p in cloud.iter_mut() { p[2] = -p[2]; } }
        return (0, vec![]);
    }

    // Build seed cloud (Vec for easy indexing)
    let seed_cloud: Vec<[f32; 3]> = cloud[seed_start..seed_end].to_vec();

    // Step 2: cheap RANSAC on seed points only → best plane
    let mut best_plane = ([0.0f32; 3], [0.0f32; 3]); // (point_on_plane, normal)
    let mut best_count = 0usize;

    for _ in 0..iterations {
        let idx = select_some(0, n_seed, 3);
        if idx.len() < 3 { continue; }
        let (p1, p2, p3) = (&seed_cloud[idx[0]], &seed_cloud[idx[1]], &seed_cloud[idx[2]]);

        let v1 = [p2[0]-p1[0], p2[1]-p1[1], p2[2]-p1[2]];
        let v2 = [p3[0]-p1[0], p3[1]-p1[1], p3[2]-p1[2]];
        let nx = v1[1]*v2[2] - v1[2]*v2[1];
        let ny = v1[2]*v2[0] - v1[0]*v2[2];
        let nz = v1[0]*v2[1] - v1[1]*v2[0];

        let len = (nx*nx + ny*ny + nz*nz).sqrt();
        if len < 1e-6 { continue; }
        let (nx, ny, nz) = (nx/len, ny/len, nz/len);

        let count = seed_cloud.iter().filter(|p| {
            let dx = p[0] - p1[0];
            let dy = p[1] - p1[1];
            let dz = p[2] - p1[2];
            (nx*dx + ny*dy + nz*dz).abs() < distance
        }).count();

        if count > best_count {
            best_count = count;
            best_plane = (*p1, [nx, ny, nz]);
        }
    }

    if best_count < 3 {
        if upside_down { for p in cloud.iter_mut() { p[2] = -p[2]; } }
        return (0, vec![]);
    }

    // Step 3: grow to full cloud
    let (pp, norm) = &best_plane;
    let inlier_mask: Vec<bool> = cloud.iter().map(|p| {
        let dx = p[0] - pp[0];
        let dy = p[1] - pp[1];
        let dz = p[2] - pp[2];
        (norm[0]*dx + norm[1]*dy + norm[2]*dz).abs() < distance
    }).collect();

    let mut write = 0;
    for read in 0..n {
        if inlier_mask[read] {
            cloud.swap(read, write);
            write += 1;
        }
    }

    let buds = make_buds(&cloud[..write]);
    if upside_down { for p in cloud.iter_mut() { p[2] = -p[2]; } }
    (write, buds)
}

// ═══════════════════════════════════════════════════════════════
// 辅助
// ═══════════════════════════════════════════════════════════════
fn find_peak_bin(bins: &[usize], upside_down: bool) -> usize {
    if upside_down {
        // Z 已取反，地面在 LOW Z 端：从底部向上扫，找第一个超过平均值的 bin
        let avg = bins.iter().sum::<usize>() / bins.len().max(1);
        bins.iter()
            .enumerate()
            .find(|(_, c)| **c > avg)
            .map(|(i, _)| i)
            .unwrap_or(0)
    } else {
        // 正常：取全局最高峰
        bins.iter().enumerate()
            .max_by_key(|(_, c)| *c)
            .map(|(i, _)| i)
            .unwrap_or(0)
    }
}

/// SVD 平面拟合 — 返回 (法向量, d) 满足 n·x + d = 0
fn fit_plane_svd(points: &[[f32; 3]]) -> ([f32; 3], f32) {
    let n = points.len();
    if n < 3 { return ([0.0, 0.0, 1.0], 0.0); }

    let n_f = n as f32;
    let mut mx = 0.0; let mut my = 0.0; let mut mz = 0.0;
    for p in points { mx += p[0]; my += p[1]; mz += p[2]; }
    mx /= n_f; my /= n_f; mz /= n_f;

    let mut cov = Matrix3::zeros();
    for p in points {
        let x = p[0] - mx; let y = p[1] - my; let z = p[2] - mz;
        cov[(0,0)] += x*x; cov[(0,1)] += x*y; cov[(0,2)] += x*z;
        cov[(1,0)] += y*x; cov[(1,1)] += y*y; cov[(1,2)] += y*z;
        cov[(2,0)] += z*x; cov[(2,1)] += z*y; cov[(2,2)] += z*z;
    }
    cov /= n_f;

    let svd = SVD::new(cov, true, false);
    let norm = match svd.v_t {
        Some(vt) => {
            let v = vt.transpose();
            let col = v.column(2);
            [col[0], col[1], col[2]]
        }
        None => [0.0, 0.0, 1.0], // 回退：水平法向量
    };
    let d = -(norm[0] * mx + norm[1] * my + norm[2] * mz);
    (norm, d)
}

/// GPF（Ground Plane Fitting）— 迭代 SVD 平面拟合
fn gpf_ground(cloud: &mut [[f32; 3]], n_lpr: usize, th_seed: f32, th_dist: f32, n_iter: usize, upside_down: bool) -> (usize, Vec<CldBud>) {
    let n = cloud.len();
    if n < 10 { return (0, vec![]); }

    if upside_down {
        for p in cloud.iter_mut() { p[2] = -p[2]; }
    }

    cloud.sort_by(|a, b| a[2].partial_cmp(&b[2]).unwrap());

    // 1. Initial seed: lowest N_LPR points
    let n_lpr = n_lpr.min(n);
    let lpr: f32 = cloud[..n_lpr].iter().map(|p| p[2]).sum::<f32>() / n_lpr as f32;

    let mut mask = vec![false; n];
    let mut seed_count = 0;
    for (i, p) in cloud.iter().enumerate() {
        if p[2] < lpr + th_seed {
            mask[i] = true;
            seed_count += 1;
        }
    }

    if seed_count < 3 {
        if upside_down { for p in cloud.iter_mut() { p[2] = -p[2]; } }
        return (0, vec![]);
    }

    // 2. Iterative SVD plane fitting
    let mut ground_count = seed_count;
    loop {
        let gp: Vec<[f32; 3]> = cloud.iter().enumerate()
            .filter(|(i, _)| mask[*i]).map(|(_, p)| *p).collect();
        let (normal, d) = fit_plane_svd(&gp);

        let mut new_count = 0;
        let mut new_mask = vec![false; n];
        for (i, p) in cloud.iter().enumerate() {
            if (normal[0]*p[0] + normal[1]*p[1] + normal[2]*p[2] + d).abs() < th_dist {
                new_mask[i] = true;
                new_count += 1;
            }
        }

        if new_count <= ground_count { break; }
        mask = new_mask;
        ground_count = new_count;
    }

    // 3. Swap ground to front
    let mut write = 0;
    for read in 0..n {
        if mask[read] { cloud.swap(read, write); write += 1; }
    }

    if upside_down {
        for p in cloud.iter_mut() { p[2] = -p[2]; }
    }

    let buds = make_buds(&cloud[..write]);
    (write, buds)
}

fn make_buds(points: &[[f32; 3]]) -> Vec<CldBud> {
    if points.is_empty() { return vec![]; }
    let mut box3d = Box3D::empty_box();
    let v: Vec<[f32; 3]> = points.to_vec();
    box3d.cloud2box(&v);
    vec![CldBud::new(box3d, 0, "ground".to_string(), 0.9)]
}
