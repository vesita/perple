use std::time::Instant;

use perple::optional::data_loader::DataLoader;
use perple::swapl::global_swapl;
use perple::utils::random::select_some;
use perple::utils::boxes::Box3D;
use perple::cloud::ground::{
    GroundPickStrategy, create_ground_strategy,
    HistogramExpandStrategy,
};

use nalgebra::{Matrix3, SVD};
use redra_client::{RdraWriter, spawn_sphere, spawn_cube};

/// 单个测试用例的结果
struct BenchResult {
    label: String,
    ground_mask: Vec<bool>,
    elapsed_us: u128,
    n_ground: usize,
    n_total: usize,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();

    println!("=== 地面检测策略对比测试（并行） ===\n");

    // ── 加载数据（不限帧数） ──
    let mut data_loader = DataLoader::new("./data/test".to_string());
    data_loader.set_frame_limit(5);
    data_loader.load().await?;
    data_loader.load_next().await?;

    let swapl = global_swapl();
    let cloud: Vec<[f32; 3]> = {
        let mut stream = swapl.clouds.lock().await;
        stream.read().unwrap_or_default()
    };

    {
        let ds = create_ground_strategy();
        println!("当前默认策略: {}\n", ds.strategy_name());
    }

    let n_total = cloud.len();
    println!("点云总数: {}\n", n_total);

    // ── 构建所有测试用例 ──
    struct TestCase {
        label_fn: Box<dyn Fn() -> String + Send + Sync>,
        run_fn: Box<dyn Fn(&mut [[f32; 3]]) -> (usize, Vec<bool>) + Send + Sync>,
    }

    let mut cases: Vec<TestCase> = Vec::new();

    // 策略 1：Z-直方图 + expand
    for &expand in &[0.05, 0.10, 0.15, 0.20, 0.30] {
        cases.push(TestCase {
            label_fn: Box::new(move || format!("expand={:.2}", expand)),
            run_fn: Box::new(move |c: &mut [[f32; 3]]| histogram_expand(c, expand, upside_down)),
        });
    }

    // 策略 2：峰下扫 + 上扩
    for &threshold in &[0.05, 0.10, 0.15, 0.20] {
        for &expand in &[0.05, 0.10, 0.20] {
            cases.push(TestCase {
                label_fn: Box::new(move || format!("sd={:.2}_ex={:.2}", threshold, expand)),
                run_fn: Box::new(move |c: &mut [[f32; 3]]| {
                    peak_down_expand_up(c, threshold, expand, 128, upside_down)
                }),
            });
        }
    }

    // 策略 3：RANSAC
    for &distance in &[0.3, 0.5] {
        cases.push(TestCase {
            label_fn: Box::new(move || format!("ransac={:.1}", distance)),
            run_fn: Box::new(move |c: &mut [[f32; 3]]| ransac_ground(c, distance, 200, 0.25)),
        });
    }

    // 策略 4：种子+生长
    for &distance in &[0.3, 0.5] {
        for &expand in &[0.10, 0.20] {
            cases.push(TestCase {
                label_fn: Box::new(move || format!("seed_ex={:.2}_d={:.1}", expand, distance)),
                run_fn: Box::new(move |c: &mut [[f32; 3]]| histoseed_grow(c, expand, distance, 100, upside_down)),
            });
        }
    }

    // 策略 5：GPF
    for &n_lpr in &[100, 200] {
        for &th_dist in &[0.2, 0.3, 0.5] {
            cases.push(TestCase {
                label_fn: Box::new(move || format!("gpf_nlpr={}_d={:.1}", n_lpr, th_dist)),
                run_fn: Box::new(move |c: &mut [[f32; 3]]| gpf_ground(c, n_lpr, 0.5, th_dist, upside_down)),
            });
        }
    }

    let n_cases = cases.len();
    println!("共 {} 个测试用例，并行执行...\n", n_cases);

    // ── 并行执行所有测试 ──
    let total_start = Instant::now();

    let results: Vec<BenchResult> = std::thread::scope(|s| {
        let handles: Vec<_> = cases.iter().map(|case| {
            let label = (case.label_fn)();
            let cloud_clone = cloud.clone();
            s.spawn(move || {
                let mut c = cloud_clone;
                let start = Instant::now();
                let (n_ground, ground_mask) = (case.run_fn)(&mut c);
                let elapsed = start.elapsed();
                BenchResult {
                    label,
                    ground_mask,
                    elapsed_us: elapsed.as_micros(),
                    n_ground,
                    n_total,
                }
            })
        }).collect();

        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    let total_elapsed = total_start.elapsed();

    // ── 输出统计表 ──
    println!("┌{:<24}┬{:>8}┬{:>8}┬{:>8}┬{:>7}┐",
        "────────────────────────", "────────", "────────", "────────", "───────");
    println!("│{:<24}│{:>8}│{:>8}│{:>8}│{:>7}│",
        " 策略", "耗时(μs)", "地面", "非地面", "比例");
    println!("├{:<24}┼{:>8}┼{:>8}┼{:>8}┼{:>7}┤",
        "────────────────────────", "────────", "────────", "────────", "───────");

    for r in &results {
        let ng = r.n_ground.min(r.n_total);
        let non_g = r.n_total - ng;
        let pct = ng as f64 / r.n_total as f64 * 100.0;
        println!("│{:<24}│{:>8}│{:>8}│{:>8}│{:>6.1}%│",
            format!(" {}", r.label), r.elapsed_us, ng, non_g, pct);
    }

    println!("└{:<24}┴{:>8}┴{:>8}┴{:>8}┴{:>7}┘",
        "────────────────────────", "────────", "────────", "────────", "───────");
    println!("并行总耗时: {:.1}ms\n", total_elapsed.as_secs_f64() * 1000.0);

    // ── 每个策略写入独立 .rdra 文件 ──
    let output_dir = "output/ground_bench";
    std::fs::create_dir_all(output_dir)?;

    let cloud_step = (cloud.len() / 5000).max(1);
    let sampled_indices: Vec<usize> = (0..cloud.len())
        .filter(|i| i % cloud_step == 0)
        .collect();

    for (idx, r) in results.iter().enumerate() {
        if r.n_ground == 0 {
            println!("  [跳过] {} — 无地面点", r.label);
            continue;
        }

        let mut writer = RdraWriter::new();

        // 写入完整点云（白色 = 非地面，绿色 = 地面）
        let mut n_written_ground = 0usize;
        for (si, &orig_i) in sampled_indices.iter().enumerate() {
            let p = cloud[orig_i];
            let mat = if r.ground_mask[orig_i] { n_written_ground += 1; "green" } else { "white" };
            writer.spawn(
                spawn_sphere(p, 0.03, mat)
                    .id(1_000_000 + si as u64 * 4)
            );
        }

        // 地面点集包围盒
        let ground_pts: Vec<[f32; 3]> = cloud.iter().enumerate()
            .filter(|(i, _)| r.ground_mask[*i])
            .map(|(_, p)| *p)
            .collect();
        let mut box3d = Box3D::empty_box();
        box3d.cloud2box(&ground_pts);
        let verts: Vec<(f32, f32, f32)> = box3d.vertices().iter()
            .map(|v| (v.x, v.y, v.z))
            .collect();
        let tag = format!("{} | {}/{} 地面 | {}μs", r.label, n_written_ground, sampled_indices.len(), r.elapsed_us);
        writer.spawn(
            spawn_cube(verts, "glass")
                .id(2_000_000)
                .tag(tag)
        );

        writer.end_frame();

        let safe_label = r.label.replace(['=', '.', ' '], "_");
        let path = format!("{}/{}.rdra", output_dir, safe_label);
        writer.save(&path)?;
        println!("  [{}] {} → {} ({}/{} 点)", idx + 1, r.label, path, n_written_ground, sampled_indices.len());
    }

    println!("\n完成，共 {} 个 .rdra 文件保存到 {}", results.len(), output_dir);
    Ok(())
}

// ═══════════════════════════════════════════════════════════════
// 策略 1：Z-直方图 + expand
// ═══════════════════════════════════════════════════════════════
fn histogram_expand(cloud: &mut [[f32; 3]], expand: f32, upside_down: bool) -> (usize, Vec<bool>) {
    let n = cloud.len();
    if n < 10 { return (0, vec![false; n]); }
    if upside_down { for p in cloud.iter_mut() { p[2] = -p[2]; } }
    let mut indexed: Vec<(usize, [f32; 3])> = (0..n).zip(cloud.iter().copied()).collect();
    indexed.sort_by(|a, b| a.1[2].partial_cmp(&b.1[2]).unwrap());
    for (i, (_, p)) in indexed.iter().enumerate() { cloud[i] = *p; }

    let z_min = cloud[0][2];
    let z_max = cloud[n - 1][2];
    let z_range = z_max - z_min;
    if z_range < 1e-6 {
        if upside_down { for p in cloud.iter_mut() { p[2] = -p[2]; } }
        return (0, vec![false; n]);
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

    let mut start = 0;
    for (i, p) in cloud.iter().enumerate() {
        if p[2] >= z_low { start = i; break; }
    }
    let mut end = n;
    for (i, p) in cloud.iter().enumerate().rev() {
        if p[2] <= z_high { end = i + 1; break; }
    }

    let n_ground = end - start;
    let mut ground_mask = vec![false; n];
    for i in start..end {
        ground_mask[indexed[i].0] = true;
    }

    if upside_down { for p in cloud.iter_mut() { p[2] = -p[2]; } }
    (n_ground, ground_mask)
}

// ═══════════════════════════════════════════════════════════════
// 策略 2：峰下扫 + 上扩
// ═══════════════════════════════════════════════════════════════
fn peak_down_expand_up(cloud: &mut [[f32; 3]], threshold: f32, expand: f32, num_bins: usize, upside_down: bool) -> (usize, Vec<bool>) {
    let n = cloud.len();
    if n < 10 { return (0, vec![false; n]); }
    let mut indexed: Vec<(usize, [f32; 3])> = (0..n).zip(cloud.iter().copied()).collect();
    if upside_down { for p in cloud.iter_mut() { p[2] = -p[2]; } }
    indexed.sort_by(|a, b| a.1[2].partial_cmp(&b.1[2]).unwrap());
    for (i, (_, p)) in indexed.iter().enumerate() { cloud[i] = *p; }
    let z_min = cloud[0][2];
    let z_max = cloud[n - 1][2];
    let z_range = z_max - z_min;
    if z_range < 1e-6 {
        if upside_down { for p in cloud.iter_mut() { p[2] = -p[2]; } }
        return (0, vec![false; n]);
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

    let mut ground_start_bin = 0;
    for i in (0..peak).rev() {
        if bins[i] < t {
            ground_start_bin = i + 1;
            break;
        }
    }
    let z_lower = z_min + ground_start_bin as f32 * bin_w;
    let z_upper = peak_z + expand;

    let mut ground_start = 0;
    for (i, p) in cloud.iter().enumerate() {
        if p[2] >= z_lower { ground_start = i; break; }
    }
    let mut ground_end = n;
    for (i, p) in cloud.iter().enumerate().rev() {
        if p[2] <= z_upper { ground_end = i + 1; break; }
    }

    let n_ground = if ground_end > ground_start { ground_end - ground_start } else { 0 };
    let mut ground_mask = vec![false; n];
    for i in ground_start..ground_end {
        ground_mask[indexed[i].0] = true;
    }

    if upside_down { for p in cloud.iter_mut() { p[2] = -p[2]; } }
    (n_ground, ground_mask)
}

// ═══════════════════════════════════════════════════════════════
// 策略 3：RANSAC
// ═══════════════════════════════════════════════════════════════
fn ransac_ground(cloud: &mut [[f32; 3]], distance_threshold: f32, iterations: usize, min_ratio: f32) -> (usize, Vec<bool>) {
    let n = cloud.len();
    if n < 10 { return (0, vec![false; n]); }

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
    if best_count < min_inliers { return (0, vec![false; n]); }

    let (pp, norm) = &best_plane;
    let inlier_mask: Vec<bool> = cloud.iter().map(|p| {
        let dx = p[0] - pp[0];
        let dy = p[1] - pp[1];
        let dz = p[2] - pp[2];
        (norm[0]*dx + norm[1]*dy + norm[2]*dz).abs() < distance_threshold
    }).collect();

    let n_ground = inlier_mask.iter().filter(|&&m| m).count();
    (n_ground, inlier_mask)
}

// ═══════════════════════════════════════════════════════════════
// 策略 4：直方图种子 + 平面拟合生长
// ═══════════════════════════════════════════════════════════════
fn histoseed_grow(cloud: &mut [[f32; 3]], expand: f32, distance: f32, iterations: usize, upside_down: bool) -> (usize, Vec<bool>) {
    let n = cloud.len();
    if n < 10 { return (0, vec![false; n]); }
    let mut indexed: Vec<(usize, [f32; 3])> = (0..n).zip(cloud.iter().copied()).collect();
    if upside_down { for p in cloud.iter_mut() { p[2] = -p[2]; } }
    indexed.sort_by(|a, b| a.1[2].partial_cmp(&b.1[2]).unwrap());
    for (i, (_, p)) in indexed.iter().enumerate() { cloud[i] = *p; }
    let z_min = cloud[0][2];
    let z_max = cloud[n - 1][2];
    let z_range = z_max - z_min;
    if z_range < 1e-6 {
        if upside_down { for p in cloud.iter_mut() { p[2] = -p[2]; } }
        return (0, vec![false; n]);
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
        return (0, vec![false; n]);
    }

    let seed_cloud: Vec<[f32; 3]> = cloud[seed_start..seed_end].to_vec();

    let mut best_plane = ([0.0f32; 3], [0.0f32; 3]);
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
        return (0, vec![false; n]);
    }

    let (pp, norm) = &best_plane;
    let inlier_mask: Vec<bool> = cloud.iter().map(|p| {
        let dx = p[0] - pp[0];
        let dy = p[1] - pp[1];
        let dz = p[2] - pp[2];
        (norm[0]*dx + norm[1]*dy + norm[2]*dz).abs() < distance
    }).collect();

    let n_ground = inlier_mask.iter().filter(|&&m| m).count();
    if upside_down { for p in cloud.iter_mut() { p[2] = -p[2]; } }
    (n_ground, inlier_mask)
}

// ═══════════════════════════════════════════════════════════════
// 策略 5：GPF
// ═══════════════════════════════════════════════════════════════
fn gpf_ground(cloud: &mut [[f32; 3]], n_lpr: usize, th_seed: f32, th_dist: f32, upside_down: bool) -> (usize, Vec<bool>) {
    let n = cloud.len();
    if n < 10 { return (0, vec![false; n]); }

    let mut indexed: Vec<(usize, [f32; 3])> = (0..n).zip(cloud.iter().copied()).collect();
    if upside_down { for p in cloud.iter_mut() { p[2] = -p[2]; } }
    indexed.sort_by(|a, b| a.1[2].partial_cmp(&b.1[2]).unwrap());
    for (i, (_, p)) in indexed.iter().enumerate() { cloud[i] = *p; }

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
        return (0, vec![false; n]);
    }

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

    // 将排序后的 mask 转换为原始索引顺序
    let mut ground_mask = vec![false; n];
    for (sorted_i, &is_ground) in mask.iter().enumerate() {
        if is_ground {
            ground_mask[indexed[sorted_i].0] = true;
        }
    }

    let n_ground = ground_mask.iter().filter(|&&m| m).count();
    if upside_down { for p in cloud.iter_mut() { p[2] = -p[2]; } }

    (n_ground, ground_mask)
}

// ═══════════════════════════════════════════════════════════════
// 辅助
// ═══════════════════════════════════════════════════════════════
fn find_peak_bin(bins: &[usize], upside_down: bool) -> usize {
    if upside_down {
        let avg = bins.iter().sum::<usize>() / bins.len().max(1);
        bins.iter()
            .enumerate()
            .find(|(_, c)| **c > avg)
            .map(|(i, _)| i)
            .unwrap_or(0)
    } else {
        bins.iter().enumerate()
            .max_by_key(|(_, c)| *c)
            .map(|(i, _)| i)
            .unwrap_or(0)
    }
}

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
        None => [0.0, 0.0, 1.0],
    };
    let d = -(norm[0] * mx + norm[1] * my + norm[2] * mz);
    (norm, d)
}
