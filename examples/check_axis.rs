use pcd_rs::DynReader;
use redra_client::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();

    // ── 读取点云 ──
    let mut reader = DynReader::open("data/lidar/000000.pcd")?;
    let mut points = Vec::new();
    while let Some(result) = reader.next() {
        if let Ok(point) = result {
            if let Some(coords) = point.to_xyz() {
                points.push(coords);
            }
        }
    }
    println!("点云: {} 点", points.len());

    // ── 下采样（最多 5000 点） ──
    let step = (points.len() / 5000).max(1);
    let sampled: Vec<[f32; 3]> = points.iter()
        .enumerate()
        .filter(|(i, _)| i % step == 0)
        .map(|(_, p)| *p)
        .collect();
    println!("采样: {} 点 (1/{})", sampled.len(), step);

    let mut writer = RdraWriter::new();

    // ── 点云（青色） ──
    for (i, p) in sampled.iter().enumerate() {
        writer.spawn(spawn_sphere(*p, 0.05, "cyan").id(1_000_000 + i as u64 * 4));
    }

    // ── 坐标轴（从原点沿各方向延伸 15m） ──
    // 红色线 = X轴，绿色线 = Y轴，蓝色线 = Z轴
    writer.spawn(spawn_line([0.0, 0.0, 0.0], [15.0, 0.0, 0.0], "red").id(1_500_000));
    writer.spawn(spawn_line([0.0, 0.0, 0.0], [0.0, 15.0, 0.0], "green").id(1_500_001));
    writer.spawn(spawn_line([0.0, 0.0, 0.0], [0.0, 0.0, 15.0], "blue").id(1_500_002));

    // ── 轴标签 ──
    writer.spawn(spawn_point([16.0, 0.0, 0.0], "red").id(1_600_000).tag("X"));
    writer.spawn(spawn_point([0.0, 16.0, 0.0], "green").id(1_600_001).tag("Y"));
    writer.spawn(spawn_point([0.0, 0.0, 16.0], "blue").id(1_600_002).tag("Z"));

    writer.end_frame();

    writer.save("output/check_axis.rdra")?;
    println!("已保存到 output/check_axis.rdra");
    println!("  青色=点云  红=X  绿=Y  蓝=Z");
    println!("  注意：PCD(Z轴朝上)与Redra(Z轴朝前)坐标系差异，点云可能看起来旋转了90度");
    Ok(())
}
