use pcd_rs::DynReader;
use perple::utils::rdra::FrameWriter;
use redra_client::spawn_point;

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

    let mut writer = FrameWriter::new("output/check_axis.db")?;
    writer.begin_frame(0);

    // ── 点云（暖白，语义层） ──
    writer.write_cloud(&points, "point_cloud", 5000);

    // ── 坐标轴（从原点沿各方向延伸 15m） ──
    writer.write_line([0.0, 0.0, 0.0], [15.0, 0.0, 0.0], "axis_x");
    writer.write_line([0.0, 0.0, 0.0], [0.0, 15.0, 0.0], "axis_y");
    writer.write_line([0.0, 0.0, 0.0], [0.0, 0.0, 15.0], "axis_z");

    // ── 轴标签（用 spawn 写入自定义 ShapeBuilder） ──
    writer.spawn(spawn_point([16.0, 0.0, 0.0], "axis_x").id(950_000).tag("X"));
    writer.spawn(spawn_point([0.0, 16.0, 0.0], "axis_y").id(950_001).tag("Y"));
    writer.spawn(spawn_point([0.0, 0.0, 16.0], "axis_z").id(950_002).tag("Z"));

    writer.end_frame();

    writer.save()?;
    println!("已保存到 output/check_axis.db");
    println!("  暖白=点云  红=X  绿=Y  蓝=Z");
    println!("  注意：PCD(Z轴朝上)与Redra(Z轴朝前)坐标系差异，点云可能看起来旋转了90度");
    Ok(())
}
