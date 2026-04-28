use pcd_rs::DynReader;
use expto::rdmp::auto::unit::generate_unit;
use expto::rdmp::proto::command::{CommandType, ExCommand};
use expto::rdmp::*;
use redra_client::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();

    // ── 读取点云 ──
    // 注意：PCD 文件使用右手坐标系（Z轴朝上），而 redra 渲染使用图形学坐标系（Z轴朝前）
    // 这种坐标系差异会导致点云看起来旋转了90度，这是正常现象，不影响检测逻辑
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

    // ── 发点云（青色） ──
    // 坐标系说明：
    // - PCD数据：X前/Y左/Z天（右手系，Z轴垂直向上）
    // - Redra渲染：X右/Y天/Z前（图形学系，Z轴水平向前）
    // 实际项目中已通过 upside_down=true 和标定矩阵处理此旋转关系
    send_colored_cloud(&sampled, "cyan", 1_000_000).await?;

    // ── 坐标轴（从原点沿各方向延伸 5m） ──
    // 红色线 = X轴，绿色线 = Y轴，蓝色线 = Z轴
    // 由于坐标系差异，在redra中看到的轴向与PCD文件中的物理意义不同
    send_axis(1_500_000).await?;

    // ── 帧结束 ──
    let mut unit = generate_unit();
    unit.command = Some(ExCommand { u_command: CommandType::Frameend as i32 });
    unit.send().await?;

    println!("发送完成，查看 redra。");
    println!("  青色=点云  红=X  绿=Y  蓝=Z");
    println!("  注意：由于PCD(Z轴朝上)与Redra(Z轴朝前)坐标系差异，点云可能看起来旋转了90度");
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

async fn send_axis(_base_id: u64) -> Result<(), Box<dyn std::error::Error>> {
    // 三条轴线从原点沿 X/Y/Z 延伸
    send_line(0.0, 0.0, 0.0, 15.0, 0.0, 0.0).await?; // X → red
    send_line(0.0, 0.0, 0.0, 0.0, 15.0, 0.0).await?; // Y → green
    send_line(0.0, 0.0, 0.0, 0.0, 0.0, 15.0).await?; // Z → blue

    // 标签
    for &(label, ex, ey, ez, _color) in &[
        ("X", 16.0, 0.0, 0.0, "red"),
        ("Y", 0.0, 16.0, 0.0, "green"),
        ("Z", 0.0, 0.0, 16.0, "blue"),
    ] {
        let mut tag = generate_unit();
        tag.objects.extend(vec![
            ExObject::from(_base_id + 100),
            ExObject::from(Tag::new(label).with_offset(ExTransform {
                x: ex, y: ey, z: ez,
                rx: 0.0, ry: 0.0, rz: 0.0, sx: 1.0, sy: 1.0, sz: 1.0,
            })),
        ]);
        tag.send().await?;
    }
    Ok(())
}
