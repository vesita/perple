use std::sync::{Arc, Mutex};

use bevy::prelude::*;

use crate::{Perple, Swapl, cloud::Lifra, optional::{data_loader::DataLoader, visual::{resource::VisResource, interface::draw_cloud::*}}};

pub fn vis() -> Result<(), Box<dyn std::error::Error>> {
    let swapl = Arc::new(Mutex::new(Swapl::new()));
    let mut data_loader = DataLoader::new(
        Arc::clone(&swapl),
        "./data/test".to_string(),
    );
    let _ = data_loader.load()?;
    let mut perple = Perple::new(
        &swapl.lock().unwrap(),
        "./module/color/yolo11n.onnx", // 使用正确的模型路径
        "./config/camera.toml", // 占位符路径
    );
    let _ = perple.run();
    
    App::new()
        .add_plugins(DefaultPlugins)
        .insert_resource(VisResource {
            clouds: Arc::new(Mutex::new(Lifra::new())),
            swapl: Arc::clone(&swapl),
        })
        .add_systems(Startup, setup_point_cloud)
        .add_systems(Update, render_point_cloud)
        .run();
        
    Ok(())
}

fn setup_point_cloud(
    mut commands: Commands,
    resource: Res<VisResource>
) {
    // 创建初始点云数据
    let lifra = {
        let guard = resource.clouds.lock().unwrap();
        guard.clone()
    };
    
    // 添加光源
    commands.spawn((
        PointLight {
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(4.0, 8.0, 4.0),
    ));

    // 添加相机
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(-2.5, 4.5, 9.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
    
    // 绘制点云
    draw_point_cloud(commands, lifra);
}