use std::{sync::Arc, thread, time};

use bevy::prelude::*;
use smooth_bevy_cameras::{
    LookTransformPlugin,
    controllers::fps::{FpsCameraPlugin},
};

use crate::{Perple, Swapl, optional::{data_loader::DataLoader, visual::{resource::VisResource, interface::draw::{setup_scene, update_visualization}}}};

pub fn vis() -> Result<(), Box<dyn std::error::Error>> {
    let swapl = Arc::new(Swapl::new());
    let mut data_loader = DataLoader::new(
        Arc::clone(&swapl),
        "./data/test".to_string(),
    );
    let _ = data_loader.load()?;
    let mut perple = Perple::new(
        Arc::clone(&swapl),
        "./module/color/yolo11n.onnx", // 使用正确的模型路径
        "./config/camera.toml", // 占位符路径
    );
    let _ = perple.run();
    
    thread::sleep(time::Duration::from_secs(5));

    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Perple Visualizer".to_string(),
                resizable: true,
                prevent_default_event_handling: false,
                ..default()
            }),
            ..default()
        }))
        .add_plugins(LookTransformPlugin)
        .add_plugins(FpsCameraPlugin::default())
        .insert_resource(VisResource {
            swapl: Arc::clone(&swapl),
            lidar: Arc::clone(&perple.lidar),
        })
        .add_systems(Startup, setup_scene)
        .add_systems(Update, update_visualization)
        .run();
        
    Ok(())
}