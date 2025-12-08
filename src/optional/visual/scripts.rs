
use bevy::prelude::*;
use smooth_bevy_cameras::{
    LookTransformPlugin,
    controllers::fps::{FpsCameraPlugin},
};

use crate::{Perple, optional::{data_loader::DataLoader, visual::{interface::draw::{setup_scene, update_visualization}}}};

pub fn vis() -> Result<(), Box<dyn std::error::Error>> {
    let mut data_loader = DataLoader::new("./data/test".to_string());
    let _ = data_loader.load()?;
    let mut perple = Perple::new();
    let _ = perple.run();

    // 启动Bevy应用
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(FpsCameraPlugin::default())
        .add_plugins(LookTransformPlugin)
        .add_systems(Startup, setup_scene)
        .add_systems(Update, update_visualization)
        .run();

    Ok(())
}