use std::{sync::{Arc, Mutex}, thread, time};

use bevy::prelude::*;

use crate::{Perple, Swapl, optional::{data_loader::DataLoader, visual::{draw_setup, resource::VisResource}}};

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
    
    thread::sleep(time::Duration::from_secs(5));

    App::new()
        .add_plugins(DefaultPlugins)
        .insert_resource(VisResource {
            swapl: Arc::clone(&swapl),
        })
        .add_systems(Startup, draw_setup)
        .run();
        
    Ok(())
}
