pub const STREAM_CAPACITY: usize = 16;  // 减小容量以避免栈溢出
pub const DETECTIONS_CAPACITY: usize = 16;
pub const PERSON_CLASS_LABEL: &str = "person";
pub const POINTS_CAPACITY: usize = 16384;
pub const RESOLUTION: f32 = 0.07; // 米

// 目标检测超参数配置
pub const DEFAULT_INPUT_WIDTH: usize = 640;
pub const DEFAULT_INPUT_HEIGHT: usize = 640;
pub const DEFAULT_CONFIDENCE_THRESHOLD: f32 = 0.6;
pub const DEFAULT_NMS_THRESHOLD: f32 = 0.7;

// DBSCAN聚类算法参数
pub const DBSCAN_MIN_POINTS: usize = 3;

use nalgebra::{Matrix3, Matrix4};
use std::fs;

#[derive(serde::Deserialize)]
struct CameraConfig {
    intrinsic: [f32; 9],
    extrinsic: [f32; 16],
}

/// 从TOML文件加载相机参数
/// 
/// # 参数
/// * `config_path` - 配置文件路径
/// 
/// # 返回值
/// 返回(内参矩阵, 外参矩阵)元组
pub fn load_camera_config(config_path: &str) -> Result<(Matrix3<f32>, Matrix4<f32>), Box<dyn std::error::Error>> {
    let contents = fs::read_to_string(config_path)?;
    let config: CameraConfig = toml::from_str(&contents)?;
    
    let intrinsic = Matrix3::from_row_slice(&config.intrinsic);
    let extrinsic = Matrix4::from_row_slice(&config.extrinsic);
    
    Ok((intrinsic, extrinsic))
}