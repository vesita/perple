use std::{fs, sync::LazyLock};
use serde::{Deserialize, Serialize};

// 调查到一般使用LazyLock和OnceLock替换lazy_static和once_cell
static THE_FIXIF: LazyLock<Config> = LazyLock::new(Config::new);

pub fn fixif() -> &'static Config {
    &THE_FIXIF
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Config {
    pub stream_capacity: usize,
    pub detections_capacity: usize,
    pub person_class_label: String,
    pub points_capacity: usize,
    pub resolution: f32,

    pub default_input_width: usize,
    pub default_input_height: usize,
    pub default_confidence_threshold: f32,
    pub default_nms_threshold: f32,

    pub dbscan_min_points: usize,

    pub camera: CameraConfig,

    pub lidar: LidarConfig,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CameraConfig {
    pub intrinsic: [[f32; 3]; 3],
    pub extrinsic: [[f32; 4]; 4],
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct LidarConfig {
    pub extrinsic: [[f32; 4]; 4],
}

impl Config {
    pub fn new() -> Self {
        let config_path = "config/default.toml";
        let config_str = fs::read_to_string(config_path)
            .expect(&format!("加载{}失败", config_path));
        toml::from_str(&config_str).expect(&format!("{}解析失败", config_str))
    }
    
    pub fn default() -> Self {
        Self::new()
    }

    /// 从TOML字符串增量更新配置
    /// 只会更新TOML中明确指定的字段，其余字段保持原值
    pub fn update_from_toml(&mut self, toml_str: &str) -> Result<(), toml::de::Error> {
        let partial_config: PartialConfig = toml::from_str(toml_str)?;
        
        if let Some(stream_capacity) = partial_config.stream_capacity {
            self.stream_capacity = stream_capacity;
        }
        if let Some(detections_capacity) = partial_config.detections_capacity {
            self.detections_capacity = detections_capacity;
        }
        if let Some(person_class_label) = partial_config.person_class_label {
            self.person_class_label = person_class_label;
        }
        if let Some(points_capacity) = partial_config.points_capacity {
            self.points_capacity = points_capacity;
        }
        if let Some(resolution) = partial_config.resolution {
            self.resolution = resolution;
        }
        if let Some(default_input_width) = partial_config.default_input_width {
            self.default_input_width = default_input_width;
        }
        if let Some(default_input_height) = partial_config.default_input_height {
            self.default_input_height = default_input_height;
        }
        if let Some(default_confidence_threshold) = partial_config.default_confidence_threshold {
            self.default_confidence_threshold = default_confidence_threshold;
        }
        if let Some(default_nms_threshold) = partial_config.default_nms_threshold {
            self.default_nms_threshold = default_nms_threshold;
        }
        if let Some(dbscan_min_points) = partial_config.dbscan_min_points {
            self.dbscan_min_points = dbscan_min_points;
        }
        if let Some(camera) = partial_config.camera {
            if let Some(intrinsic) = camera.intrinsic {
                self.camera.intrinsic = intrinsic;
            }
            if let Some(extrinsic) = camera.extrinsic {
                self.camera.extrinsic = extrinsic;
            }
        }
        if let Some(lidar) = partial_config.lidar {
            if let Some(extrinsic) = lidar.extrinsic {
                self.lidar.extrinsic = extrinsic;
            }
        }

        Ok(())
    }

    /// 从文件中加载增量配置更新
    pub fn update_from_file(&mut self, file_path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let toml_str = fs::read_to_string(file_path)?;
        self.update_from_toml(&toml_str)?;
        Ok(())
    }
}

/// 用于增量更新的部分配置结构体
/// 所有字段都是Option类型，表示它们可能是缺失的
#[derive(Serialize, Deserialize, Debug)]
struct PartialConfig {
    pub stream_capacity: Option<usize>,
    pub detections_capacity: Option<usize>,
    pub person_class_label: Option<String>,
    pub points_capacity: Option<usize>,
    pub resolution: Option<f32>,

    pub default_input_width: Option<usize>,
    pub default_input_height: Option<usize>,
    pub default_confidence_threshold: Option<f32>,
    pub default_nms_threshold: Option<f32>,

    pub dbscan_min_points: Option<usize>,

    pub camera: Option<PartialCameraConfig>,

    pub lidar: Option<PartialLidarConfig>,
}

#[derive(Serialize, Deserialize, Debug)]
struct PartialCameraConfig {
    pub intrinsic: Option<[[f32; 3]; 3]>,
    pub extrinsic: Option<[[f32; 4]; 4]>,
}

#[derive(Serialize, Deserialize, Debug)]
struct PartialLidarConfig {
    pub extrinsic: Option<[[f32; 4]; 4]>,
}