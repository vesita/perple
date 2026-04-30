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

    // 点云聚类参数移到claster配置中
    pub claster: ClasterConfig,

    // 地面检测参数（直方图种子 + RANSAC 平面拟合生长）
    pub ground_expand: f32,
    pub ground_ransac_distance: f32,
    pub ground_ransac_iterations: usize,
    pub upside_down: bool,
    pub has_ceiling: bool,

    // 模型路径配置
    pub model_path: String,

    pub camera: CameraConfig,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ClasterConfig {
    pub strategy: String,
    pub merge_patience: f32,
    pub merge_threshold: f32,
    pub voxel_size: f32,
    pub min_points_per_cluster: Option<usize>,
    pub max_points_per_node: Option<usize>,
    pub max_tree_depth: Option<usize>,
    pub use_parallel: bool,
    pub eps_slope: f32,
    pub azimuth_resolution: f32,
    pub elevation_resolution: f32,
    pub cluster_threshold: f32,
    pub downsample_method: String,
    pub gaussian_downsample_rate: f32,
    pub density_weight_alpha: f32,
    pub use_pca_obb: bool,
    pub max_range: f32,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CameraConfig {
    pub intrinsic: [[f32; 3]; 3],
    pub extrinsic: [[f32; 4]; 4],
}


impl Config {
    pub fn new() -> Self {
        let config_path = "config/default.toml";
        
        match fs::read_to_string(config_path) {
            Ok(config_str) => {
                match toml::from_str(&config_str) {
                    Ok(config) => config,
                    Err(e) => {
                        eprintln!("解析配置文件 {} 失败: {}", config_path, e);
                        eprintln!("请检查配置文件格式是否正确");
                        std::process::exit(1);
                    }
                }
            },
            Err(e) => {
                eprintln!("读取配置文件 {} 失败: {}", config_path, e);
                eprintln!("请确保配置文件存在且路径正确");
                std::process::exit(1);
            }
        }
    }

    /// 从TOML字符串增量更新配置
    /// 只会更新TOML中明确指定的字段，其余字段保持原值
    pub fn update_from_toml(&mut self, toml_str: &str) -> Result<(), toml::de::Error> {
        let partial_config: PartialConfig = toml::from_str(toml_str)?;
        
        macro_rules! update_field {
            ($field:ident) => {
                if let Some(value) = partial_config.$field {
                    self.$field = value;
                }
            };
        }

        macro_rules! update_nested_field {
            ($parent:ident, $field:ident) => {
                if let Some(ref $parent) = partial_config.$parent {
                    if let Some(value) = $parent.$field {
                        self.$parent.$field = value;
                    }
                }
            };
        }

        // 添加专门用于claster配置更新的宏
        macro_rules! update_claster_field {
            ($field:ident) => {
                if let Some(ref claster) = partial_config.claster {
                    if let Some(ref value) = claster.$field {
                        self.claster.$field = value.clone();
                    }
                }
            };
        }

        update_field!(stream_capacity);
        update_field!(detections_capacity);
        update_field!(person_class_label);
        update_field!(points_capacity);
        update_field!(resolution);
        update_field!(default_input_width);
        update_field!(default_input_height);
        update_field!(default_confidence_threshold);
        update_field!(default_nms_threshold);
        update_field!(dbscan_min_points);
        update_field!(model_path);

        update_field!(ground_expand);
        update_field!(ground_ransac_distance);
        update_field!(ground_ransac_iterations);
        update_field!(upside_down);
        update_field!(has_ceiling);

        // 使用新的宏来更新claster配置
        update_claster_field!(merge_patience);
        update_claster_field!(merge_threshold);
        update_claster_field!(voxel_size);
        update_claster_field!(use_parallel);
        update_claster_field!(eps_slope);
        update_claster_field!(strategy);
        update_claster_field!(azimuth_resolution);
        update_claster_field!(elevation_resolution);
        update_claster_field!(cluster_threshold);
        update_claster_field!(downsample_method);
        update_claster_field!(gaussian_downsample_rate);
        update_claster_field!(density_weight_alpha);
        update_claster_field!(use_pca_obb);
        update_claster_field!(max_range);

        update_nested_field!(camera, intrinsic);
        update_nested_field!(camera, extrinsic);

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

    pub claster: Option<PartialClasterConfig>,

    pub ground_expand: Option<f32>,
    pub ground_ransac_distance: Option<f32>,
    pub ground_ransac_iterations: Option<usize>,
    pub upside_down: Option<bool>,
    pub has_ceiling: Option<bool>,

    pub model_path: Option<String>,

    pub camera: Option<PartialCameraConfig>,
}

#[derive(Serialize, Deserialize, Debug)]
struct PartialCameraConfig {
    pub intrinsic: Option<[[f32; 3]; 3]>,
    pub extrinsic: Option<[[f32; 4]; 4]>,
}

#[derive(Serialize, Deserialize, Debug)]
struct PartialClasterConfig {
    pub strategy: Option<String>,
    pub merge_patience: Option<f32>,
    pub merge_threshold: Option<f32>,
    pub voxel_size: Option<f32>,
    pub min_points_per_cluster: Option<usize>,
    pub max_points_per_node: Option<usize>,
    pub max_tree_depth: Option<usize>,
    pub use_parallel: Option<bool>,
    pub eps_slope: Option<f32>,
    pub azimuth_resolution: Option<f32>,
    pub elevation_resolution: Option<f32>,
    pub cluster_threshold: Option<f32>,
    pub downsample_method: Option<String>,
    pub gaussian_downsample_rate: Option<f32>,
    pub density_weight_alpha: Option<f32>,
    pub use_pca_obb: Option<bool>,
    pub max_range: Option<f32>,
}
