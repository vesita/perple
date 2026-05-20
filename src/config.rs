mod partial;

use std::{fs, sync::OnceLock};
use serde::{Deserialize, Serialize};

use self::partial::PartialConfig;

static THE_FIXIF: OnceLock<Config> = OnceLock::new();

/// Initialize config before first access to `fixif()`.
/// Used by eval/bench tools that need to override config at runtime.
pub fn init_config(config: Config) {
    THE_FIXIF.set(config).expect("config already initialized");
}

pub fn fixif() -> &'static Config {
    THE_FIXIF.get_or_init(|| {
        let config_path = std::env::var("PERPLE_CONFIG_PATH")
            .unwrap_or_else(|_| "config/default.toml".to_string());
        Config::from_file(&config_path)
    })
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Config {
    pub stream_capacity: usize,
    pub detections_capacity: usize,
    pub person_class_label: String,
    pub points_capacity: usize,
    pub max_range: f32,
    pub min_range: f32,

    pub default_input_width: usize,
    pub default_input_height: usize,
    pub default_confidence_threshold: f32,
    pub default_nms_threshold: f32,

    // 点云聚类参数
    pub cluster: ClusterConfig,

    // 墙体检测参数
    pub wall_strategy: String,
    pub wall_distance: f32,
    pub wall_iterations: usize,
    pub wall_max_walls: usize,
    pub wall_eps: f32,
    pub wall_min_pts: usize,
    pub wall_min_z_span: f32,
    pub wall_angle_tolerance: f32,

    // 地面检测参数
    pub ground_strategy: String,
    pub ground_expand: f32,
    pub ground_ransac_distance: f32,
    pub ground_ransac_iterations: usize,
    pub upside_down: bool,
    pub has_ceiling: bool,

    // 模型路径配置
    pub model_path: String,

    pub camera: CameraConfig,

    // 跟踪器参数
    pub tracker: TrackerConfig,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ClusterConfig {
    pub strategy: String,
    pub merge_patience: f32,
    pub voxel_size: f32,
    pub min_points_per_cluster: Option<usize>,
    pub max_points_per_node: Option<usize>,
    pub max_tree_depth: Option<usize>,
    pub eps_slope: f32,
    pub azimuth_resolution: f32,
    pub elevation_resolution: f32,
    pub cluster_threshold: f32,
    pub downsample_method: String,
    pub gaussian_downsample_rate: f32,
    pub density_weight_alpha: f32,
    pub max_range: f32,
    pub ceiling_filter: bool,
    pub ceiling_height: f32,
    pub denoise_radius: f32,
    pub denoise_min_pts: usize,
    // 剪叶聚类 (prune_qt) 参数
    pub min_occ: usize,
    // 自适应深度分裂（根据距离动态调整四叉树分辨率）
    pub adaptive_depth: bool,
    pub adaptive_res0: f32,
    pub adaptive_r0: f32,
    pub adaptive_k: f32,
    pub adaptive_global_max_depth: usize,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CameraConfig {
    pub intrinsic: [[f32; 3]; 3],
    pub extrinsic: [[f32; 4]; 4],
    /// OpenCV 畸变系数 [k1, k2, p1, p2, k3]，None 表示无畸变
    pub dist_coeffs: Option<[f32; 5]>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TrackerConfig {
    pub max_disappeared: u32,
    pub min_confidence: f32,
    pub min_appearances: u32,
    pub use_point_cloud_voting: bool,
    pub point_cloud_vote_threshold: f32,
    pub point_cloud_skip_frames: usize,
    pub point_vel_threshold: f32,
    pub point_cloud_history_len: usize,
    pub use_fix_size: bool,
    pub fix_size_frames: usize,
    pub fix_size_dim_thresh: f32,
    pub kf_avg_frames: usize,
    pub floating_to_static_frames: usize,
    pub moving_speed_threshold: f32,
    pub voting_consistency_frames: usize,
    // ─── 帧间平滑参数 ────────────────────────────────────────────────────
    pub use_centroid_smoothing: bool,
    pub centroid_fc_min: f64,
    pub centroid_beta: f64,
    pub use_box_smoothing: bool,
    pub box_smoothing_alpha: f32,
    pub vel_smoothing_alpha: f32,
    pub class_cooldown_frames: u32,
    // ─── 航迹分级管理 ────────────────────────────────────────────────────
    pub confirmation_frames: usize,
    pub tentative_max_missed: usize,
    // ─── 轨迹评分 ──────────────────────────────────────────────────────────
    pub track_score_match_bonus: f64,
    pub track_score_miss_penalty: f64,
    pub track_score_confirm_threshold: f64,
    pub track_score_delete_threshold: f64,
    pub track_score_output_threshold: f64,
    pub track_score_max: f64,
    // ─── 卡尔曼滤波器参数（9D CA 模型） ────────────────────────────────────
    pub kf_process_noise_pos: f64,
    pub kf_process_noise_vel: f64,
    pub kf_process_noise_acc: f64,
    pub kf_process_noise_size: f64,
    pub kf_measurement_noise_pos: f64,
    pub kf_measurement_noise_vel: f64,
    pub kf_measurement_noise_acc: f64,
    pub kf_measurement_noise_size: f64,
    pub kf_initial_covariance_scale: f64,
    /// 新息门控阈值（马氏距离），超过则降级为位置-only 修正
    pub kf_gate_threshold: f64,
    /// 几何后端连续通过帧数阈值（通过达到此值标记为 person）
    pub geo_pass_threshold: u32,
    /// 几何后端连续失败帧数阈值（失败达到此值回退为 obstacle）
    pub geo_fail_threshold: u32,
    /// 几何后端速度激活阈值（m/s），速度超过此值直接标记为 person，与几何判断 OR
    pub geo_speed_threshold: f32,
    /// 几何形状行人判断（trick）开关
    pub use_trick: bool,
}


impl Config {
    pub fn new() -> Self {
        Self::from_file("config/default.toml")
    }

    /// 从 TOML 文件加载配置
    pub fn from_file(path: &str) -> Self {
        match fs::read_to_string(path) {
            Ok(config_str) => {
                match toml::from_str(&config_str) {
                    Ok(config) => config,
                    Err(e) => {
                        eprintln!("解析配置文件 {} 失败: {}", path, e);
                        eprintln!("请检查配置文件格式是否正确");
                        std::process::exit(1);
                    }
                }
            },
            Err(e) => {
                eprintln!("读取配置文件 {} 失败: {}", path, e);
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

        // 添加专门用于cluster配置更新的宏
        macro_rules! update_cluster_field {
            ($field:ident) => {
                if let Some(ref cluster) = partial_config.cluster {
                    if let Some(ref value) = cluster.$field {
                        self.cluster.$field = value.clone();
                    }
                }
            };
        }

        update_field!(stream_capacity);
        update_field!(detections_capacity);
        update_field!(person_class_label);
        update_field!(points_capacity);
        update_field!(max_range);
        update_field!(min_range);
        update_field!(default_input_width);
        update_field!(default_input_height);
        update_field!(default_confidence_threshold);
        update_field!(default_nms_threshold);
        update_field!(model_path);

        update_field!(wall_strategy);
        update_field!(wall_distance);
        update_field!(wall_iterations);
        update_field!(wall_max_walls);
        update_field!(wall_eps);
        update_field!(wall_min_pts);
        update_field!(wall_min_z_span);
        update_field!(wall_angle_tolerance);

        update_field!(ground_strategy);
        update_field!(ground_expand);
        update_field!(ground_ransac_distance);
        update_field!(ground_ransac_iterations);
        update_field!(upside_down);
        update_field!(has_ceiling);

        // 使用新的宏来更新cluster配置
        update_cluster_field!(merge_patience);
        update_cluster_field!(voxel_size);
        update_cluster_field!(eps_slope);
        update_cluster_field!(strategy);
        update_cluster_field!(azimuth_resolution);
        update_cluster_field!(elevation_resolution);
        update_cluster_field!(cluster_threshold);
        update_cluster_field!(downsample_method);
        update_cluster_field!(gaussian_downsample_rate);
        update_cluster_field!(density_weight_alpha);
        update_cluster_field!(max_range);
        update_cluster_field!(ceiling_filter);
        update_cluster_field!(ceiling_height);
        update_cluster_field!(denoise_radius);
        update_cluster_field!(denoise_min_pts);
        update_cluster_field!(min_occ);
        update_cluster_field!(adaptive_depth);
        update_cluster_field!(adaptive_res0);
        update_cluster_field!(adaptive_r0);
        update_cluster_field!(adaptive_k);
        update_cluster_field!(adaptive_global_max_depth);
        // Option-typed fields — macro destructures to inner type, re-wrap
        if let Some(ref cluster) = partial_config.cluster {
            if let Some(value) = cluster.min_points_per_cluster {
                self.cluster.min_points_per_cluster = Some(value);
            }
            if let Some(value) = cluster.max_points_per_node {
                self.cluster.max_points_per_node = Some(value);
            }
            if let Some(value) = cluster.max_tree_depth {
                self.cluster.max_tree_depth = Some(value);
            }
        }

        update_nested_field!(camera, intrinsic);
        update_nested_field!(camera, extrinsic);
        if let Some(ref camera) = partial_config.camera {
            if let Some(value) = camera.dist_coeffs {
                self.camera.dist_coeffs = Some(value);
            }
        }

        // tracker 配置更新
        if let Some(ref tracker) = partial_config.tracker {
            macro_rules! update_tracker {
                ($field:ident) => {
                    if let Some(value) = tracker.$field {
                        self.tracker.$field = value;
                    }
                };
            }
            update_tracker!(max_disappeared);
            update_tracker!(min_confidence);
            update_tracker!(min_appearances);
            update_tracker!(use_point_cloud_voting);
            update_tracker!(point_cloud_vote_threshold);
            update_tracker!(point_cloud_skip_frames);
            update_tracker!(point_vel_threshold);
            update_tracker!(point_cloud_history_len);
            update_tracker!(use_fix_size);
            update_tracker!(fix_size_frames);
            update_tracker!(fix_size_dim_thresh);
            update_tracker!(kf_avg_frames);
            update_tracker!(floating_to_static_frames);
            update_tracker!(moving_speed_threshold);
            update_tracker!(voting_consistency_frames);
            update_tracker!(use_centroid_smoothing);
            update_tracker!(centroid_fc_min);
            update_tracker!(centroid_beta);
            update_tracker!(use_box_smoothing);
            update_tracker!(box_smoothing_alpha);
            update_tracker!(vel_smoothing_alpha);
            update_tracker!(class_cooldown_frames);
            update_tracker!(confirmation_frames);
            update_tracker!(tentative_max_missed);
            update_tracker!(track_score_match_bonus);
            update_tracker!(track_score_miss_penalty);
            update_tracker!(track_score_confirm_threshold);
            update_tracker!(track_score_delete_threshold);
            update_tracker!(track_score_output_threshold);
            update_tracker!(track_score_max);
            update_tracker!(kf_process_noise_pos);
            update_tracker!(kf_process_noise_vel);
            update_tracker!(kf_process_noise_acc);
            update_tracker!(kf_process_noise_size);
            update_tracker!(kf_measurement_noise_pos);
            update_tracker!(kf_measurement_noise_vel);
            update_tracker!(kf_measurement_noise_acc);
            update_tracker!(kf_measurement_noise_size);
            update_tracker!(kf_initial_covariance_scale);
            update_tracker!(kf_gate_threshold);
            update_tracker!(geo_pass_threshold);
            update_tracker!(geo_fail_threshold);
            update_tracker!(geo_speed_threshold);
            update_tracker!(use_trick);
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
