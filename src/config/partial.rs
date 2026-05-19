/// 用于增量更新的部分配置结构体
/// 所有字段都是Option类型，表示它们可能是缺失的
#[derive(serde::Serialize, serde::Deserialize, Debug)]
pub(crate) struct PartialConfig {
    pub stream_capacity: Option<usize>,
    pub detections_capacity: Option<usize>,
    pub person_class_label: Option<String>,
    pub points_capacity: Option<usize>,
    pub max_range: Option<f32>,
    pub min_range: Option<f32>,

    pub default_input_width: Option<usize>,
    pub default_input_height: Option<usize>,
    pub default_confidence_threshold: Option<f32>,
    pub default_nms_threshold: Option<f32>,

    pub cluster: Option<PartialClusterConfig>,

    pub wall_strategy: Option<String>,
    pub wall_distance: Option<f32>,
    pub wall_iterations: Option<usize>,
    pub wall_max_walls: Option<usize>,
    pub wall_eps: Option<f32>,
    pub wall_min_pts: Option<usize>,
    pub wall_min_z_span: Option<f32>,
    pub wall_angle_tolerance: Option<f32>,

    pub ground_strategy: Option<String>,
    pub ground_expand: Option<f32>,
    pub ground_ransac_distance: Option<f32>,
    pub ground_ransac_iterations: Option<usize>,
    pub upside_down: Option<bool>,
    pub has_ceiling: Option<bool>,

    pub model_path: Option<String>,

    pub camera: Option<PartialCameraConfig>,

    pub tracker: Option<PartialTrackerConfig>,
}

#[derive(serde::Serialize, serde::Deserialize, Debug)]
pub(crate) struct PartialCameraConfig {
    pub intrinsic: Option<[[f32; 3]; 3]>,
    pub extrinsic: Option<[[f32; 4]; 4]>,
    pub dist_coeffs: Option<[f32; 5]>,
}

#[derive(serde::Serialize, serde::Deserialize, Debug)]
pub(crate) struct PartialTrackerConfig {
    pub max_disappeared: Option<u32>,
    pub min_confidence: Option<f32>,
    pub min_appearances: Option<u32>,
    pub use_point_cloud_voting: Option<bool>,
    pub point_cloud_vote_threshold: Option<f32>,
    pub point_cloud_skip_frames: Option<usize>,
    pub point_vel_threshold: Option<f32>,
    pub point_cloud_history_len: Option<usize>,
    pub use_fix_size: Option<bool>,
    pub fix_size_frames: Option<usize>,
    pub fix_size_dim_thresh: Option<f32>,
    pub kf_avg_frames: Option<usize>,
    pub floating_to_static_frames: Option<usize>,
    pub moving_speed_threshold: Option<f32>,
    pub voting_consistency_frames: Option<usize>,
    pub use_centroid_smoothing: Option<bool>,
    pub centroid_fc_min: Option<f64>,
    pub centroid_beta: Option<f64>,
    pub use_box_smoothing: Option<bool>,
    pub box_smoothing_alpha: Option<f32>,
    pub vel_smoothing_alpha: Option<f32>,
    pub class_cooldown_frames: Option<u32>,
    pub confirmation_frames: Option<usize>,
    pub tentative_max_missed: Option<usize>,
    pub track_score_match_bonus: Option<f64>,
    pub track_score_miss_penalty: Option<f64>,
    pub track_score_confirm_threshold: Option<f64>,
    pub track_score_delete_threshold: Option<f64>,
    pub track_score_output_threshold: Option<f64>,
    pub track_score_max: Option<f64>,
    pub kf_process_noise_pos: Option<f64>,
    pub kf_process_noise_vel: Option<f64>,
    pub kf_process_noise_acc: Option<f64>,
    pub kf_process_noise_size: Option<f64>,
    pub kf_measurement_noise_pos: Option<f64>,
    pub kf_measurement_noise_vel: Option<f64>,
    pub kf_measurement_noise_acc: Option<f64>,
    pub kf_measurement_noise_size: Option<f64>,
    pub kf_initial_covariance_scale: Option<f64>,
    pub kf_gate_threshold: Option<f64>,
    pub geo_pass_threshold: Option<u32>,
    pub geo_fail_threshold: Option<u32>,
    pub geo_speed_threshold: Option<f32>,
}

#[derive(serde::Serialize, serde::Deserialize, Debug)]
pub(crate) struct PartialClusterConfig {
    pub strategy: Option<String>,
    pub merge_patience: Option<f32>,
    pub voxel_size: Option<f32>,
    pub min_points_per_cluster: Option<usize>,
    pub max_points_per_node: Option<usize>,
    pub max_tree_depth: Option<usize>,
    pub eps_slope: Option<f32>,
    pub azimuth_resolution: Option<f32>,
    pub elevation_resolution: Option<f32>,
    pub cluster_threshold: Option<f32>,
    pub downsample_method: Option<String>,
    pub gaussian_downsample_rate: Option<f32>,
    pub density_weight_alpha: Option<f32>,
    pub max_range: Option<f32>,
    pub ceiling_filter: Option<bool>,
    pub ceiling_height: Option<f32>,
    pub denoise_radius: Option<f32>,
    pub denoise_min_pts: Option<usize>,
    pub min_occ: Option<usize>,
    pub adaptive_depth: Option<bool>,
    pub adaptive_res0: Option<f32>,
    pub adaptive_r0: Option<f32>,
    pub adaptive_beta: Option<f32>,
    pub adaptive_global_max_depth: Option<usize>,
}
