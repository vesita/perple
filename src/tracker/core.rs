use std::collections::BTreeMap;
use std::time::{Instant, SystemTime};

use crate::{
    cloud::CldBud,
    cloud::ego_motion::EgoMotion,
    config::fixif,
    swapl::global_swapl,
    tracker::{
        analysis::{analyze_point_cloud_voting_direct, analyze_velocity_clusters_from_snapshot},
        association::{associate, update_object_point_clouds},
        kalman::KalmanConfigCA,
        lifecycle::apply_state_machine,
        object::{TargetClass, TrackStatus, TrackedObject},
        output::Target,
        trick,
    },
    utils::{
        boxes::Box3D,
        stream::{DualBuf, Eap, Stream},
    },
};

pub use super::error::TrackerError;

/// 主跟踪器
pub struct Tracker {
    tar3d: Eap<Stream<Vec<CldBud>>>,
    target: Eap<Stream<Vec<Target>>>,
    /// DualBuf consumer：后融合阶段读取检测阶段写入的体素过滤点云
    clouds_filtered: DualBuf<Vec<[f32; 3]>>,
    next_id: usize,
    tracked_objects: BTreeMap<usize, TrackedObject>,
    max_disappeared: u32,
    min_confidence: f32,
    min_appearances: u32,
    /// 点云投票配置
    use_point_cloud_voting: bool,
    point_cloud_vote_threshold: f32,
    point_cloud_skip_frames: usize,
    point_vel_threshold: f32,
    point_cloud_history_len: usize,
    /// fix_size 配置
    use_fix_size: bool,
    fix_size_frames: usize,
    fix_size_dim_thresh: f32,
    /// k 帧速度观测平均帧数
    kf_avg_frames: usize,
    /// 状态机配置
    floating_to_static_frames: usize,
    moving_speed_threshold: f32,
    voting_consistency_frames: usize,
    // ─── 帧间平滑配置 ────────────────────────────────────────────────────
    use_centroid_smoothing: bool,
    centroid_fc_min: f64,
    centroid_beta: f64,
    use_box_smoothing: bool,
    box_smoothing_alpha: f32,
    vel_smoothing_alpha: f32,
    class_cooldown_frames: u32,
    /// 自运动补偿
    ego_motion: EgoMotion,
    /// 匈牙利算法代价矩阵缓冲区（避免每帧堆分配）
    cost_buf: Vec<Vec<f64>>,
    /// 匈牙利算法方阵缓冲区（避免每帧堆分配）
    sq_buf: Vec<Vec<f64>>,
    // ─── 航迹分级管理 ────────────────────────────────────────────────────
    tentative_max_missed: usize,
    // ─── 轨迹评分 ──────────────────────────────────────────────────────────
    track_score_match_bonus: f64,
    track_score_miss_penalty: f64,
    track_score_confirm_threshold: f64,
    track_score_delete_threshold: f64,
    track_score_output_threshold: f64,
    track_score_max: f64,
    // ─── 卡尔曼滤波器配置 ──────────────────────────────────────────────────
    kalman_config: KalmanConfigCA,
    kf_gate_threshold: f64,
    // ─── 几何后端累计阈值 ──────────────────────────────────────────────────
    geo_pass_threshold: u32,
    geo_fail_threshold: u32,
    geo_speed_threshold: f32,
    /// 几何形状行人判断开关
    use_trick: bool,
}

impl Tracker {
    pub fn new() -> Self {
        let swapl = global_swapl();
        let cfg = &fixif().tracker;
        let kalman_config = KalmanConfigCA {
            dt: 0.04,
            process_noise_pos: cfg.kf_process_noise_pos,
            process_noise_vel: cfg.kf_process_noise_vel,
            process_noise_acc: cfg.kf_process_noise_acc,
            process_noise_size: cfg.kf_process_noise_size,
            measurement_noise_pos: cfg.kf_measurement_noise_pos,
            measurement_noise_vel: cfg.kf_measurement_noise_vel,
            measurement_noise_acc: cfg.kf_measurement_noise_acc,
            measurement_noise_size: cfg.kf_measurement_noise_size,
            initial_covariance_scale: cfg.kf_initial_covariance_scale,
        };
        Self {
            tar3d: swapl.cld_objs.clone(),
            target: swapl.targets.clone(),
            clouds_filtered: swapl.clouds_filtered.clone(),
            next_id: 1,
            tracked_objects: BTreeMap::new(),
            max_disappeared: cfg.max_disappeared,
            min_confidence: cfg.min_confidence,
            min_appearances: cfg.min_appearances,
            use_point_cloud_voting: cfg.use_point_cloud_voting,
            point_cloud_vote_threshold: cfg.point_cloud_vote_threshold,
            point_cloud_skip_frames: cfg.point_cloud_skip_frames,
            point_vel_threshold: cfg.point_vel_threshold,
            point_cloud_history_len: cfg.point_cloud_history_len,
            use_fix_size: cfg.use_fix_size,
            fix_size_frames: cfg.fix_size_frames,
            fix_size_dim_thresh: cfg.fix_size_dim_thresh,
            kf_avg_frames: cfg.kf_avg_frames,
            floating_to_static_frames: cfg.floating_to_static_frames,
            moving_speed_threshold: cfg.moving_speed_threshold,
            voting_consistency_frames: cfg.voting_consistency_frames,
            // ─── 帧间平滑 ────────────────────────────────────────────────
            use_centroid_smoothing: cfg.use_centroid_smoothing,
            centroid_fc_min: cfg.centroid_fc_min,
            centroid_beta: cfg.centroid_beta,
            use_box_smoothing: cfg.use_box_smoothing,
            box_smoothing_alpha: cfg.box_smoothing_alpha,
            vel_smoothing_alpha: cfg.vel_smoothing_alpha,
            class_cooldown_frames: cfg.class_cooldown_frames,
            ego_motion: EgoMotion::new(),
            cost_buf: Vec::new(),
            sq_buf: Vec::new(),
            // ─── 航迹分级管理 ────────────────────────────────────────────
            tentative_max_missed: cfg.tentative_max_missed,
            track_score_match_bonus: cfg.track_score_match_bonus,
            track_score_miss_penalty: cfg.track_score_miss_penalty,
            track_score_confirm_threshold: cfg.track_score_confirm_threshold,
            track_score_delete_threshold: cfg.track_score_delete_threshold,
            track_score_output_threshold: cfg.track_score_output_threshold,
            track_score_max: cfg.track_score_max,
            kalman_config,
            kf_gate_threshold: cfg.kf_gate_threshold,
            geo_pass_threshold: cfg.geo_pass_threshold,
            geo_fail_threshold: cfg.geo_fail_threshold,
            geo_speed_threshold: cfg.geo_speed_threshold,
            use_trick: cfg.use_trick,
        }
    }

    pub fn set_max_disappeared(&mut self, max: u32) {
        self.max_disappeared = max;
    }

    pub fn set_min_confidence(&mut self, confidence: f32) {
        self.min_confidence = confidence;
    }

    pub fn set_min_appearances(&mut self, n: u32) {
        self.min_appearances = n;
    }

    pub async fn run(&mut self) -> Result<(), TrackerError> {
        let _t0 = Instant::now();
        let current_detections = {
            let mut tar3d_guard = self.tar3d.lock().unwrap();
            match tar3d_guard.read() {
                Some(data) => data.into_iter()
                    .filter(|d| d.confidence >= self.min_confidence && d.class_name != "ground" && d.class_name != "wall")
                    .collect::<Vec<_>>(),
                None => Vec::new(),
            }
        };

        let filtered_points = if self.use_point_cloud_voting {
            self.clouds_filtered.consumer().lock().unwrap().clone()
        } else {
            Vec::new()
        };

        let ego_vel = self.ego_motion.update_async().await;
        let has_ego_motion = ego_vel[0].abs() + ego_vel[1].abs() + ego_vel[2].abs() > 0.01;

        let output_targets = self.process_detections(&current_detections, &filtered_points, ego_vel, has_ego_motion);

        {
            let mut target_guard = self.target.lock().unwrap();
            target_guard.write(output_targets)?;
        }

        let n_obj = self.tracked_objects.len();
        let t_total = _t0.elapsed();
        if n_obj > 0 || t_total.as_millis() > 5 {
            log::debug!("[perf] {}obj total={:.1}ms", n_obj, t_total.as_secs_f64() * 1000.0);
        }
        Ok(())
    }

    /// 核心跟踪管线：预测 → 关联 → 修正 → 生命周期 → 分析 → 输出。
    ///
    /// 被 `run()`（在线）和 `run_with_detections()`（bench）共用。
    /// 所有非致命错误内部以 warn 日志记录，不中断管线。
    fn process_detections(
        &mut self,
        current_detections: &[CldBud],
        filtered_points: &[[f32; 3]],
        ego_vel: [f32; 3],
        has_ego_motion: bool,
    ) -> Vec<Target> {
        let now = SystemTime::now();

        // ── 步骤 1: 预测 ──
        for obj in self.tracked_objects.values_mut() {
            let dt = now.duration_since(obj.last_seen)
                .unwrap_or_default()
                .as_secs_f64();
            let dt = dt.clamp(0.001, 1.0);
            if let Err(e) = obj.predict(dt) {
                log::warn!("轨迹 {} 预测失败: {:?}", obj.id, e);
            }
        }

        // ── 步骤 2: 关联 ──
        let (matches, unmatched_detections) = associate(
            &self.tracked_objects,
            current_detections,
            &mut self.cost_buf,
            &mut self.sq_buf,
        );

        // ── 步骤 3: 更新匹配的轨迹 ──
        let mut updated_ids: Vec<usize> = Vec::new();
        for (obj_id, det_idx) in &matches {
            let detection = &current_detections[*det_idx];
            if let Some(obj) = self.tracked_objects.get_mut(obj_id) {
                // 用 AABB 中心而非点云质心修正 KF，使输出位置与 GT/评估约定一致
                let mut centroid = detection.the_box.center_single();
                if self.use_centroid_smoothing {
                    obj.apply_centroid_lpf(&mut centroid, self.centroid_fc_min, self.centroid_beta);
                }
                if let Err(e) = obj.correct(
                    &detection.the_box,
                    detection.class_name.clone(),
                    detection.confidence,
                    centroid,
                ) {
                    log::warn!("轨迹 {} 修正失败: {:?}", obj_id, e);
                    continue;
                }
                let dist = (centroid[0] as f64).powi(2)
                    + (centroid[1] as f64).powi(2)
                    + (centroid[2] as f64).powi(2);
                obj.kalman_filter.adjust_noise_for_confidence(dist.sqrt(), detection.confidence);
                obj.update_score_on_match(self.track_score_match_bonus, self.track_score_max);
                updated_ids.push(*obj_id);
            }
        }

        // ── 步骤 4: 创建新轨迹 ──
        for det_idx in unmatched_detections {
            let detection = &current_detections[det_idx];
            let new_id = self.next_id;
            self.next_id += 1;
            match TrackedObject::new(
                new_id,
                &detection.the_box,
                detection.class_name.clone(),
                detection.confidence,
                detection.the_box.center_single(),
                self.kf_avg_frames,
                self.vel_smoothing_alpha,
                self.kalman_config.clone(),
                self.kf_gate_threshold,
                10,                                      // static_leave_cooldown
                self.floating_to_static_frames as u32,    // floating_settle_cooldown
                self.voting_consistency_frames as u32,    // voting_promote_cooldown
                self.class_cooldown_frames,               // class_change_cooldown
                self.geo_pass_threshold,                  // geo_promote_cooldown
                self.geo_fail_threshold,                  // geo_demote_cooldown
            ) {
                Ok(obj) => {
                    self.tracked_objects.insert(new_id, obj);
                    updated_ids.push(new_id);
                }
                Err(e) => {
                    log::error!("创建新跟踪对象失败：{:?}", e);
                }
            }
        }

        // ── 步骤 5: 未匹配轨迹标记丢失 ──
        let matched_ids: std::collections::HashSet<usize> =
            matches.iter().map(|(id, _)| *id).collect();
        for (id, obj) in &mut self.tracked_objects {
            if !matched_ids.contains(id) {
                obj.on_missed();
                obj.update_score_on_miss(self.track_score_miss_penalty);
            }
        }

        // ── 步骤 5a: 航迹分级 ──
        {
            let mut promote_ids = Vec::new();
            let mut tentative_prune = Vec::new();
            for (id, obj) in &self.tracked_objects {
                match obj.status {
                    TrackStatus::Tentative => {
                        if obj.score >= self.track_score_confirm_threshold {
                            promote_ids.push(*id);
                        } else if obj.score <= self.track_score_delete_threshold
                            && obj.disappeared_count >= self.tentative_max_missed as u32
                        {
                            tentative_prune.push(*id);
                        }
                    }
                    TrackStatus::Confirmed => {}
                }
            }
            for id in &promote_ids {
                if let Some(obj) = self.tracked_objects.get_mut(id) {
                    log::info!("轨迹 {} 晋级: Tentative → Confirmed (score={:.1})", id, obj.score);
                    obj.status = TrackStatus::Confirmed;
                }
            }
            for id in &tentative_prune {
                self.tracked_objects.remove(id);
                log::info!("Tentative 轨迹 {} 因低分被移除 (score=0)", id);
            }
        }

        // ── 步骤 6: 移除丢失超时轨迹 ──
        self.tracked_objects.retain(|_id, obj| {
            obj.disappeared_count < self.max_disappeared
        });

        // ── 步骤 7: 点云历史 + 速度聚类 + 点云投票 ──
        if self.use_point_cloud_voting && !filtered_points.is_empty() {
            update_object_point_clouds(
                &mut self.tracked_objects,
                filtered_points,
                self.point_cloud_history_len,
                200,
            );
        }

        let vel_snapshot: Vec<(usize, [f32; 3])> = self.tracked_objects
            .iter()
            .map(|(id, obj)| (*id, obj.smoothed_velocity()))
            .collect();
        let static_ids = analyze_velocity_clusters_from_snapshot(&vel_snapshot);

        let voting_ids: Vec<usize> = if self.use_point_cloud_voting {
            analyze_point_cloud_voting_direct(
                &self.tracked_objects,
                self.point_cloud_vote_threshold,
                self.point_cloud_skip_frames,
                self.point_vel_threshold,
            )
        } else {
            Vec::new()
        };

        // ── 步骤 9: 状态机 ──
        for (_, obj) in &mut self.tracked_objects {
            let in_static = static_ids.contains(&obj.id);
            let voting_on = voting_ids.contains(&obj.id);
            let spd = if has_ego_motion {
                let vel = obj.smoothed_velocity();
                let dx = vel[0] - ego_vel[0];
                let dy = vel[1] - ego_vel[1];
                let dz = vel[2] - ego_vel[2];
                (dx * dx + dy * dy + dz * dz).sqrt()
            } else {
                obj.speed()
            };
            apply_state_machine(
                obj, in_static, voting_on, spd,
                self.moving_speed_threshold,
            );
        }

        // ── 步骤 9b: trick（几何形状行人判断） ──
        if self.use_trick {
            trick::apply(
                &mut self.tracked_objects,
                self.geo_speed_threshold,
            );
        }

        // ── 步骤 9c: 人物强制 moving（放在 trick 之后，确保 trick 标记的 person 也能被覆盖） ──
        for (_, obj) in &mut self.tracked_objects {
            if obj.class_type == "person" && obj.classification != TargetClass::Moving {
                obj.classification = TargetClass::Moving;
                obj.confirmed_moving = true;
            }
        }

        // ── 步骤 10: 箱体尺寸平滑 ──
        if self.use_box_smoothing {
            for obj in self.tracked_objects.values_mut() {
                if obj.disappeared_count == 0 {
                    obj.apply_box_smoothing(self.box_smoothing_alpha);
                }
            }
        } else if self.use_fix_size {
            for obj in self.tracked_objects.values_mut() {
                obj.apply_fix_size(self.fix_size_frames, self.fix_size_dim_thresh);
            }
        }

        // ── 步骤 11: 生成输出 ──
        let mut output_targets = Vec::new();
        for tracked_id in &updated_ids {
            if let Some(obj) = self.tracked_objects.get(tracked_id) {
                if obj.score < self.track_score_output_threshold {
                    continue;
                }
                let pos = obj.kalman_filter.get_position();
                let vel = obj.smoothed_velocity();
                let spd = obj.speed();
                let z_out = obj.z_ema as f32;

                let ref_box = if obj.disappeared_count > 0 {
                    obj.smoothed_box.as_ref()
                        .or(obj.last_box.as_ref())
                        .cloned()
                        .unwrap_or_else(Box3D::empty_box)
                } else {
                    obj.last_box.as_ref().cloned().unwrap_or_else(Box3D::empty_box)
                };
                let mut predicted_box = Box3D::from_position_and_angles(
                    pos.x as f32, pos.y as f32, z_out,
                    0.0, 0.0, 0.0,
                    ref_box.length, ref_box.width, ref_box.height,
                );
                predicted_box.pose = ref_box.pose;

                let class_str = match obj.classification {
                    TargetClass::Static => "static",
                    TargetClass::Floating => "floating",
                    TargetClass::Moving => "moving",
                    TargetClass::Movable => "movable",
                };

                if *tracked_id <= 5 && obj.appearance_count <= 1 {
                    log::debug!("new obj {}: kf_vel=({:.4},{:.4},{:.4}) speed={:.4} hist_len={}",
                        tracked_id,
                        obj.kalman_filter.get_velocity().x,
                        obj.kalman_filter.get_velocity().y,
                        obj.kalman_filter.get_velocity().z,
                        spd,
                        obj.velocity_history.len(),
                    );
                }

                let mut target = Target {
                    the_box: predicted_box,
                    class_type: obj.class_type.clone(),
                    id: *tracked_id,
                    velocity: vel,
                    speed: spd,
                    is_dynamic: obj.classification == TargetClass::Moving,
                    classification: class_str.to_string(),
                };

                if target.class_type == "ground" {
                    target.classification = "static".to_string();
                    target.is_dynamic = false;
                }
                if obj.classification == TargetClass::Static {
                    target.velocity = [0.0, 0.0, 0.0];
                    target.speed = 0.0;
                }

                output_targets.push(target);
            }
        }

        output_targets
    }

    /// 直接接收检测数据运行跟踪器（bench 用，绕过 swapl）。
    ///
    /// 与 `run()` 逻辑完全一致，但：
    /// - `detections` 直接传入（从 swapl 读取改为参数传递）
    /// - `filtered_points` 用于点云投票（非必须，可传空切片）
    /// - 不自车运动补偿（bench 场景传感器静止）
    /// - 返回 `Vec<Target>` 而非写入 swapl
    pub fn run_with_detections(
        &mut self,
        detections: &[CldBud],
        filtered_points: &[[f32; 3]],
    ) -> Vec<Target> {
        let current_detections: Vec<CldBud> = detections
            .iter()
            .filter(|d| {
                d.confidence >= self.min_confidence
                    && d.class_name != "ground"
                    && d.class_name != "wall"
            })
            .cloned()
            .collect();

        self.process_detections(&current_detections, filtered_points, [0.0; 3], false)
    }

    pub fn get_tracking_count(&self) -> usize {
        self.tracked_objects.len()
    }

    pub fn get_tracked_ids(&self) -> Vec<usize> {
        self.tracked_objects.keys().copied().collect()
    }

    pub fn clear(&mut self) {
        self.tracked_objects.clear();
    }
}

impl Default for Tracker {
    fn default() -> Self {
        Self::new()
    }
}
