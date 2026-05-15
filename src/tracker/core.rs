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
        lifecycle::apply_state_machine,
        object::{TargetClass, TrackStatus, TrackedObject},
        output::Target,
        trick,
    },
    utils::{
        boxes::Box3D,
        stream::{DualBuf, Eap, Stream, StreamError},
    },
};

/// Tracker 模块的错误类型
#[derive(Debug)]
pub enum TrackerError {
    StreamError(StreamError),
    PoisonError(String),
    KalmanError(adskalman::Error),
    AssociationError(String),
}

impl From<StreamError> for TrackerError {
    fn from(error: StreamError) -> Self {
        TrackerError::StreamError(error)
    }
}

impl<T> From<std::sync::PoisonError<T>> for TrackerError {
    fn from(_error: std::sync::PoisonError<T>) -> Self {
        TrackerError::PoisonError("线程锁中毒".to_string())
    }
}

impl From<adskalman::Error> for TrackerError {
    fn from(error: adskalman::Error) -> Self {
        TrackerError::KalmanError(error)
    }
}

impl std::fmt::Display for TrackerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TrackerError::StreamError(e) => write!(f, "流错误：{}", e),
            TrackerError::PoisonError(e) => write!(f, "线程锁中毒：{}", e),
            TrackerError::KalmanError(e) => write!(f, "卡尔曼滤波错误：{:?}", e),
            TrackerError::AssociationError(e) => write!(f, "数据关联错误：{}", e),
        }
    }
}

impl std::error::Error for TrackerError {}

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
    confirmation_frames: usize,
    tentative_max_missed: usize,
    // ─── 轨迹评分 ──────────────────────────────────────────────────────────
    track_score_match_bonus: f64,
    track_score_miss_penalty: f64,
    track_score_confirm_threshold: f64,
    track_score_delete_threshold: f64,
    track_score_output_threshold: f64,
    track_score_max: f64,
}

impl Tracker {
    pub fn new() -> Self {
        let swapl = global_swapl();
        let cfg = &fixif().tracker;
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
            confirmation_frames: cfg.confirmation_frames,
            tentative_max_missed: cfg.tentative_max_missed,
            track_score_match_bonus: cfg.track_score_match_bonus,
            track_score_miss_penalty: cfg.track_score_miss_penalty,
            track_score_confirm_threshold: cfg.track_score_confirm_threshold,
            track_score_delete_threshold: cfg.track_score_delete_threshold,
            track_score_output_threshold: cfg.track_score_output_threshold,
            track_score_max: cfg.track_score_max,
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

        let now = SystemTime::now();

        // 步骤 1: 对所有轨迹做预测
        let _t_pred = Instant::now();
        for obj in self.tracked_objects.values_mut() {
            let dt = now.duration_since(obj.last_seen)
                .unwrap_or_default()
                .as_secs_f64();
            let dt = dt.clamp(0.001, 1.0);
            obj.predict(dt)?;
        }

        // 步骤 2: 关联（复用代价矩阵缓冲区）
        let _t_assoc = Instant::now();
        let (matches, unmatched_detections) = associate(
            &self.tracked_objects,
            &current_detections,
            &mut self.cost_buf,
            &mut self.sq_buf,
        );

        // 步骤 3: 更新匹配的轨迹（matches 内含实际 obj_id）
        let _t_update = Instant::now();
        let mut updated_ids: Vec<usize> = Vec::new();
        for (obj_id, det_idx) in &matches {
            let detection = &current_detections[*det_idx];
            if let Some(obj) = self.tracked_objects.get_mut(obj_id) {
                // 1€ 质心低通滤波（静止强平滑，运动低延迟）
                let mut centroid = detection.centroid;
                if self.use_centroid_smoothing {
                    obj.apply_centroid_lpf(&mut centroid, self.centroid_fc_min, self.centroid_beta);
                }
                obj.correct(
                    &detection.the_box,
                    detection.class_name.clone(),
                    detection.confidence,
                    centroid,
                )?;
                // 距离自适应 KF 噪声：远距离点云稀疏 → 质心不可靠 → 增大测量噪声
                let dist = (centroid[0] as f64).powi(2)
                    + (centroid[1] as f64).powi(2)
                    + (centroid[2] as f64).powi(2);
                obj.kalman_filter.adjust_noise_for_confidence(dist.sqrt(), detection.confidence);
                obj.update_score_on_match(self.track_score_match_bonus, self.track_score_max);
                updated_ids.push(*obj_id);
            }
        }

        // 步骤 4: 创建新轨迹（仍无法匹配的检测）
        for det_idx in unmatched_detections {
            let detection = &current_detections[det_idx];
            let new_id = self.next_id;
            self.next_id += 1;
            match TrackedObject::new(
                new_id,
                &detection.the_box,
                detection.class_name.clone(),
                detection.confidence,
                detection.centroid,
                self.kf_avg_frames,
                self.vel_smoothing_alpha,
            ) {
                Ok(obj) => {
                    self.tracked_objects.insert(new_id, obj);
                    updated_ids.push(new_id);
                }
                Err(e) => {
                    eprintln!("创建新跟踪对象失败：{:?}", e);
                }
            }
        }

        // 步骤 5: 未匹配的轨迹标记丢失（用 matched_ids 直接查找，不依赖 HashMap 顺序）
        let matched_ids: std::collections::HashSet<usize> =
            matches.iter().map(|(id, _)| *id).collect();
        for (id, obj) in &mut self.tracked_objects {
            if !matched_ids.contains(id) {
                obj.on_missed();
                obj.update_score_on_miss(self.track_score_miss_penalty);
            }
        }

        // 步骤 5a: 航迹分级生命周期管理（基于 Track Score）
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

        // 步骤 6: 移除丢失超时的轨迹
        self.tracked_objects.retain(|_id, obj| {
            obj.disappeared_count < self.max_disappeared
        });

        // 步骤 7+8: 并行分析（速度聚类 + 点云投票）
        let _t_pc = Instant::now();
        // 先更新点云历史（需要 &mut self）
        if self.use_point_cloud_voting {
            let filter_points = self.clouds_filtered.consumer().lock().unwrap().clone();
            if !filter_points.is_empty() {
                update_object_point_clouds(
                    &mut self.tracked_objects,
                    &filter_points,
                    self.point_cloud_history_len,
                    200,
                );
            }
        }

        // 步骤 7: 速度聚类分析 → 静态簇信号
        let _t_vel = Instant::now();
        let vel_snapshot: Vec<(usize, [f32; 3])> = self.tracked_objects
            .iter()
            .map(|(id, obj)| (*id, obj.smoothed_velocity()))
            .collect();
        let static_ids = analyze_velocity_clusters_from_snapshot(&vel_snapshot);

        // 步骤 8: 点云投票分析 → 运动信号（直接引用 tracked_objects，无拷贝开销）
        let _t_vote = Instant::now();
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

        // 步骤 9: 应用分类状态机（使用自运动补偿速度）
        let _t_sm = Instant::now();
        let ego_vel = self.ego_motion.update_async().await;
        let has_ego_motion = ego_vel[0].abs() + ego_vel[1].abs() + ego_vel[2].abs() > 0.01;
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
                self.floating_to_static_frames,
                self.voting_consistency_frames,
                self.class_cooldown_frames,
            );
        }

        // 步骤 9b: 人 → 强制 moving（状态机层面，跨帧持久）
        for (_, obj) in &mut self.tracked_objects {
            if obj.class_type == "person" && obj.classification != TargetClass::Moving {
                obj.classification = TargetClass::Moving;
                obj.confirmed_moving = true;
                obj.voting_streak = 0;
            }
        }

        // 步骤 9c: trick — 正在移动的目标标记为行人
        trick::apply(&mut self.tracked_objects);

        // 步骤 10: 箱体尺寸平滑
        let _t_fix = Instant::now();
        if self.use_box_smoothing {
            for obj in self.tracked_objects.values_mut() {
                // 短暂丢失时不更新箱体（保留最后已知尺寸）
                if obj.disappeared_count == 0 {
                    obj.apply_box_smoothing(self.box_smoothing_alpha);
                }
            }
        } else if self.use_fix_size {
            for obj in self.tracked_objects.values_mut() {
                obj.apply_fix_size(self.fix_size_frames, self.fix_size_dim_thresh);
            }
        }

        // 步骤 11: 生成输出（基于 Track Score 过滤低分轨迹）
        let _t_out = Instant::now();
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

                // 短暂丢失时优先用平滑箱体（更稳定）
                let ref_box = if obj.disappeared_count > 0 {
                    obj.smoothed_box.as_ref()
                        .or(obj.last_box.as_ref())
                        .cloned()
                        .unwrap_or_else(Box3D::empty_box)
                } else {
                    obj.last_box.as_ref().cloned().unwrap_or_else(Box3D::empty_box)
                };
                let mut predicted_box = Box3D::from_position_and_angles(
                    pos.x as f32,
                    pos.y as f32,
                    z_out,
                    0.0, 0.0, 0.0,
                    ref_box.length,
                    ref_box.width,
                    ref_box.height,
                );
                predicted_box.pose = ref_box.pose;

                let class_str = match obj.classification {
                    TargetClass::Static => "static",
                    TargetClass::Floating => "floating",
                    TargetClass::Moving => "moving",
                    TargetClass::Movable => "movable",
                };

                if *tracked_id <= 5 && obj.appearance_count <= 1 {
                    eprintln!("DEBUG new obj {}: kf_vel=({:.4},{:.4},{:.4}) speed={:.4} hist_len={}",
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

                // 地面始终 static（语义不变）
                if target.class_type == "ground" {
                    target.classification = "static".to_string();
                    target.is_dynamic = false;
                }
                // person：状态机层 confirmed_moving=true 保证不再退回 floating/static

                // Static 目标速度强制归零（避免输出噪声速度）
                if obj.classification == TargetClass::Static {
                    target.velocity = [0.0, 0.0, 0.0];
                    target.speed = 0.0;
                }

                output_targets.push(target);
            }
        }

        // 步骤 12: 写入输出
        let _t_write = Instant::now();
        {
            let mut target_guard = self.target.lock().unwrap();
            target_guard.write(output_targets)?;
        }

        let t_total = _t0.elapsed();
        let n_obj = self.tracked_objects.len();
        if n_obj > 0 || t_total.as_millis() > 5 {
            let t_read     = _t_pred.duration_since(_t0).as_secs_f64() * 1000.0;
            let t_predict  = _t_assoc.duration_since(_t_pred).as_secs_f64() * 1000.0;
            let t_assoc    = _t_update.duration_since(_t_assoc).as_secs_f64() * 1000.0;
            let t_update   = _t_pc.duration_since(_t_update).as_secs_f64() * 1000.0;
            let t_pc_upd   = _t_vel.duration_since(_t_pc).as_secs_f64() * 1000.0;
            let t_vel_cls  = _t_vote.duration_since(_t_vel).as_secs_f64() * 1000.0;
            let t_vote     = _t_sm.duration_since(_t_vote).as_secs_f64() * 1000.0;
            let t_sm       = _t_fix.duration_since(_t_sm).as_secs_f64() * 1000.0;
            let t_fix      = _t_out.duration_since(_t_fix).as_secs_f64() * 1000.0;
            let t_output   = _t_write.duration_since(_t_out).as_secs_f64() * 1000.0;
            let t_write    = t_total.as_secs_f64() * 1000.0 - t_read - t_predict - t_assoc - t_update - t_pc_upd - t_vel_cls - t_vote - t_sm - t_fix - t_output;
            log::debug!("[perf] {}obj read={:.1} pred={:.1} assoc={:.1} upd={:.1} pcupd={:.1} vel={:.1} vote={:.1} sm={:.1} fix={:.1} out={:.1} write={:.1} total={:.1}",
                n_obj, t_read, t_predict, t_assoc, t_update, t_pc_upd, t_vel_cls, t_vote, t_sm, t_fix, t_output, t_write, t_total.as_secs_f64() * 1000.0);
        }

        Ok(())
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
        let _t0 = Instant::now();
        let current_detections: Vec<CldBud> = detections
            .iter()
            .filter(|d| {
                d.confidence >= self.min_confidence
                    && d.class_name != "ground"
                    && d.class_name != "wall"
            })
            .cloned()
            .collect();

        let now = SystemTime::now();

        // 步骤 1: 预测
        let _t_pred = Instant::now();
        for obj in self.tracked_objects.values_mut() {
            let dt = now
                .duration_since(obj.last_seen)
                .unwrap_or_default()
                .as_secs_f64();
            let dt = dt.clamp(0.001, 1.0);
            if let Err(e) = obj.predict(dt) {
                eprintln!("预测失败：{:?}", e);
            }
        }

        // 步骤 2: 关联
        let _t_assoc = Instant::now();
        let (matches, unmatched_detections) = associate(
            &self.tracked_objects,
            &current_detections,
            &mut self.cost_buf,
            &mut self.sq_buf,
        );

        // 步骤 3: 修正匹配的轨迹
        let _t_update = Instant::now();
        let mut updated_ids: Vec<usize> = Vec::new();
        for (obj_id, det_idx) in &matches {
            let detection = &current_detections[*det_idx];
            if let Some(obj) = self.tracked_objects.get_mut(obj_id) {
                let mut centroid = detection.centroid;
                if self.use_centroid_smoothing {
                    obj.apply_centroid_lpf(
                        &mut centroid,
                        self.centroid_fc_min,
                        self.centroid_beta,
                    );
                }
                if let Err(e) = obj.correct(
                    &detection.the_box,
                    detection.class_name.clone(),
                    detection.confidence,
                    centroid,
                ) {
                    eprintln!("修正失败：{:?}", e);
                    continue;
                }
                let dist = (centroid[0] as f64).powi(2)
                    + (centroid[1] as f64).powi(2)
                    + (centroid[2] as f64).powi(2);
                obj.kalman_filter.adjust_noise_for_distance(dist.sqrt());
                obj.update_score_on_match(self.track_score_match_bonus, self.track_score_max);
                updated_ids.push(*obj_id);
            }
        }

        // 步骤 4: 创建新轨迹
        for det_idx in unmatched_detections {
            let detection = &current_detections[det_idx];
            let new_id = self.next_id;
            self.next_id += 1;
            match TrackedObject::new(
                new_id,
                &detection.the_box,
                detection.class_name.clone(),
                detection.confidence,
                detection.centroid,
                self.kf_avg_frames,
                self.vel_smoothing_alpha,
            ) {
                Ok(obj) => {
                    self.tracked_objects.insert(new_id, obj);
                    updated_ids.push(new_id);
                }
                Err(e) => {
                    eprintln!("创建新跟踪对象失败：{:?}", e);
                }
            }
        }

        // 步骤 5: 未匹配的轨迹标记丢失（用 matched_ids 直接查找 + 补齐 miss penalty）
        let matched_ids: std::collections::HashSet<usize> =
            matches.iter().map(|(id, _)| *id).collect();
        for (id, obj) in &mut self.tracked_objects {
            if !matched_ids.contains(id) {
                obj.on_missed();
                obj.update_score_on_miss(self.track_score_miss_penalty);
            }
        }

        // 步骤 5a: 航迹分级
        {
            let mut promote_ids = Vec::new();
            let mut tentative_prune = Vec::new();
            for (id, obj) in &self.tracked_objects {
                match obj.status {
                    TrackStatus::Tentative => {
                        if obj.consecutive_matches >= self.confirmation_frames as u32 {
                            promote_ids.push(*id);
                        } else if obj.disappeared_count >= self.tentative_max_missed as u32 {
                            tentative_prune.push(*id);
                        }
                    }
                    TrackStatus::Confirmed => {}
                }
            }
            for id in &promote_ids {
                if let Some(obj) = self.tracked_objects.get_mut(id) {
                    obj.status = TrackStatus::Confirmed;
                }
            }
            for id in &tentative_prune {
                self.tracked_objects.remove(id);
            }
        }

        // 步骤 6: 移除永久丢失的轨迹
        self.tracked_objects
            .retain(|_id, obj| !obj.is_permanently_lost(self.max_disappeared));

        // 步骤 7+8: 点云投票
        let _t_pc = Instant::now();
        if self.use_point_cloud_voting && !filtered_points.is_empty() {
            update_object_point_clouds(
                &mut self.tracked_objects,
                filtered_points,
                self.point_cloud_history_len,
                200,
            );
        }

        // 步骤 7: 速度聚类
        let _t_vel = Instant::now();
        let vel_snapshot: Vec<(usize, [f32; 3])> = self
            .tracked_objects
            .iter()
            .map(|(id, obj)| (*id, obj.smoothed_velocity()))
            .collect();
        let static_ids = analyze_velocity_clusters_from_snapshot(&vel_snapshot);

        // 步骤 8: 点云投票
        let _t_vote = Instant::now();
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

        // 步骤 9: 状态机（bench 模式下无自车运动补偿）
        let _t_sm = Instant::now();
        for (_, obj) in &mut self.tracked_objects {
            let in_static = static_ids.contains(&obj.id);
            let voting_on = voting_ids.contains(&obj.id);
            let spd = obj.speed();
            apply_state_machine(
                obj,
                in_static,
                voting_on,
                spd,
                self.moving_speed_threshold,
                self.floating_to_static_frames,
                self.voting_consistency_frames,
                self.class_cooldown_frames,
            );
        }

        // 步骤 9b: 人物强制 moving
        for (_, obj) in &mut self.tracked_objects {
            if obj.class_type == "person" && obj.classification != TargetClass::Moving {
                obj.classification = TargetClass::Moving;
                obj.confirmed_moving = true;
                obj.voting_streak = 0;
            }
        }

        // 步骤 9c: trick
        trick::apply(&mut self.tracked_objects);

        // 步骤 10: 箱体尺寸平滑
        let _t_fix = Instant::now();
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

        // 步骤 11: 生成输出（与 run() 一致的 score 阈值过滤）
        let _t_out = Instant::now();
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
                    obj.smoothed_box
                        .as_ref()
                        .or(obj.last_box.as_ref())
                        .cloned()
                        .unwrap_or_else(Box3D::empty_box)
                } else {
                    obj.last_box
                        .as_ref()
                        .cloned()
                        .unwrap_or_else(Box3D::empty_box)
                };
                let mut predicted_box = Box3D::from_position_and_angles(
                    pos.x as f32,
                    pos.y as f32,
                    z_out,
                    0.0,
                    0.0,
                    0.0,
                    ref_box.length,
                    ref_box.width,
                    ref_box.height,
                );
                predicted_box.pose = ref_box.pose;

                let class_str = match obj.classification {
                    TargetClass::Static => "static",
                    TargetClass::Floating => "floating",
                    TargetClass::Moving => "moving",
                    TargetClass::Movable => "movable",
                };

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
