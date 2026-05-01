use std::collections::{HashMap, VecDeque};
use std::time::{Instant, SystemTime};

use nalgebra::{Point3, Vector3, Vector6};

use crate::{
    cloud::CldBud,
    cloud::ego_motion::EgoMotion,
    config::fixif,
    swapl::global_swapl,
    tracker::{
        hungarian::hungarian,
        kalman::{KalmanConfig, KalmanFilterWrapper},
        output::Target,
    },
    utils::{
        boxes::Box3D,
        sight::Sight,
        stream::{Eap, Stream, StreamError},
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

/// 目标分类（状态机）
#[derive(Debug, Clone, PartialEq)]
enum TargetClass {
    Floating, // 待定——新对象或未确认运动能力
    Static,   // 背景/地面——confirmed 不可移动
    Moving,   // 运动中（confirmed）
    Movable,  // 可运动——曾确认运动，当前静止
}

/// 跟踪目标信息（包含卡尔曼滤波器）
struct TrackedObject {
    id: usize,
    class_type: String,
    last_seen: SystemTime,
    disappeared_count: u32,
    appearance_count: u32,
    confidence: f32,
    kalman_filter: KalmanFilterWrapper,
    velocity_history: VecDeque<[f32; 3]>,
    classification: TargetClass,
    /// 一旦为 true，永不回到 Static/Floating
    confirmed_moving: bool,
    /// 连续点云投票通过帧数（Floating→Moving 用）
    voting_streak: u32,
    /// 连续处于静态簇帧数（Floating→Static 用）
    floating_static_count: u32,
    /// 关联时缓存最近的检测框，避免输出阶段二次搜索
    last_box: Option<Box3D>,
    /// 冻结的箱体尺寸（fix_size 稳定用）
    fixed_box: Option<Box3D>,
    /// 点云历史（环形缓冲区，用于点云投票）
    point_cloud_history: VecDeque<Vec<[f32; 3]>>,
    /// k 帧位置历史（用于 LV-DOT 风格的 k 帧速度观测）
    position_history: VecDeque<[f64; 3]>,
    /// k 帧速度观测平均帧数
    kf_avg_frames: usize,
}

impl TrackedObject {
    fn new(
        id: usize,
        initial_box: &Box3D,
        class_type: String,
        confidence: f32,
        centroid: [f32; 3],
        kf_avg_frames: usize,
    ) -> Result<Self, adskalman::Error> {
        let mut kalman_filter = KalmanFilterWrapper::new(KalmanConfig::default())?;
        // 初始状态：[x, y, z, 0, 0, 0]
        kalman_filter.init_with_state(Vector6::new(
            centroid[0] as f64, centroid[1] as f64, centroid[2] as f64,
            0.0, 0.0, 0.0,
        ));
        Ok(Self {
            id,
            class_type,
            last_seen: SystemTime::now(),
            disappeared_count: 0,
            appearance_count: 0,
            confidence,
            kalman_filter,
            velocity_history: VecDeque::with_capacity(10),
            classification: TargetClass::Floating,
            confirmed_moving: false,
            voting_streak: 0,
            floating_static_count: 0,
            last_box: Some(initial_box.clone()),
            fixed_box: None,
            point_cloud_history: VecDeque::with_capacity(16),
            position_history: VecDeque::with_capacity(kf_avg_frames + 2),
            kf_avg_frames,
        })
    }

    /// 预测：将状态前推 dt 秒
    fn predict(&mut self, dt: f64) -> Result<(), adskalman::Error> {
        self.kalman_filter.predict(dt)
    }

    /// 修正（LV-DOT 风格）：用 [x,y,z,vx,vy,vz] 校正
    ///
    /// - 位置来自当前帧点云质心 centroid
    /// - 速度通过 k 帧位置差计算：(pos_t - pos_{t-k}) / (k * dt)
    ///   k = kf_avg_frames，dt 从上次修正到现在的实际时间
    fn correct(
        &mut self,
        new_box: &Box3D,
        new_class_type: String,
        new_confidence: f32,
        centroid: [f32; 3],
    ) -> Result<(), adskalman::Error> {
        let now = SystemTime::now();
        let dt_since_last = now.duration_since(self.last_seen)
            .unwrap_or_default().as_secs_f64().clamp(0.001, 1.0);

        // 记录位置历史（环形缓冲）
        self.position_history.push_back([
            centroid[0] as f64,
            centroid[1] as f64,
            centroid[2] as f64,
        ]);
        if self.position_history.len() > self.kf_avg_frames + 2 {
            self.position_history.pop_front();
        }

        // LV-DOT：v = (pos_t - pos_{t-k}) / (k * dt)
        let hist_len = self.position_history.len();
        let k = self.kf_avg_frames.min(hist_len.saturating_sub(1));
        const MIN_K_FOR_VELOCITY: usize = 3;
        if k >= MIN_K_FOR_VELOCITY {
            let old = self.position_history[hist_len - 1 - k];
            let curr = *self.position_history.back().unwrap();
            let dt_k = (k as f64 * dt_since_last).max(0.001);
            let meas_vx = (curr[0] - old[0]) / dt_k;
            let meas_vy = (curr[1] - old[1]) / dt_k;
            let meas_vz = (curr[2] - old[2]) / dt_k;
            let measurement = Vector6::new(
                centroid[0] as f64, centroid[1] as f64, centroid[2] as f64,
                meas_vx, meas_vy, meas_vz,
            );
            self.kalman_filter.correct(measurement)?;
        } else {
            // 历史不足，位置-only 修正（避免 0 速度观测污染状态）
            self.kalman_filter.correct_position(Vector3::new(
                centroid[0] as f64,
                centroid[1] as f64,
                centroid[2] as f64,
            ))?;
        }

        // 限幅：防止关联错误导致单帧速度尖峰
        self.kalman_filter.clamp_velocity(10.0);

        // 记录速度用于聚类
        let v = self.kalman_filter.get_velocity();
        if self.velocity_history.len() >= 10 {
            self.velocity_history.pop_front();
        }
        self.velocity_history.push_back([v.x as f32, v.y as f32, v.z as f32]);

        self.appearance_count += 1;
        self.class_type = new_class_type;
        self.confidence = new_confidence;
        self.last_seen = SystemTime::now();
        self.disappeared_count = 0;
        self.last_box = Some(new_box.clone());
        Ok(())
    }

    /// 帧增长（未匹配时调用）
    fn on_missed(&mut self) {
        self.disappeared_count += 1;
    }

    fn is_permanently_lost(&self, max_disappeared: u32) -> bool {
        self.disappeared_count >= max_disappeared
    }

    /// 获取 Kalman 估计速度（取最近几帧平滑值）
    fn smoothed_velocity(&self) -> [f32; 3] {
        let v = self.kalman_filter.get_velocity();
        if self.velocity_history.is_empty() {
            return [v.x as f32, v.y as f32, v.z as f32];
        }
        // 历史均值与当前 Kalman 速度混合（50/50）
        let mut avg = [0.0f32; 3];
        for hv in &self.velocity_history {
            avg[0] += hv[0];
            avg[1] += hv[1];
            avg[2] += hv[2];
        }
        let n = self.velocity_history.len() as f32;
        [
            0.5 * (v.x as f32) + 0.5 * avg[0] / n,
            0.5 * (v.y as f32) + 0.5 * avg[1] / n,
            0.5 * (v.z as f32) + 0.5 * avg[2] / n,
        ]
    }

    fn speed(&self) -> f32 {
        let v = self.smoothed_velocity();
        (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
    }

    /// 箱体尺寸锁定：跟踪稳定后如果尺寸变化小则冻结
    fn apply_fix_size(&mut self, min_frames: usize, dim_thresh: f32) {
        if self.appearance_count <= min_frames as u32 {
            return;
        }
        let current = match self.last_box {
            Some(ref b) => b.clone(),
            None => return,
        };
        let fixed = match self.fixed_box {
            Some(ref f) => f.clone(),
            None => {
                self.fixed_box = Some(current);
                return;
            }
        };
        // 计算三维尺寸变化率的最大值
        let ratio = [
            (current.length - fixed.length).abs() / fixed.length.max(1e-6),
            (current.width - fixed.width).abs() / fixed.width.max(1e-6),
            (current.height - fixed.height).abs() / fixed.height.max(1e-6),
        ]
        .iter()
        .cloned()
        .fold(0.0f32, f32::max);

        if ratio < dim_thresh {
            // 变化小 → 冻结尺寸
            let mut locked = current;
            locked.length = fixed.length;
            locked.width = fixed.width;
            locked.height = fixed.height;
            self.last_box = Some(locked);
        } else {
            // 变化大 → 更新参考
            self.fixed_box = Some(current);
        }
    }
}

/// 主跟踪器
pub struct Tracker {
    sight: Eap<Stream<Vec<Sight>>>,
    tar3d: Eap<Stream<Vec<CldBud>>>,
    target: Eap<Stream<Vec<Target>>>,
    clouds_filtered: Eap<Stream<Vec<[f32; 3]>>>,
    next_id: usize,
    tracked_objects: HashMap<usize, TrackedObject>,
    max_disappeared: u32,
    min_confidence: f32,
    min_appearances: u32,
    /// 点云投票配置
    use_point_cloud_voting: bool,
    point_cloud_vote_threshold: f32,
    point_cloud_skip_frames: usize,
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
    /// 自运动补偿
    ego_motion: EgoMotion,
    /// 匈牙利算法代价矩阵缓冲区（避免每帧堆分配）
    cost_buf: Vec<Vec<f64>>,
    /// 匈牙利算法方阵缓冲区（避免每帧堆分配）
    sq_buf: Vec<Vec<f64>>,
}

impl Tracker {
    pub fn new() -> Self {
        let swapl = global_swapl();
        let cfg = &fixif().tracker;
        Self {
            sight: swapl.sights.clone(),
            tar3d: swapl.cld_objs.clone(),
            target: swapl.targets.clone(),
            clouds_filtered: swapl.clouds_filtered.clone(),
            next_id: 1,
            tracked_objects: HashMap::new(),
            max_disappeared: cfg.max_disappeared,
            min_confidence: cfg.min_confidence,
            min_appearances: cfg.min_appearances,
            use_point_cloud_voting: cfg.use_point_cloud_voting,
            point_cloud_vote_threshold: cfg.point_cloud_vote_threshold,
            point_cloud_skip_frames: cfg.point_cloud_skip_frames,
            point_cloud_history_len: cfg.point_cloud_history_len,
            use_fix_size: cfg.use_fix_size,
            fix_size_frames: cfg.fix_size_frames,
            fix_size_dim_thresh: cfg.fix_size_dim_thresh,
            kf_avg_frames: cfg.kf_avg_frames,
            floating_to_static_frames: cfg.floating_to_static_frames,
            moving_speed_threshold: cfg.moving_speed_threshold,
            voting_consistency_frames: cfg.voting_consistency_frames,
            ego_motion: EgoMotion::new(),
            cost_buf: Vec::new(),
            sq_buf: Vec::new(),
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

    /// 马氏距离门控阈值
    ///
    /// χ²(3) 在 α=0.05 时阈值为 7.815
    /// sqrt(7.815) ≈ 2.795 用于距离比较
    const MAHALANOBIS_THRESHOLD: f64 = 2.796;

    /// 马氏距离关联（匈牙利算法最优指派）
    ///
    /// `cost_buf` / `sq_buf` 复用缓冲区，避免每帧堆分配。
    fn associate(
        objects: &HashMap<usize, TrackedObject>,
        detections: &[CldBud],
        cost_buf: &mut Vec<Vec<f64>>,
        sq_buf: &mut Vec<Vec<f64>>,
    ) -> (Vec<(usize, usize)>, Vec<usize>) {
        let n_objects = objects.len();
        let n_detections = detections.len();

        if n_objects == 0 || n_detections == 0 {
            return (Vec::new(), (0..n_detections).collect());
        }

        let obj_ids: Vec<usize> = objects.keys().copied().collect();

        // 构建代价矩阵（复用缓冲区）
        cost_buf.clear();
        cost_buf.resize(n_objects, Vec::with_capacity(n_detections));
        for row in cost_buf.iter_mut() {
            row.clear();
            row.resize(n_detections, f64::MAX);
        }
        for (obj_idx, &obj_id) in obj_ids.iter().enumerate() {
            let obj = &objects[&obj_id];
            for (det_idx, det) in detections.iter().enumerate() {
                let center = det.the_box.center();
                let meas = Vector3::new(center.x as f64, center.y as f64, center.z as f64);
                let dist = obj.kalman_filter.mahalanobis_distance(meas);
                if dist < Self::MAHALANOBIS_THRESHOLD {
                    cost_buf[obj_idx][det_idx] = dist;
                }
            }
        }

        // 匈牙利最优指派（复用 sq_buf）
        let assignment = hungarian(cost_buf, sq_buf);

        // 提取匹配结果
        let mut used_det = vec![false; n_detections];
        let mut matches = Vec::new();

        for (obj_idx, &det_idx) in assignment.iter().enumerate() {
            if det_idx < n_detections && cost_buf[obj_idx][det_idx] < f64::MAX / 2.0 {
                matches.push((obj_idx, det_idx));
                used_det[det_idx] = true;
            }
        }

        let unmatched: Vec<usize> = (0..n_detections)
            .filter(|&i| !used_det[i])
            .collect();

        (matches, unmatched)
    }

    /// 提取包围盒内的点（AABB 快速过滤），最多取 `max_out` 个点
    /// 超出的部分步长抽样，保证 O(N²) 投票可控
    fn extract_points_in_box(points: &[[f32; 3]], box3d: &Box3D, max_out: usize) -> Vec<[f32; 3]> {
        // 先用 AABB 粗略过滤
        let verts = box3d.vertices();
        let (mut x_min, mut x_max) = (verts[0].x, verts[0].x);
        let (mut y_min, mut y_max) = (verts[0].y, verts[0].y);
        let (mut z_min, mut z_max) = (verts[0].z, verts[0].z);
        for v in &verts {
            x_min = x_min.min(v.x);
            x_max = x_max.max(v.x);
            y_min = y_min.min(v.y);
            y_max = y_max.max(v.y);
            z_min = z_min.min(v.z);
            z_max = z_max.max(v.z);
        }

        // 预计算逆矩阵（避免每点重复求逆）
        let inv_pose = box3d.pose.try_inverse().unwrap_or_else(|| panic!("矩阵不可求逆: {}", box3d.pose));
        let hl = box3d.length / 2.0;
        let hw = box3d.width / 2.0;
        let hh = box3d.height / 2.0;

        let candidates: Vec<[f32; 3]> = points.iter()
            .filter(|p| {
                p[0] >= x_min && p[0] <= x_max
                    && p[1] >= y_min && p[1] <= y_max
                    && p[2] >= z_min && p[2] <= z_max
                    && {
                        let local = inv_pose.transform_point(&Point3::new(p[0], p[1], p[2]));
                        local.x >= -hl && local.x <= hl
                            && local.y >= -hw && local.y <= hw
                            && local.z >= -hh && local.z <= hh
                    }
            })
            .copied()
            .collect();

        if candidates.len() <= max_out {
            candidates
        } else {
            // 均匀步长下采样到 max_out 个点
            let step = candidates.len() / max_out;
            candidates.into_iter().step_by(step).take(max_out).collect()
        }
    }

    /// 更新所有活跃轨迹的点云历史
    ///
    /// 先计算所有目标 AABB 的并集，快速过滤非目标区域点云，
    /// 再逐目标精确过滤，避免对每个目标扫描整个点云。
    fn update_object_point_clouds(
        objects: &mut HashMap<usize, TrackedObject>,
        filter_points: &[[f32; 3]],
        max_history: usize,
        max_points_per_obj: usize,
    ) {
        // Step 1: 收集所有目标 box AABB 并集
        let boxes: Vec<&Box3D> = objects.values()
            .filter_map(|obj| obj.last_box.as_ref())
            .collect();

        if boxes.is_empty() {
            return;
        }

        let (mut ax_min, mut ax_max) = (f32::MAX, f32::NEG_INFINITY);
        let (mut ay_min, mut ay_max) = (f32::MAX, f32::NEG_INFINITY);
        let (mut az_min, mut az_max) = (f32::MAX, f32::NEG_INFINITY);
        for b in &boxes {
            let v = b.vertices();
            for p in &v {
                ax_min = ax_min.min(p.x); ax_max = ax_max.max(p.x);
                ay_min = ay_min.min(p.y); ay_max = ay_max.max(p.y);
                az_min = az_min.min(p.z); az_max = az_max.max(p.z);
            }
        }

        // Step 2: 一次扫描，得到落在联合 AABB 内的候选点
        let candidates: Vec<[f32; 3]> = filter_points.iter()
            .filter(|p| {
                p[0] >= ax_min && p[0] <= ax_max
                    && p[1] >= ay_min && p[1] <= ay_max
                    && p[2] >= az_min && p[2] <= az_max
            })
            .copied()
            .collect();

        // Step 3: 逐目标精确过滤（仅对候选点操作）
        for obj in objects.values_mut() {
            if let Some(ref last_box) = obj.last_box {
                let pts = if candidates.is_empty() {
                    Self::extract_points_in_box(filter_points, last_box, max_points_per_obj)
                } else {
                    let verts = last_box.vertices();
                    let (mut x_min, mut x_max) = (verts[0].x, verts[0].x);
                    let (mut y_min, mut y_max) = (verts[0].y, verts[0].y);
                    let (mut z_min, mut z_max) = (verts[0].z, verts[0].z);
                    for v in &verts {
                        x_min = x_min.min(v.x); x_max = x_max.max(v.x);
                        y_min = y_min.min(v.y); y_max = y_max.max(v.y);
                        z_min = z_min.min(v.z); z_max = z_max.max(v.z);
                    }
                    // 预计算逆矩阵（避免每点重复求逆）
                    let inv_pose = last_box.pose.try_inverse().unwrap_or_else(|| panic!("矩阵不可求逆"));
                    let hl = last_box.length / 2.0;
                    let hw = last_box.width / 2.0;
                    let hh = last_box.height / 2.0;
                    let c: Vec<[f32; 3]> = candidates.iter()
                        .filter(|p| {
                            p[0] >= x_min && p[0] <= x_max
                                && p[1] >= y_min && p[1] <= y_max
                                && p[2] >= z_min && p[2] <= z_max
                                && {
                                    let local = inv_pose.transform_point(&Point3::new(p[0], p[1], p[2]));
                                    local.x >= -hl && local.x <= hl
                                        && local.y >= -hw && local.y <= hw
                                        && local.z >= -hh && local.z <= hh
                                }
                        })
                        .copied()
                        .collect();
                    if c.len() <= max_points_per_obj {
                        c
                    } else {
                        let step = c.len() / max_points_per_obj;
                        c.into_iter().step_by(step).take(max_points_per_obj).collect()
                    }
                };
                if pts.is_empty() {
                    continue;
                }
                if obj.point_cloud_history.len() >= max_history {
                    obj.point_cloud_history.pop_front();
                }
                obj.point_cloud_history.push_back(pts);
            }
        }
    }

    /// 点云投票分析（信号产生，不修改分类）
    ///
    /// 对每个轨迹：当前帧点云 vs skip_frames 帧前点云
    /// 分类状态机
    ///
    /// 变迁规则：
    ///   Static ←→ Floating（滞后：上浮无门槛，沉淀需 30 帧）
    ///   Floating ──> Moving（点云投票 + 速度 + 连续帧）
    ///   Moving ←──→ Movable（同层往返，速度决定）
    ///   一旦 confirmed_moving=true，永不回到 Static/Floating
    fn apply_state_machine(
        obj: &mut TrackedObject,
        in_static_cluster: bool,
        voting_active: bool,
        speed: f32,
        moving_speed_threshold: f32,
        floating_to_static_frames: usize,
        voting_consistency_frames: usize,
    ) {
        match obj.classification {
            TargetClass::Static => {
                if !in_static_cluster {
                    // Static → Floating：任何偏离静态簇（无门槛）
                    obj.classification = TargetClass::Floating;
                    obj.floating_static_count = 0;
                }
            }
            TargetClass::Floating => {
                if in_static_cluster {
                    // 持续在静态簇中 → 累积计数
                    obj.floating_static_count += 1;
                    if obj.floating_static_count >= floating_to_static_frames as u32 {
                        obj.classification = TargetClass::Static;
                        obj.floating_static_count = 0;
                    }
                } else {
                    obj.floating_static_count = 0;
                }

                // 点云投票 → Floating → Moving
                if voting_active && speed >= 0.2 {
                    obj.voting_streak += 1;
                    if obj.voting_streak >= voting_consistency_frames as u32 {
                        obj.classification = TargetClass::Moving;
                        obj.confirmed_moving = true;
                        obj.voting_streak = 0;
                    }
                } else {
                    obj.voting_streak = 0;
                }
            }
            TargetClass::Moving | TargetClass::Movable => {
                obj.confirmed_moving = true;
                // Moving ↔ Movable：同层往返
                if speed > moving_speed_threshold {
                    obj.classification = TargetClass::Moving;
                } else {
                    obj.classification = TargetClass::Movable;
                }
            }
        }
    }

    fn refine_classification_with_sight(target: &mut Target, sight_data: &[Sight]) {
        for sight in sight_data {
            if sight.slab(&target.the_box) {
                target.class_type = "person".to_string();
                return;
            }
        }
        // 把无意义的 cluster_N 替换为通用障碍物标签
        if target.class_type.is_empty() || target.class_type.starts_with("cluster_") {
            target.class_type = "obstacle".to_string();
        }
    }

    pub async fn run(&mut self) -> Result<(), TrackerError> {
        let _t0 = Instant::now();
        let current_detections = {
            let mut tar3d_guard = self.tar3d.lock().await;
            match tar3d_guard.read() {
                Some(data) => data.into_iter()
                    .filter(|d| d.confidence >= self.min_confidence)
                    .collect::<Vec<_>>(),
                None => Vec::new(),
            }
        };

        let sight_data = {
            let mut sight_guard = self.sight.lock().await;
            match sight_guard.read() {
                Some(data) => data,
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
        let (matches, unmatched_detections) = Self::associate(
            &self.tracked_objects,
            &current_detections,
            &mut self.cost_buf,
            &mut self.sq_buf,
        );

        // 步骤 3: 更新匹配的轨迹
        let _t_update = Instant::now();
        let mut updated_ids: Vec<usize> = Vec::new();
        for (obj_idx, det_idx) in &matches {
            let obj_id: usize = self.tracked_objects.keys().nth(*obj_idx).copied().unwrap();
            let detection = &current_detections[*det_idx];
            if let Some(obj) = self.tracked_objects.get_mut(&obj_id) {
                obj.correct(
                    &detection.the_box,
                    detection.class_name.clone(),
                    detection.confidence,
                    detection.centroid,
                )?;
                // 距离自适应 KF 噪声：远距离点云稀疏 → 质心不可靠 → 增大测量噪声
                let dist = (detection.centroid[0] as f64).powi(2)
                    + (detection.centroid[1] as f64).powi(2)
                    + (detection.centroid[2] as f64).powi(2);
                obj.kalman_filter.adjust_noise_for_distance(dist.sqrt());
                updated_ids.push(obj_id);
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

        // 步骤 5: 未匹配的轨迹标记丢失
        let matched_obj_indices: std::collections::HashSet<usize> =
            matches.iter().map(|(oi, _)| *oi).collect();
        let all_ids: Vec<usize> = self.tracked_objects.keys().copied().collect();
        for (idx, obj_id) in all_ids.iter().enumerate() {
            if !matched_obj_indices.contains(&idx) {
                if let Some(obj) = self.tracked_objects.get_mut(obj_id) {
                    obj.on_missed();
                }
            }
        }

        // 步骤 6: 移除永久丢失的轨迹
        self.tracked_objects.retain(|id, obj| {
            if obj.is_permanently_lost(self.max_disappeared) {
                eprintln!("移除消失的目标 ID: {}, 丢失帧数：{}", id, obj.disappeared_count);
                false
            } else {
                true
            }
        });

        // 步骤 7+8: 并行分析（速度聚类 + 点云投票）
        let _t_pc = Instant::now();
        // 先更新点云历史（需要 &mut self）
        if self.use_point_cloud_voting {
            let filter_points = {
                let mut cf = self.clouds_filtered.lock().await;
                match cf.read() {
                    Some(data) => data,
                    None => Vec::new(),
                }
            };
            if !filter_points.is_empty() {
                Self::update_object_point_clouds(
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
            Self::apply_state_machine(
                obj, in_static, voting_on, spd,
                self.moving_speed_threshold,
                self.floating_to_static_frames,
                self.voting_consistency_frames,
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

        // 步骤 10: 箱体尺寸锁定 fix_size
        let _t_fix = Instant::now();
        if self.use_fix_size {
            for obj in self.tracked_objects.values_mut() {
                obj.apply_fix_size(self.fix_size_frames, self.fix_size_dim_thresh);
            }
        }

        // 步骤 11: 生成输出（过滤短命目标）
        let _t_out = Instant::now();
        let mut output_targets = Vec::new();
        for tracked_id in &updated_ids {
            if let Some(obj) = self.tracked_objects.get(tracked_id) {
                if obj.appearance_count < self.min_appearances {
                    continue;
                }
                let pos = obj.kalman_filter.get_position();
                let vel = obj.smoothed_velocity();
                let spd = obj.speed();

                let ref_box = obj.last_box.as_ref().cloned().unwrap_or_else(Box3D::empty_box);
                let mut predicted_box = Box3D::from_position_and_angles(
                    pos.x as f32,
                    pos.y as f32,
                    pos.z as f32,
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

                Self::refine_classification_with_sight(&mut target, &sight_data);

                // 地面始终 static（语义不变）
                if target.class_type == "ground" {
                    target.classification = "static".to_string();
                    target.is_dynamic = false;
                }
                // person：状态机层 confirmed_moving=true 保证不再退回 floating/static

                output_targets.push(target);
            }
        }

        // 步骤 12: 写入输出
        let _t_write = Instant::now();
        {
            let mut target_guard = self.target.lock().await;
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

// ─── 分析辅助函数 ──────────────────────────────────────────────────────────

/// 基于 (id, vel) 快照的 DBSCAN 速度聚类
fn analyze_velocity_clusters_from_snapshot(snapshot: &[(usize, [f32; 3])]) -> Vec<usize> {
    let n = snapshot.len();
    if n < 2 {
        return Vec::new();
    }

    let ids: Vec<usize> = snapshot.iter().map(|(id, _)| *id).collect();
    let velocities: Vec<[f32; 3]> = snapshot.iter().map(|(_, v)| *v).collect();

    let eps = 0.3f32;
    let min_pts = 2;
    let mut neighbor_counts = vec![0; n];

    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let d = ((velocities[i][0] - velocities[j][0]).powi(2)
                + (velocities[i][1] - velocities[j][1]).powi(2)
                + (velocities[i][2] - velocities[j][2]).powi(2))
            .sqrt();
            if d < eps {
                neighbor_counts[i] += 1;
            }
        }
    }

    let mut clusters: Vec<Vec<usize>> = Vec::new();
    let mut assigned = vec![false; n];

    for i in 0..n {
        if assigned[i] || neighbor_counts[i] < min_pts {
            continue;
        }
        let mut cluster = Vec::new();
        let mut stack = vec![i];
        assigned[i] = true;
        while let Some(idx) = stack.pop() {
            cluster.push(idx);
            for j in 0..n {
                if assigned[j] {
                    continue;
                }
                let d = ((velocities[idx][0] - velocities[j][0]).powi(2)
                    + (velocities[idx][1] - velocities[j][1]).powi(2)
                    + (velocities[idx][2] - velocities[j][2]).powi(2))
                .sqrt();
                if d < eps {
                    assigned[j] = true;
                    stack.push(j);
                }
            }
        }
        clusters.push(cluster);
    }

    if let Some(largest) = clusters.iter().max_by_key(|c| c.len()) {
        largest.iter().map(|&pos| ids[pos]).collect()
    } else {
        Vec::new()
    }
}

/// 点云投票分析（直接引用 tracked_objects，无拷贝）
fn analyze_point_cloud_voting_direct(
    objects: &HashMap<usize, TrackedObject>,
    vote_threshold: f32,
    skip_frames: usize,
) -> Vec<usize> {
    let mut pass_ids = Vec::new();
    let ids: Vec<usize> = objects.keys().copied().collect();

    for id in &ids {
        let obj = match objects.get(id) {
            Some(obj) => obj,
            None => continue,
        };

        let hist_len = obj.point_cloud_history.len();
        if hist_len <= skip_frames {
            continue;
        }

        let old_pts = &obj.point_cloud_history[hist_len - 1 - skip_frames];
        let new_pts = &obj.point_cloud_history[hist_len - 1];

        if old_pts.is_empty() || new_pts.is_empty() {
            continue;
        }

        let speed = obj.speed();
        if speed < 0.2 {
            continue;
        }

        let vel = obj.kalman_filter.get_velocity();

        let mut votes = 0usize;
        let total = new_pts.len().min(old_pts.len());
        if total == 0 {
            continue;
        }

        for i in 0..total {
            let np = new_pts[i];
            let best_old = old_pts.iter().min_by(|a, b| {
                let da = (np[0] - a[0]).powi(2) + (np[1] - a[1]).powi(2) + (np[2] - a[2]).powi(2);
                let db = (np[0] - b[0]).powi(2) + (np[1] - b[1]).powi(2) + (np[2] - b[2]).powi(2);
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            });
            if let Some(op) = best_old {
                let dx = np[0] - op[0];
                let dy = np[1] - op[1];
                let dz = np[2] - op[2];
                let dot = dx * vel.x as f32 + dy * vel.y as f32 + dz * vel.z as f32;
                if dot > 0.0 {
                    votes += 1;
                }
            }
        }

        let ratio = votes as f32 / total as f32;
        if ratio >= vote_threshold {
            pass_ids.push(*id);
        }
    }

    pass_ids
}
