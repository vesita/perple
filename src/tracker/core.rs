use std::collections::HashMap;
use std::time::SystemTime;

use nalgebra::Vector3;

use crate::{
    cloud::CldBud,
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

/// 目标分类
#[derive(Debug, Clone, PartialEq)]
enum TargetClass {
    Unknown,
    Static,
    Dynamic,
    Movable,
}

/// 跟踪目标信息（包含卡尔曼滤波器）
struct TrackedObject {
    id: usize,
    class_type: String,
    last_seen: SystemTime,
    disappeared_count: u32,
    confidence: f32,
    kalman_filter: KalmanFilterWrapper,
    velocity_history: Vec<[f32; 3]>,
    classification: TargetClass,
    classification_history: Vec<TargetClass>,
    /// 关联时缓存最近的检测框，避免输出阶段二次搜索
    last_box: Option<Box3D>,
}

impl TrackedObject {
    fn new(
        id: usize,
        initial_box: &Box3D,
        class_type: String,
        confidence: f32,
    ) -> Result<Self, adskalman::Error> {
        let mut kalman_filter = KalmanFilterWrapper::new(KalmanConfig::default())?;
        let center = initial_box.center();
        kalman_filter.init_with_state(
            Vector3::new(center.x as f64, center.y as f64, center.z as f64),
            None,
        );
        Ok(Self {
            id,
            class_type,
            last_seen: SystemTime::now(),
            disappeared_count: 0,
            confidence,
            kalman_filter,
            velocity_history: Vec::with_capacity(10),
            classification: TargetClass::Unknown,
            classification_history: Vec::with_capacity(10),
            last_box: Some(initial_box.clone()),
        })
    }

    /// 预测：将状态前推 dt 秒
    fn predict(&mut self, dt: f64) -> Result<(), adskalman::Error> {
        self.kalman_filter.predict(dt)
    }

    /// 修正：用观测值校正状态（不含预测）
    fn correct(
        &mut self,
        new_box: &Box3D,
        new_class_type: String,
        new_confidence: f32,
    ) -> Result<(), adskalman::Error> {
        let center = new_box.center();
        let measurement = Vector3::new(center.x as f64, center.y as f64, center.z as f64);
        self.kalman_filter.correct(measurement)?;

        // 记录速度用于聚类
        let v = self.kalman_filter.get_velocity();
        if self.velocity_history.len() >= 10 {
            self.velocity_history.remove(0);
        }
        self.velocity_history.push([v.x as f32, v.y as f32, v.z as f32]);

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
}

/// 主跟踪器
pub struct Tracker {
    sight: Eap<Stream<Vec<Sight>>>,
    tar3d: Eap<Stream<Vec<CldBud>>>,
    target: Eap<Stream<Vec<Target>>>,
    next_id: usize,
    tracked_objects: HashMap<usize, TrackedObject>,
    max_disappeared: u32,
    min_confidence: f32,
}

impl Tracker {
    pub fn new() -> Self {
        let swapl = global_swapl();
        Self {
            sight: swapl.sights.clone(),
            tar3d: swapl.cld_objs.clone(),
            target: swapl.targets.clone(),
            next_id: 1,
            tracked_objects: HashMap::new(),
            max_disappeared: 5,
            min_confidence: 0.3,
        }
    }

    pub fn set_max_disappeared(&mut self, max: u32) {
        self.max_disappeared = max;
    }

    pub fn set_min_confidence(&mut self, confidence: f32) {
        self.min_confidence = confidence;
    }

    /// 马氏距离门控阈值
    ///
    /// χ²(3) 在 α=0.05 时阈值为 7.815
    /// sqrt(7.815) ≈ 2.795 用于距离比较
    const MAHALANOBIS_THRESHOLD: f64 = 2.796;

    /// 马氏距离关联（匈牙利算法最优指派）
    fn associate(
        objects: &HashMap<usize, TrackedObject>,
        detections: &[CldBud],
    ) -> (Vec<(usize, usize)>, Vec<usize>) {
        let n_objects = objects.len();
        let n_detections = detections.len();

        if n_objects == 0 || n_detections == 0 {
            return (Vec::new(), (0..n_detections).collect());
        }

        let obj_ids: Vec<usize> = objects.keys().copied().collect();

        // 构建代价矩阵：马氏距离，超门限的标记为 INF
        let mut cost = vec![vec![f64::MAX; n_detections]; n_objects];
        for (obj_idx, &obj_id) in obj_ids.iter().enumerate() {
            let obj = &objects[&obj_id];
            for (det_idx, det) in detections.iter().enumerate() {
                let center = det.the_box.center();
                let meas = Vector3::new(center.x as f64, center.y as f64, center.z as f64);
                let dist = obj.kalman_filter.mahalanobis_distance(meas);
                if dist < Self::MAHALANOBIS_THRESHOLD {
                    cost[obj_idx][det_idx] = dist;
                }
            }
        }

        // 匈牙利最优指派
        let assignment = hungarian(&cost);

        // 提取匹配结果
        let mut used_det = vec![false; n_detections];
        let mut matches = Vec::new();

        for (obj_idx, &det_idx) in assignment.iter().enumerate() {
            if det_idx < n_detections && cost[obj_idx][det_idx] < f64::MAX / 2.0 {
                matches.push((obj_idx, det_idx));
                used_det[det_idx] = true;
            }
        }

        let unmatched: Vec<usize> = (0..n_detections)
            .filter(|&i| !used_det[i])
            .collect();

        (matches, unmatched)
    }

    /// 速度空间聚类分类
    ///
    /// 所有目标在 LiDAR 帧中跟踪，自车运动导致静态目标呈现相同速度（-v_car）。
    /// 找到最大速度簇 → 标记为 Static，偏离者 → Dynamic/Movable。
    fn classify_by_velocity(objects: &mut HashMap<usize, TrackedObject>) {
        if objects.is_empty() {
            return;
        }

        let ids: Vec<usize> = objects.keys().copied().collect();
        let n = ids.len();
        if n < 2 {
            // 只有一个目标，通过绝对速度判断
            let obj = objects.get_mut(&ids[0]).unwrap();
            let spd = obj.speed();
            if spd < 0.3 {
                set_classification(obj, TargetClass::Static);
            } else {
                set_classification(obj, TargetClass::Dynamic);
            }
            return;
        }

        // 收集所有速度向量
        let velocities: Vec<[f32; 3]> = ids.iter().map(|id| objects[id].smoothed_velocity()).collect();

        // DBSCAN 简化版：ε = 0.3 m/s，min_pts = 2
        let eps = 0.3f32;
        let min_pts = 2;

        // 计算每个点的邻居数
        let mut neighbor_counts = vec![0; n];
        let mut clusters: Vec<Vec<usize>> = Vec::new();
        let mut assigned = vec![false; n];

        for i in 0..n {
            for j in 0..n {
                if i == j { continue; }
                let d = ((velocities[i][0] - velocities[j][0]).powi(2)
                    + (velocities[i][1] - velocities[j][1]).powi(2)
                    + (velocities[i][2] - velocities[j][2]).powi(2))
                .sqrt();
                if d < eps {
                    neighbor_counts[i] += 1;
                }
            }
        }

        // 构建簇：从每个未分配的 core 点开始
        for i in 0..n {
            if assigned[i] { continue; }
            if neighbor_counts[i] < min_pts { continue; } // noise

            // 新簇
            let mut cluster = Vec::new();
            let mut stack = vec![i];
            assigned[i] = true;

            while let Some(idx) = stack.pop() {
                cluster.push(idx);
                for j in 0..n {
                    if assigned[j] { continue; }
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

        // 未分配的点 = noise
        let noise: Vec<usize> = (0..n).filter(|&i| !assigned[i]).collect();

        // 找到最大簇 → Static
        if let Some(largest) = clusters.iter().max_by_key(|c| c.len()) {
            let largest_set: std::collections::HashSet<usize> = largest.iter().copied().collect();
            for (pos, id) in ids.iter().enumerate() {
                if let Some(obj) = objects.get_mut(id) {
                    if largest_set.contains(&pos) {
                        set_classification(obj, TargetClass::Static);
                    } else if noise.contains(&pos) {
                        let spd = obj.speed();
                        if spd > 0.5 {
                            set_classification(obj, TargetClass::Dynamic);
                        } else {
                            set_classification(obj, TargetClass::Movable);
                        }
                    } else {
                        // 属于非最大簇
                        let spd = obj.speed();
                        if spd > 0.5 {
                            set_classification(obj, TargetClass::Dynamic);
                        } else {
                            set_classification(obj, TargetClass::Movable);
                        }
                    }
                }
            }
        } else {
            // 没有形成簇，全部是 noise
            for id in &ids {
                if let Some(obj) = objects.get_mut(id) {
                    let spd = obj.speed();
                    if spd > 0.5 {
                        set_classification(obj, TargetClass::Dynamic);
                    } else {
                        set_classification(obj, TargetClass::Static);
                    }
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
        if target.class_type.is_empty() {
            target.class_type = "obstacle".to_string();
        }
    }

    pub fn run(&mut self) -> Result<(), TrackerError> {
        let current_detections = {
            let mut tar3d_guard = self.tar3d.blocking_lock();
            match tar3d_guard.read() {
                Some(data) => data.into_iter()
                    .filter(|d| d.confidence >= self.min_confidence)
                    .collect::<Vec<_>>(),
                None => Vec::new(),
            }
        };

        let sight_data = {
            let mut sight_guard = self.sight.blocking_lock();
            match sight_guard.read() {
                Some(data) => data,
                None => Vec::new(),
            }
        };

        let now = SystemTime::now();

        // 步骤 1: 对所有轨迹做预测
        for obj in self.tracked_objects.values_mut() {
            let dt = now.duration_since(obj.last_seen)
                .unwrap_or_default()
                .as_secs_f64();
            // 限幅：最大 1s，最小 1ms
            let dt = dt.clamp(0.001, 1.0);
            obj.predict(dt)?;
        }

        // 步骤 2: 关联
        let (matches, unmatched_detections) = Self::associate(
            &self.tracked_objects,
            &current_detections,
        );

        // 步骤 3: 更新匹配的轨迹（只修正，不预测）
        let mut updated_ids: Vec<usize> = Vec::new();
        for (obj_idx, det_idx) in &matches {
            let obj_id: usize = self.tracked_objects.keys().nth(*obj_idx).copied().unwrap();
            let detection = &current_detections[*det_idx];
            if let Some(obj) = self.tracked_objects.get_mut(&obj_id) {
                obj.correct(
                    &detection.the_box,
                    detection.class_name.clone(),
                    detection.confidence,
                )?;
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

        // 步骤 7: 速度聚类分类
        Self::classify_by_velocity(&mut self.tracked_objects);

        // 步骤 8: 生成输出
        let mut output_targets = Vec::new();
        for tracked_id in &updated_ids {
            if let Some(obj) = self.tracked_objects.get(tracked_id) {
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
                    TargetClass::Dynamic => "dynamic",
                    TargetClass::Movable => "movable",
                    TargetClass::Unknown => "unknown",
                };

                let mut target = Target {
                    the_box: predicted_box,
                    class_type: obj.class_type.clone(),
                    id: *tracked_id,
                    velocity: vel,
                    speed: spd,
                    is_dynamic: obj.classification == TargetClass::Dynamic,
                    classification: class_str.to_string(),
                };

                Self::refine_classification_with_sight(&mut target, &sight_data);
                output_targets.push(target);
            }
        }

        // 步骤 9: 写入输出
        {
            let mut target_guard = self.target.blocking_lock();
            target_guard.write(output_targets)?;
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

/// 设置分类并维护历史
fn set_classification(obj: &mut TrackedObject, class: TargetClass) {
    obj.classification_history.push(class.clone());
    if obj.classification_history.len() > 5 {
        obj.classification_history.remove(0);
    }
    // 多数投票平滑
    let static_count = obj.classification_history.iter().filter(|c| **c == TargetClass::Static).count();
    let dynamic_count = obj.classification_history.iter().filter(|c| **c == TargetClass::Dynamic).count();
    let movable_count = obj.classification_history.iter().filter(|c| **c == TargetClass::Movable).count();

    let total = obj.classification_history.len();
    // 需要 > 60% 一致才切换
    if static_count as f64 > total as f64 * 0.6 {
        obj.classification = TargetClass::Static;
    } else if dynamic_count as f64 > total as f64 * 0.6 {
        obj.classification = TargetClass::Dynamic;
    } else if movable_count as f64 > total as f64 * 0.6 {
        obj.classification = TargetClass::Movable;
    }
    // 否则保持之前分类
}
