use std::collections::HashMap;
use std::time::SystemTime;

use nalgebra::{Vector3, Vector6};

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
    appearance_count: u32,
    confidence: f32,
    kalman_filter: KalmanFilterWrapper,
    velocity_history: Vec<[f32; 3]>,
    classification: TargetClass,
    classification_history: Vec<TargetClass>,
    /// 关联时缓存最近的检测框，避免输出阶段二次搜索
    last_box: Option<Box3D>,
    /// 冻结的箱体尺寸（fix_size 稳定用）
    fixed_box: Option<Box3D>,
    /// 点云历史（环形缓冲区，用于点云投票）
    point_cloud_history: Vec<Vec<[f32; 3]>>,
    /// 连续被判定为 dynamic candidate 的帧数
    dynamic_candidate_count: u32,
    /// k 帧位置历史（用于 LV-DOT 风格的 k 帧速度观测）
    position_history: Vec<[f64; 3]>,
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
            velocity_history: Vec::with_capacity(10),
            classification: TargetClass::Unknown,
            classification_history: Vec::with_capacity(10),
            last_box: Some(initial_box.clone()),
            fixed_box: None,
            point_cloud_history: Vec::with_capacity(16),
            dynamic_candidate_count: 0,
            position_history: Vec::with_capacity(kf_avg_frames + 2),
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
    /// 修正（LV-DOT 风格）：用 [x,y,z,vx,vy,vz] 校正
    ///
    /// - 位置来自当前帧点云质心 centroid
    /// - 速度通过 k 帧位置差计算：(pos_t - pos_{t-k}) / (k * dt)
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
        self.position_history.push([
            centroid[0] as f64,
            centroid[1] as f64,
            centroid[2] as f64,
        ]);
        if self.position_history.len() > self.kf_avg_frames + 2 {
            self.position_history.remove(0);
        }

        // LV-DOT：v = (pos_t - pos_{t-k}) / (k * dt)
        let hist_len = self.position_history.len();
        let k = self.kf_avg_frames.min(hist_len.saturating_sub(1));
        const MIN_K_FOR_VELOCITY: usize = 3;
        if k >= MIN_K_FOR_VELOCITY {
            let old = self.position_history[hist_len - 1 - k];
            let curr = *self.position_history.last().unwrap();
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

        // 记录速度用于聚类
        let v = self.kalman_filter.get_velocity();
        if self.velocity_history.len() >= 10 {
            self.velocity_history.remove(0);
        }
        self.velocity_history.push([v.x as f32, v.y as f32, v.z as f32]);

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
    point_cloud_consistency_frames: usize,
    point_cloud_history_len: usize,
    /// fix_size 配置
    use_fix_size: bool,
    fix_size_frames: usize,
    fix_size_dim_thresh: f32,
    /// k 帧速度观测平均帧数
    kf_avg_frames: usize,
}

impl Tracker {
    pub fn new() -> Self {
        let swapl = global_swapl();
        Self {
            sight: swapl.sights.clone(),
            tar3d: swapl.cld_objs.clone(),
            target: swapl.targets.clone(),
            clouds_filtered: swapl.clouds_filtered.clone(),
            next_id: 1,
            tracked_objects: HashMap::new(),
            max_disappeared: 8,
            min_confidence: 0.3,
            min_appearances: 3,
            use_point_cloud_voting: true,
            point_cloud_vote_threshold: 0.8,
            point_cloud_skip_frames: 5,
            point_cloud_consistency_frames: 15,
            point_cloud_history_len: 10,
            use_fix_size: true,
            fix_size_frames: 10,
            fix_size_dim_thresh: 0.4,
            kf_avg_frames: 10,
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

    /// 提取包围盒内的点（AABB 快速过滤）
    fn extract_points_in_box(points: &[[f32; 3]], box3d: &Box3D) -> Vec<[f32; 3]> {
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

        points.iter()
            .filter(|p| {
                p[0] >= x_min && p[0] <= x_max
                    && p[1] >= y_min && p[1] <= y_max
                    && p[2] >= z_min && p[2] <= z_max
                    && box3d.contains(p)
            })
            .copied()
            .collect()
    }

    /// 更新所有活跃轨迹的点云历史
    fn update_object_point_clouds(
        objects: &mut HashMap<usize, TrackedObject>,
        filter_points: &[[f32; 3]],
        max_history: usize,
    ) {
        for obj in objects.values_mut() {
            if let Some(ref last_box) = obj.last_box {
                let pts = Self::extract_points_in_box(filter_points, last_box);
                if pts.is_empty() {
                    continue; // 不推入空帧，避免稀释历史
                }
                if obj.point_cloud_history.len() >= max_history {
                    obj.point_cloud_history.remove(0);
                }
                obj.point_cloud_history.push(pts);
            }
        }
    }

    /// 点云投票动态分类（LV-DOT 启发）
    ///
    /// 对每个轨迹：当前帧点云 vs skip_frames 帧前点云
    /// 每个点找最近邻 → 点对方向与 KF 速度做点乘 → 方向一致=有效投票
    /// votes/total >= threshold 且 speed >= 0.2m/s → dynamic 候选
    /// 连续 consistency_frames 帧为候选 → 标记 Dynamic
    fn classify_by_point_cloud_voting(
        objects: &mut HashMap<usize, TrackedObject>,
        vote_threshold: f32,
        skip_frames: usize,
        consistency_frames: usize,
    ) {
        let ids: Vec<usize> = objects.keys().copied().collect();
        for id in &ids {
            let obj = match objects.get_mut(id) {
                Some(obj) => obj,
                None => continue,
            };

            let hist_len = obj.point_cloud_history.len();
            if hist_len <= skip_frames {
                continue; // 历史不够，跳过
            }

            let old_pts = &obj.point_cloud_history[hist_len - 1 - skip_frames];
            let new_pts = &obj.point_cloud_history[hist_len - 1];

            if old_pts.is_empty() || new_pts.is_empty() {
                continue;
            }

            let vel = obj.kalman_filter.get_velocity();
            let speed = (vel.x * vel.x + vel.y * vel.y + vel.z * vel.z).sqrt();
            if speed < 0.2 {
                // 速度太小，重置候选计数
                obj.dynamic_candidate_count = 0;
                continue;
            }

            // 对每个新点找 NN 旧点，投票
            let mut votes = 0usize;
            let total = new_pts.len().min(old_pts.len()); // 对称比较
            if total == 0 {
                continue;
            }

            for i in 0..total {
                let np = new_pts[i];
                // 在 old_pts 中找一个距离最近的旧点
                let best_old = old_pts.iter()
                    .min_by(|a, b| {
                        let da = (np[0] - a[0]).powi(2) + (np[1] - a[1]).powi(2) + (np[2] - a[2]).powi(2);
                        let db = (np[0] - b[0]).powi(2) + (np[1] - b[1]).powi(2) + (np[2] - b[2]).powi(2);
                        da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                    });

                if let Some(op) = best_old {
                    let dx = np[0] - op[0];
                    let dy = np[1] - op[1];
                    let dz = np[2] - op[2];
                    // 方向与 KF 速度点乘 > 0 → 方向一致
                    let dot = dx * vel.x as f32 + dy * vel.y as f32 + dz * vel.z as f32;
                    if dot > 0.0 {
                        votes += 1;
                    }
                }
            }

            let ratio = votes as f32 / total as f32;
            if ratio >= vote_threshold {
                obj.dynamic_candidate_count += 1;
                if obj.dynamic_candidate_count >= consistency_frames as u32 {
                    set_classification(obj, TargetClass::Dynamic);
                }
            } else {
                obj.dynamic_candidate_count = 0;
            }
        }
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

    pub async fn run(&mut self) -> Result<(), TrackerError> {
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
                    detection.centroid,
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

        // 步骤 7: 速度聚类分类
        Self::classify_by_velocity(&mut self.tracked_objects);

        // 步骤 8: 点云投票动态分类（LV-DOT 启发）
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
                );
                Self::classify_by_point_cloud_voting(
                    &mut self.tracked_objects,
                    self.point_cloud_vote_threshold,
                    self.point_cloud_skip_frames,
                    self.point_cloud_consistency_frames,
                );
            }
        }

        // 步骤 9: 箱体尺寸锁定 fix_size
        if self.use_fix_size {
            for obj in self.tracked_objects.values_mut() {
                obj.apply_fix_size(self.fix_size_frames, self.fix_size_dim_thresh);
            }
        }

        // 步骤 10: 生成输出（过滤短命目标）
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
                    TargetClass::Dynamic => "dynamic",
                    TargetClass::Movable => "movable",
                    TargetClass::Unknown => "unknown",
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
                    is_dynamic: obj.classification == TargetClass::Dynamic,
                    classification: class_str.to_string(),
                };

                Self::refine_classification_with_sight(&mut target, &sight_data);
                output_targets.push(target);
            }
        }

        // 步骤 10: 写入输出
        {
            let mut target_guard = self.target.lock().await;
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
