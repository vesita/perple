use std::collections::VecDeque;
use std::time::SystemTime;

use nalgebra::{SVector, Vector2};

use crate::{
    tracker::kalman::{KalmanConfigCA, KalmanFilterCA},
    utils::boxes::Box3D,
};

/// 目标分类（状态机）
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum TargetClass {
    Floating, // 待定——新对象或未确认运动能力
    Static,   // 背景/地面——confirmed 不可移动
    Moving,   // 运动中（confirmed）
    Movable,  // 可运动——曾确认运动，当前静止
}

/// 航迹分级状态
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum TrackStatus {
    Tentative,  // 新生，需连续匹配 N 帧后晋级
    Confirmed,  // 稳定轨迹，输出 + 可入 archive
}

/// 跟踪目标信息（包含卡尔曼滤波器）
pub(crate) struct TrackedObject {
    pub(crate) id: usize,
    pub(crate) class_type: String,
    pub(crate) last_seen: SystemTime,
    pub(crate) disappeared_count: u32,
    pub(crate) appearance_count: u32,
    pub(crate) confidence: f32,
    pub(crate) kalman_filter: KalmanFilterCA,
    pub(crate) velocity_history: VecDeque<[f32; 3]>,
    /// KF 原始速度历史（用于加速度观测，LV-DOT 风格）
    pub(crate) kf_vel_history: VecDeque<[f64; 3]>,
    pub(crate) classification: TargetClass,
    /// 一旦为 true，永不回到 Static/Floating
    pub(crate) confirmed_moving: bool,
    /// 连续点云投票通过帧数（Floating→Moving 用）
    pub(crate) voting_streak: u32,
    /// 连续处于静态簇帧数（Floating→Static 用）
    pub(crate) floating_static_count: u32,
    /// 关联时缓存最近的检测框，避免输出阶段二次搜索
    pub(crate) last_box: Option<Box3D>,
    /// 冻结的箱体尺寸（fix_size 稳定用）
    pub(crate) fixed_box: Option<Box3D>,
    /// 点云历史（环形缓冲区，用于点云投票）
    pub(crate) point_cloud_history: VecDeque<Vec<[f32; 3]>>,
    /// k 帧位置历史（用于 LV-DOT 风格的 k 帧速度观测）
    pub(crate) position_history: VecDeque<[f64; 3]>,
    /// k 帧速度观测平均帧数
    pub(crate) kf_avg_frames: usize,
    /// 1€ Filter 低通滤波质心（Option = 未初始化）
    pub(crate) centroid_lpf: Option<[f64; 3]>,
    /// 上一帧速度大小（用于自适应截止频率）
    pub(crate) centroid_prev_vel_mag: f64,
    /// EMA 平滑后的箱体（替代 fix_size 硬锁定）
    pub(crate) smoothed_box: Option<Box3D>,
    /// 分类转换冷却计数器（Moving↔Movable 迟滞）
    pub(crate) class_cooldown: u32,
    /// 速度 EMA 系数（0=自适应置信度）
    pub(crate) vel_smoothing_alpha: f32,
    /// 静态簇缺失连续计数（Static→Floating 迟滞用）
    pub(crate) static_miss_count: u32,
    /// Z 独立 EMA（KF 状态不含 z，用 EMA 平滑跟踪）
    pub(crate) z_ema: f64,
    /// 最后一次更新的点云质心（用于特征级关联的质心偏移一致性）
    pub(crate) last_centroid: [f32; 3],
    /// KF 预测后的 box（在 predict() 中更新，关联时使用）
    pub(crate) predicted_box: Option<Box3D>,
    /// 航迹分级状态
    pub(crate) status: TrackStatus,
    /// 连续匹配帧数（用于 Tentative→Confirmed 晋级）
    pub(crate) consecutive_matches: u32,
    /// 轨迹评分（match +bonus, miss -penalty, 用于生命周期决策）
    pub(crate) score: f64,
}

impl TrackedObject {
    pub(crate) fn new(
        id: usize,
        initial_box: &Box3D,
        class_type: String,
        confidence: f32,
        centroid: [f32; 3],
        kf_avg_frames: usize,
        vel_smoothing_alpha: f32,
    ) -> Result<Self, adskalman::Error> {
        // 9D CA 模型：状态 [x, y, vx, vy, ax, ay, l, w, h]
        let mut kalman_filter = KalmanFilterCA::new(KalmanConfigCA::default())?;
        let init_state = SVector::<f64, 9>::from_column_slice(&[
            centroid[0] as f64, centroid[1] as f64, // x, y
            0.0, 0.0,        // vx, vy
            0.0, 0.0,        // ax, ay
            initial_box.length as f64,
            initial_box.width as f64,
            initial_box.height as f64,
        ]);
        kalman_filter.init_with_state(init_state);
        Ok(Self {
            id,
            class_type,
            last_seen: SystemTime::now(),
            disappeared_count: 0,
            appearance_count: 1, // 创建即计为第一次出现
            confidence,
            kalman_filter,
            velocity_history: VecDeque::with_capacity(10),
            kf_vel_history: VecDeque::with_capacity(16),
            classification: TargetClass::Floating,
            confirmed_moving: false,
            voting_streak: 0,
            floating_static_count: 0,
            last_box: Some(initial_box.clone()),
            fixed_box: None,
            point_cloud_history: VecDeque::with_capacity(16),
            position_history: VecDeque::with_capacity(kf_avg_frames + 2),
            kf_avg_frames,
            centroid_lpf: None,
            centroid_prev_vel_mag: 0.0,
            smoothed_box: None,
            class_cooldown: 0,
            vel_smoothing_alpha,
            static_miss_count: 0,
            z_ema: centroid[2] as f64,
            last_centroid: centroid,
            predicted_box: None,
            status: TrackStatus::Tentative,
            consecutive_matches: 0,
            score: 0.0,
        })
    }

    /// 预测：将状态前推 dt 秒，并更新 predicted_box
    pub(crate) fn predict(&mut self, dt: f64) -> Result<(), adskalman::Error> {
        self.kalman_filter.predict(dt)?;

        // 从 KF 状态构建 predicted_box（位置 + 尺寸来自滤波，pose 来自 last_box）
        if let Some(ref last) = self.last_box {
            let pos = self.kalman_filter.get_position();
            let size = self.kalman_filter.get_size();
            let mut pb = Box3D::from_position_and_angles(
                pos.x as f32, pos.y as f32, self.z_ema as f32,
                0.0, 0.0, 0.0,
                size.x as f32, size.y as f32, size.z as f32,
            );
            pb.pose = last.pose;
            self.predicted_box = Some(pb);
        }
        Ok(())
    }

    /// 修正（LV-DOT 风格）：用 [x,y,z,vx,vy,vz,ax,ay,az] 校正
    ///
    /// - 位置 (x,y) 来自当前帧点云质心 centroid
    /// - 速度 (vx,vy) 通过 k 帧位置差计算
    /// - 加速度 (ax,ay) 通过速度历史差计算
    /// - 尺寸 (l,w,h) 直接观测
    /// - z 用独立 EMA 跟踪（不在 KF 状态中）
    pub(crate) fn correct(
        &mut self,
        new_box: &Box3D,
        new_class_type: String,
        new_confidence: f32,
        centroid: [f32; 3],
    ) -> Result<(), adskalman::Error> {
        let now = SystemTime::now();
        let dt_since_last = now.duration_since(self.last_seen)
            .unwrap_or_default().as_secs_f64().clamp(0.001, 1.0);

        // Z 独立 EMA 平滑
        const Z_ALPHA: f64 = 0.3;
        self.z_ema = Z_ALPHA * centroid[2] as f64 + (1.0 - Z_ALPHA) * self.z_ema;

        // 记录位置历史（保持 3D 用于 z_ema）
        self.position_history.push_back([
            centroid[0] as f64,
            centroid[1] as f64,
            centroid[2] as f64,
        ]);
        if self.position_history.len() > self.kf_avg_frames + 2 {
            self.position_history.pop_front();
        }

        // v = (pos_t - pos_{t-k}) / (k * dt) — 仅 2D
        let hist_len = self.position_history.len();
        let k = self.kf_avg_frames.min(hist_len.saturating_sub(1));
        const MIN_K_FOR_VELOCITY: usize = 3;
        if k >= MIN_K_FOR_VELOCITY {
            let old = self.position_history[hist_len - 1 - k];
            let curr = *self.position_history.back().unwrap();
            let dt_k = (k as f64 * dt_since_last).max(0.001);
            let meas_vx = (curr[0] - old[0]) / dt_k;
            let meas_vy = (curr[1] - old[1]) / dt_k;

            // 加速度：a = (v_t - v_{t-k}) / (k*dt) — 仅 2D
            let vel_hist_len = self.kf_vel_history.len();
            let k_acc = self.kf_avg_frames.min(vel_hist_len);
            const MIN_K_FOR_ACC: usize = 3;
            let (meas_ax, meas_ay) = if k_acc >= MIN_K_FOR_ACC {
                let old_v = self.kf_vel_history[vel_hist_len - k_acc];
                let dt_k_acc = (k_acc as f64 * dt_since_last).max(0.001);
                ((meas_vx - old_v[0]) / dt_k_acc,
                 (meas_vy - old_v[1]) / dt_k_acc)
            } else {
                (0.0, 0.0)
            };

            // 9D 观测：[x, y, vx, vy, ax, ay, l, w, h]
            let measurement = SVector::<f64, 9>::from_column_slice(&[
                centroid[0] as f64, centroid[1] as f64,
                meas_vx, meas_vy,
                meas_ax, meas_ay,
                new_box.length as f64,
                new_box.width as f64,
                new_box.height as f64,
            ]);
            self.kalman_filter.correct(measurement)?;
        } else {
            // 历史不足，仅 (x,y) 位置修正
            self.kalman_filter.correct_position(Vector2::new(
                centroid[0] as f64,
                centroid[1] as f64,
            ))?;
        }

        // 限幅：速度 3.0 m/s，加速度 10.0 m/s²，尺寸 [0.05, 20.0]m
        self.kalman_filter.clamp_state(3.0, 10.0, 0.05, 20.0);

        // 记录 KF 原始速度（用于加速度观测）
        let v = self.kalman_filter.get_velocity();
        if self.kf_vel_history.len() >= 16 {
            self.kf_vel_history.pop_front();
        }
        self.kf_vel_history.push_back([v.x, v.y, v.z]);

        // 记录平滑速度用于聚类
        if self.velocity_history.len() >= 10 {
            self.velocity_history.pop_front();
        }
        self.velocity_history.push_back([v.x as f32, v.y as f32, v.z as f32]);

        // 连续匹配计数（生命周期晋级用）
        self.consecutive_matches += 1;

        self.appearance_count += 1;
        // 保留 person 标签：避免 YOLO 间歇性漏检导致标签闪烁
        if !(self.class_type == "person" && new_class_type != "person") {
            self.class_type = new_class_type;
        }
        self.confidence = new_confidence;
        self.last_seen = SystemTime::now();
        self.last_centroid = centroid;
        self.disappeared_count = 0;
        self.last_box = Some(new_box.clone());
        Ok(())
    }

    /// 帧增长（未匹配时调用）
    pub(crate) fn on_missed(&mut self) {
        self.disappeared_count += 1;
        self.consecutive_matches = 0;
    }

    /// 1€ Filter 质心低通滤波
    ///
    /// 静止时 fc=fc_min → 强平滑；运动时 fc↑ → 低延迟。
    /// 用 KF 速度大小自适应截止频率。
    pub(crate) fn apply_centroid_lpf(&mut self, centroid: &mut [f32; 3], fc_min: f64, beta: f64) {
        let now = SystemTime::now();
        let dt = now.duration_since(self.last_seen)
            .unwrap_or_default().as_secs_f64().clamp(0.001, 1.0);

        let vel_mag = self.kalman_filter.get_velocity().norm() as f64;
        let fc = (fc_min + beta * vel_mag).min(5.0); // fc_max = 5Hz
        let tau = 1.0 / (2.0 * std::f64::consts::PI * fc);
        let alpha = 1.0 / (1.0 + tau / dt);

        if let Some(prev) = self.centroid_lpf {
            centroid[0] = (alpha * centroid[0] as f64 + (1.0 - alpha) * prev[0]) as f32;
            centroid[1] = (alpha * centroid[1] as f64 + (1.0 - alpha) * prev[1]) as f32;
            centroid[2] = (alpha * centroid[2] as f64 + (1.0 - alpha) * prev[2]) as f32;
        }
        self.centroid_lpf = Some([centroid[0] as f64, centroid[1] as f64, centroid[2] as f64]);
        self.centroid_prev_vel_mag = vel_mag;
    }

    pub(crate) fn is_permanently_lost(&self, max_disappeared: u32) -> bool {
        self.disappeared_count >= max_disappeared
    }

    /// 获取 Kalman 估计速度（自适应 EMA 平滑）
    ///
    /// alpha 按置信度自适应：高置信度快速跟踪，低置信度强平滑。
    /// 可通过 `vel_smoothing_alpha` 固定覆盖。
    pub(crate) fn smoothed_velocity(&self) -> [f32; 3] {
        let v = self.kalman_filter.get_velocity();
        // 自适应 alpha：固定值 > 0 则使用固定值，否则按置信度调整
        let alpha = if self.vel_smoothing_alpha > 0.0 {
            self.vel_smoothing_alpha
        } else {
            match self.confidence {
                c if c > 0.9 => 0.4,
                c if c > 0.7 => 0.25,
                _ => 0.15,
            }
        };
        if self.velocity_history.is_empty() {
            return [v.x as f32, v.y as f32, v.z as f32];
        }
        // 历史均值
        let mut avg = [0.0f32; 3];
        for hv in &self.velocity_history {
            avg[0] += hv[0];
            avg[1] += hv[1];
            avg[2] += hv[2];
        }
        let n = self.velocity_history.len() as f32;
        let hist_avg = [avg[0] / n, avg[1] / n, avg[2] / n];
        // EMA: alpha * KF 速度 + (1-alpha) * 历史均值
        [
            alpha * (v.x as f32) + (1.0 - alpha) * hist_avg[0],
            alpha * (v.y as f32) + (1.0 - alpha) * hist_avg[1],
            alpha * (v.z as f32) + (1.0 - alpha) * hist_avg[2],
        ]
    }

    pub(crate) fn speed(&self) -> f32 {
        let v = self.smoothed_velocity();
        (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
    }

    /// 箱体尺寸 EMA 平滑（替代 fix_size 硬锁定）
    ///
    /// alpha 按置信度自适应：高置信度快速跟踪，低置信度强平滑。
    pub(crate) fn apply_box_smoothing(&mut self, base_alpha: f32) {
        let current = match self.last_box {
            Some(ref b) => b.clone(),
            None => return,
        };
        // 置信度自适应 alpha
        let alpha = if self.confidence > 0.9 {
            base_alpha.max(0.4).min(0.5)
        } else if self.confidence > 0.7 {
            base_alpha.max(0.15).min(0.5)
        } else {
            base_alpha.max(0.05).min(0.3)
        };
        let smoothed = match self.smoothed_box {
            Some(ref prev) => {
                let mut b = current.clone();
                b.length = alpha * current.length + (1.0 - alpha) * prev.length;
                b.width  = alpha * current.width  + (1.0 - alpha) * prev.width;
                b.height = alpha * current.height + (1.0 - alpha) * prev.height;
                b
            }
            None => current.clone(),
        };
        self.smoothed_box = Some(smoothed.clone());
        self.last_box = Some(smoothed);
    }

    /// 箱体尺寸锁定：跟踪稳定后如果尺寸变化小则冻结
    pub(crate) fn apply_fix_size(&mut self, min_frames: usize, dim_thresh: f32) {
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

    /// 匹配后更新轨迹评分（上限 clamp）
    pub(crate) fn update_score_on_match(&mut self, bonus: f64, max_score: f64) {
        self.score = (self.score + bonus).min(max_score);
    }

    /// 丢失后更新轨迹评分（下限 0）
    pub(crate) fn update_score_on_miss(&mut self, penalty: f64) {
        self.score = (self.score - penalty).max(0.0);
    }
}
