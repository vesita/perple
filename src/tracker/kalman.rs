//! 卡尔曼滤波模块
//!
//! 基于常速模型（CV Model）的卡尔曼滤波器，6 维状态：
//!   [x, y, z, vx, vy, vz]
//!
//! 测量值（LV-DOT 风格）：6 维 [x, y, z, vx, vy, vz]
//! - 位置直接观测量测值
//! - 速度通过 k 帧位置差计算：(pos_t - pos_{t-k}) / (k * dt)
//!
//! 关键设计：
//! - predict() 和 correct() 分离，避免重复预测
//! - 动态 dt：基于帧间隔时间戳实时计算
//! - Q/R 对位置和速度分别设值（LV-DOT 启发）

use nalgebra as na;
use na::{Matrix3, Matrix6, OMatrix, OVector, Vector3, Vector6, U6};
use adskalman::{
    ObservationModel,
    TransitionModelLinearNoControl,
    StateAndCovariance,
};

/// 状态维度 (6：x, y, z, vx, vy, vz)
pub const STATE_DIM: usize = 6;
/// 观测维度 (6：x, y, z, vx, vy, vz)
pub const OBS_DIM: usize = 6;

#[derive(Debug, Clone)]
pub struct KalmanConfig {
    pub dt: f64,
    pub process_noise_pos: f64,
    pub process_noise_vel: f64,
    pub measurement_noise_pos: f64,
    pub measurement_noise_vel: f64,
    pub initial_covariance_scale: f64,
}

impl Default for KalmanConfig {
    fn default() -> Self {
        Self {
            dt: 0.04,
            process_noise_pos: 0.1,
            process_noise_vel: 0.02,
            measurement_noise_pos: 0.2,
            measurement_noise_vel: 0.1,
            initial_covariance_scale: 0.5,
        }
    }
}

/// 常速运动模型
struct ConstantVelocityModel {
    transition_matrix: Matrix6<f64>,
    transition_matrix_transpose: Matrix6<f64>,
    process_noise: Matrix6<f64>,
    config: KalmanConfig,
}

impl ConstantVelocityModel {
    fn new(config: KalmanConfig) -> Self {
        let dt = config.dt;
        let transition_matrix = Matrix6::<f64>::new(
            1.0, 0.0, 0.0,  dt, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,  dt, 0.0,
            0.0, 0.0, 1.0, 0.0, 0.0,  dt,
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        );
        // LV-DOT 风格：位置和速度的 process noise 分开
        let q_pos = config.process_noise_pos;
        let q_vel = config.process_noise_vel;
        let process_noise = Matrix6::<f64>::from_diagonal(&Vector6::new(
            q_pos, q_pos, q_pos, q_vel, q_vel, q_vel,
        ));
        Self {
            transition_matrix,
            transition_matrix_transpose: transition_matrix.transpose(),
            process_noise,
            config,
        }
    }

    fn set_dt(&mut self, dt: f64) {
        self.config.dt = dt;
        self.transition_matrix = Matrix6::<f64>::new(
            1.0, 0.0, 0.0,  dt, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,  dt, 0.0,
            0.0, 0.0, 1.0, 0.0, 0.0,  dt,
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        );
        self.transition_matrix_transpose = self.transition_matrix.transpose();
    }
}

impl TransitionModelLinearNoControl<f64, U6> for ConstantVelocityModel {
    fn F(&self) -> &OMatrix<f64, U6, U6> { &self.transition_matrix }
    fn FT(&self) -> &OMatrix<f64, U6, U6> { &self.transition_matrix_transpose }
    fn Q(&self) -> &OMatrix<f64, U6, U6> { &self.process_noise }
}

/// 全状态观测模型（LV-DOT 风格）：同时观测位置和速度
///
/// H = I_6（直接观测全部 6 个状态）
struct FullStateObservationModel {
    observation_matrix: Matrix6<f64>,
    observation_matrix_transpose: Matrix6<f64>,
    measurement_noise: Matrix6<f64>,
}

impl FullStateObservationModel {
    fn new(noise_pos: f64, noise_vel: f64) -> Self {
        let observation_matrix = Matrix6::<f64>::identity();
        let measurement_noise = Matrix6::<f64>::from_diagonal(&Vector6::new(
            noise_pos, noise_pos, noise_pos, noise_vel, noise_vel, noise_vel,
        ));
        Self {
            observation_matrix,
            observation_matrix_transpose: observation_matrix.transpose(),
            measurement_noise,
        }
    }
}

impl ObservationModel<f64, U6, U6> for FullStateObservationModel {
    fn H(&self) -> &OMatrix<f64, U6, U6> { &self.observation_matrix }
    fn HT(&self) -> &OMatrix<f64, U6, U6> { &self.observation_matrix_transpose }
    fn R(&self) -> &OMatrix<f64, U6, U6> { &self.measurement_noise }
}

/// 封装的卡尔曼滤波器，提供 predict/correct 分离 API
pub struct KalmanFilterWrapper {
    motion_model: ConstantVelocityModel,
    observation_model: FullStateObservationModel,
    current_estimate: StateAndCovariance<f64, U6>,
    config: KalmanConfig,
}

impl KalmanFilterWrapper {
    pub fn new(config: KalmanConfig) -> Result<Self, adskalman::Error> {
        let motion_model = ConstantVelocityModel::new(config.clone());
        let observation_model = FullStateObservationModel::new(
            config.measurement_noise_pos,
            config.measurement_noise_vel,
        );
        let initial_state = OVector::<f64, U6>::zeros();
        let initial_covariance = OMatrix::<f64, U6, U6>::identity() * config.initial_covariance_scale;
        let current_estimate = StateAndCovariance::new(initial_state, initial_covariance);
        Ok(Self { motion_model, observation_model, current_estimate, config })
    }

    /// 用初始位置（和可选速度）初始化滤波器
    pub fn init_with_state(&mut self, position: Vector6<f64>) {
        let mut state = OVector::<f64, U6>::zeros();
        state[0] = position[0];
        state[1] = position[1];
        state[2] = position[2];
        state[3] = position[3];
        state[4] = position[4];
        state[5] = position[5];
        let covariance = OMatrix::<f64, U6, U6>::identity() * self.config.initial_covariance_scale;
        self.current_estimate = StateAndCovariance::new(state, covariance);
    }

    /// 预测：将状态前推 dt 秒
    pub fn predict(&mut self, dt: f64) -> Result<(), adskalman::Error> {
        self.motion_model.set_dt(dt);
        self.current_estimate = self.motion_model.predict(&self.current_estimate);
        Ok(())
    }

    /// 修正（LV-DOT 风格）：用 6 维观测 [x,y,z,vx,vy,vz] 校正
    ///
    /// 速度观测通过 k 帧位置差计算，由上层调用者提供。
    pub fn correct(&mut self, measurement: Vector6<f64>) -> Result<(), adskalman::Error> {
        let x = self.current_estimate.state();
        let p = self.current_estimate.covariance();
        let h = self.observation_model.H();
        let r = self.observation_model.R();

        // innovation = z - H * x
        let y = measurement - (h * x);
        // S = H * P * H^T + R
        let s = h * p * h.transpose() + r;
        // K = P * H^T * S^-1
        let si = s.try_inverse().ok_or(adskalman::Error::CovarianceNotPositiveSemiDefinite)?;
        let k = p * h.transpose() * si;
        // x = x + K * y
        let new_x = x + &k * y;
        // P = (I - K * H) * P
        let i = OMatrix::<f64, U6, U6>::identity();
        let new_p = (i - &k * h) * p;

        self.current_estimate = StateAndCovariance::new(new_x, new_p);
        Ok(())
    }

    // ==================== 查询接口 ====================

    pub fn get_position(&self) -> Vector3<f64> {
        let s = self.current_estimate.state();
        Vector3::new(s[0], s[1], s[2])
    }

    pub fn get_velocity(&self) -> Vector3<f64> {
        let s = self.current_estimate.state();
        Vector3::new(s[3], s[4], s[5])
    }

    /// 返回全状态 [x,y,z,vx,vy,vz]
    pub fn get_full_state(&self) -> Vector6<f64> {
        let s = self.current_estimate.state();
        Vector6::new(s[0], s[1], s[2], s[3], s[4], s[5])
    }

    pub fn get_state(&self) -> &OVector<f64, U6> {
        self.current_estimate.state()
    }

    pub fn get_covariance(&self) -> &OMatrix<f64, U6, U6> {
        self.current_estimate.covariance()
    }

    pub fn get_position_uncertainty(&self) -> Vector6<f64> {
        let cov = self.current_estimate.covariance();
        Vector6::new(
            cov[(0, 0)].sqrt(), cov[(1, 1)].sqrt(), cov[(2, 2)].sqrt(),
            cov[(3, 3)].sqrt(), cov[(4, 4)].sqrt(), cov[(5, 5)].sqrt(),
        )
    }

    /// 马氏距离（位置分量，用于数据关联门控）
    ///
    /// 仅使用 [x,y,z] 位置分量计算，因为检测值不含速度信息。
    /// d = sqrt(innovation_pos^T * S_pos^-1 * innovation_pos)
    /// 服从 χ²(3) 分布，α=0.05 时阈值为 sqrt(7.815) ≈ 2.795
    pub fn mahalanobis_distance(&self, measurement: Vector3<f64>) -> f64 {
        let s = self.current_estimate.state();
        let innovation = Vector3::new(
            measurement.x - s[0],
            measurement.y - s[1],
            measurement.z - s[2],
        );
        // 取 S 的 3x3 位置子矩阵
        let cov = self.current_estimate.covariance();
        let h = self.observation_model.H();
        let r = self.observation_model.R();
        let s_full = h * cov * h.transpose() + r;
        let s_pos = s_full.fixed_view::<3, 3>(0, 0);
        s_pos.try_inverse()
            .map(|s_inv| (&innovation).dot(&(&s_inv * &innovation)).sqrt())
            .unwrap_or(f64::MAX)
    }

    /// 全状态马氏距离（6D）
    pub fn mahalanobis_distance_full(&self, measurement: Vector6<f64>) -> f64 {
        let innovation = measurement - self.current_estimate.state();
        let cov = self.current_estimate.covariance();
        let h = self.observation_model.H();
        let r = self.observation_model.R();
        let s = h * cov * h.transpose() + r;
        s.try_inverse()
            .map(|s_inv| (&innovation).dot(&(&s_inv * &innovation)).sqrt())
            .unwrap_or(f64::MAX)
    }

    pub fn get_innovation_covariance(&self) -> Matrix6<f64> {
        let p = self.current_estimate.covariance();
        let h = self.observation_model.H();
        let ht = self.observation_model.HT();
        let r = self.observation_model.R();
        h * p * ht + r
    }

    /// 位置-only 修正（用于早期帧，速度测量尚不可靠时）
    ///
    /// 使用 H_pos = [I_3 | 0_3x3]（3x6），仅修正位置子空间。
    /// 速度通过预测步骤保持（P_vp 交叉协方差会产生少量间接更新）。
    /// 待 position_history 积累足够帧数后切换到全状态 6D 修正。
    pub fn correct_position(&mut self, measurement: Vector3<f64>) -> Result<(), adskalman::Error> {
        let x = self.current_estimate.state().clone();
        let p = self.current_estimate.covariance().clone();
        let r_val = self.config.measurement_noise_pos;

        // innovation = z - H_pos * x
        let innovation = Vector3::new(measurement.x - x[0], measurement.y - x[1], measurement.z - x[2]);

        // S_pos = P[0:3, 0:3] + R_pos
        let s = p.fixed_view::<3, 3>(0, 0) + Matrix3::identity() * r_val;
        let s_inv = s.try_inverse()
            .ok_or(adskalman::Error::CovarianceNotPositiveSemiDefinite)?;

        // K = P * H_pos^T * S^(-1) = P[:, 0:3] * S^(-1)  (6x3)
        let k = p.columns(0, 3) * s_inv;

        // x_new = x + K * innovation
        let new_x = x + &k * innovation;

        // P_new = P - K * H_pos * P = P - K * P[0:3, :]
        let p_slices = p.clone();
        let hp = p_slices.fixed_view::<3, 6>(0, 0);
        let new_p = p - k * hp;

        self.current_estimate = StateAndCovariance::new(new_x, new_p);
        Ok(())
    }

    pub fn reset(&mut self) {
        let initial_state = OVector::<f64, U6>::zeros();
        let initial_covariance = OMatrix::<f64, U6, U6>::identity() * self.config.initial_covariance_scale;
        self.current_estimate = StateAndCovariance::new(initial_state, initial_covariance);
    }

    /// 根据目标距离动态调整测量噪声（Phase 9D）
    ///
    /// 远距离目标点云更稀疏 → 质心更不可靠 → 增加测量噪声。
    /// 噪声缩放：基准值 × (1.0 + distance / distance_scale)
    /// - distance < 10m: 接近基准值（质心可靠）
    /// - distance > 30m: 大幅增加噪声（质心不可靠）
    pub fn adjust_noise_for_distance(&mut self, distance: f64) {
        let scale = 1.0 + distance / 20.0; // 10m→1.5x, 30m→2.5x
        let noise_pos = self.config.measurement_noise_pos * scale;
        let noise_vel = self.config.measurement_noise_vel * scale.min(2.0); // 速度噪声最多 2x
        self.observation_model = FullStateObservationModel::new(noise_pos, noise_vel);
    }

    pub fn set_measurement_noise(&mut self, noise_pos: f64, noise_vel: f64) {
        self.observation_model = FullStateObservationModel::new(noise_pos, noise_vel);
        self.config.measurement_noise_pos = noise_pos;
        self.config.measurement_noise_vel = noise_vel;
    }

    pub fn set_process_noise(&mut self, noise_pos: f64, noise_vel: f64) {
        self.motion_model = ConstantVelocityModel::new(KalmanConfig {
            process_noise_pos: noise_pos,
            process_noise_vel: noise_vel,
            ..self.config.clone()
        });
        self.config.process_noise_pos = noise_pos;
        self.config.process_noise_vel = noise_vel;
    }

    /// 事后限幅：限制速度分量在 [-max_speed, max_speed] 内
    ///
    /// 用于防止关联错误导致的单帧速度尖峰。
    /// 在每帧 correct() 之后调用。
    pub fn clamp_velocity(&mut self, max_speed: f64) {
        let mut state = self.current_estimate.state().clone();
        state[3] = state[3].clamp(-max_speed, max_speed);
        state[4] = state[4].clamp(-max_speed, max_speed);
        state[5] = state[5].clamp(-max_speed, max_speed);
        self.current_estimate = StateAndCovariance::new(state, self.current_estimate.covariance().clone());
    }
}

impl Default for KalmanFilterWrapper {
    fn default() -> Self {
        Self::new(KalmanConfig::default()).expect("Failed to create default Kalman filter")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_kalman_creation() {
        let kf = KalmanFilterWrapper::new(KalmanConfig::default()).unwrap();
        assert_eq!(kf.get_state().len(), 6);
    }

    #[test]
    fn test_init_with_state() {
        let mut kf = KalmanFilterWrapper::new(KalmanConfig::default()).unwrap();
        // 使用 Vector6 初始化：[x, y, z, vx, vy, vz]
        let initial_state = Vector6::new(1.0, 2.0, 3.0, 0.1, 0.2, 0.3);
        kf.init_with_state(initial_state);
        assert_relative_eq!(kf.get_position().x, 1.0, epsilon = 1e-10);
        assert_relative_eq!(kf.get_velocity().x, 0.1, epsilon = 1e-10);
    }

    #[test]
    fn test_predict_then_correct() {
        let mut kf = KalmanFilterWrapper::new(KalmanConfig::default()).unwrap();
        // 初始状态：位置(0,0,0)，速度(1,0,0)
        let initial_state = Vector6::new(0.0, 0.0, 0.0, 1.0, 0.0, 0.0);
        kf.init_with_state(initial_state);

        // 预测一步 (dt=0.1)
        kf.predict(0.1).unwrap();
        let pos = kf.get_position();
        assert_relative_eq!(pos.x, 0.1, epsilon = 1e-10);

        // 用 6 维观测值修正 [x, y, z, vx, vy, vz]
        let measurement = Vector6::new(1.0, 0.0, 0.0, 1.0, 0.0, 0.0);
        kf.correct(measurement).unwrap();
        let pos = kf.get_position();
        assert!(pos.x > 0.1); // 观测值拉向 1.0
    }

    #[test]
    fn test_predict_correct_cycle() {
        let mut kf = KalmanFilterWrapper::new(KalmanConfig::default()).unwrap();
        // 初始状态：位置(0,0,0)，速度(1,0,0)
        let initial_state = Vector6::new(0.0, 0.0, 0.0, 1.0, 0.0, 0.0);
        kf.init_with_state(initial_state);

        for i in 1..=10 {
            let true_pos = i as f64 * 0.1;
            kf.predict(0.1).unwrap();
            // 6 维观测：位置 + 速度
            let measurement = Vector6::new(true_pos, 0.0, 0.0, 1.0, 0.0, 0.0);
            kf.correct(measurement).unwrap();
        }

        let pos = kf.get_position();
        assert!((pos.x - 1.0).abs() < 0.5);
    }

    #[test]
    fn test_mahalanobis_distance() {
        let mut kf = KalmanFilterWrapper::new(KalmanConfig::default()).unwrap();
        // 仅初始化位置，速度为 0
        let initial_state = Vector6::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        kf.init_with_state(initial_state);
        let d = kf.mahalanobis_distance(Vector3::new(1.0, 0.0, 0.0));
        assert!(d > 0.0);
    }

    #[test]
    fn test_reset() {
        let mut kf = KalmanFilterWrapper::new(KalmanConfig::default()).unwrap();
        // 使用 Vector6 初始化
        let initial_state = Vector6::new(10.0, 20.0, 30.0, 1.0, 2.0, 3.0);
        kf.init_with_state(initial_state);
        kf.reset();
        assert_relative_eq!(kf.get_position().x, 0.0, epsilon = 1e-10);
        assert_relative_eq!(kf.get_velocity().x, 0.0, epsilon = 1e-10);
    }
}
