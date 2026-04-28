//! 卡尔曼滤波模块
//!
//! 基于常速模型（CV Model）的卡尔曼滤波器，6 维状态：
//!   [x, y, z, vx, vy, vz]
//!
//! 测量值：3 维位置 [x, y, z]
//!
//! 关键设计：
//! - predict() 和 correct() 分离，避免重复预测
//! - 动态 dt：基于帧间隔时间戳实时计算
//! - 马氏距离可用于数据关联门控

use nalgebra as na;
use na::{Matrix3, Matrix6, OMatrix, OVector, Vector3, U3, U6};
use adskalman::{
    ObservationModel,
    TransitionModelLinearNoControl,
    StateAndCovariance,
};

/// 状态维度 (6：x, y, z, vx, vy, vz)
pub const STATE_DIM: usize = 6;
/// 观测维度 (3：x, y, z)
pub const OBS_DIM: usize = 3;

#[derive(Debug, Clone)]
pub struct KalmanConfig {
    pub dt: f64,
    pub process_noise_scale: f64,
    pub measurement_noise_scale: f64,
    pub initial_covariance_scale: f64,
}

impl Default for KalmanConfig {
    fn default() -> Self {
        Self {
            dt: 0.04,                              // 默认 40ms（匹配 MultiLoop 间隔）
            process_noise_scale: 0.1,              // 移动场景，适当放宽
            measurement_noise_scale: 0.1,
            initial_covariance_scale: 1.0,         // 收敛更快
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
        let q = config.process_noise_scale;
        let process_noise = Matrix6::<f64>::identity() * q;
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

/// 位置观测模型（只观测位置，不观测速度）
struct PositionObservationModel {
    observation_matrix: OMatrix<f64, U3, U6>,
    observation_matrix_transpose: OMatrix<f64, U6, U3>,
    measurement_noise: Matrix3<f64>,
}

impl PositionObservationModel {
    fn new(noise_scale: f64) -> Self {
        let observation_matrix = OMatrix::<f64, U3, U6>::new(
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
        );
        let measurement_noise = Matrix3::<f64>::identity() * noise_scale;
        Self {
            observation_matrix,
            observation_matrix_transpose: observation_matrix.transpose(),
            measurement_noise,
        }
    }
}

impl ObservationModel<f64, U6, U3> for PositionObservationModel {
    fn H(&self) -> &OMatrix<f64, U3, U6> { &self.observation_matrix }
    fn HT(&self) -> &OMatrix<f64, U6, U3> { &self.observation_matrix_transpose }
    fn R(&self) -> &OMatrix<f64, U3, U3> { &self.measurement_noise }
}

/// 封装的卡尔曼滤波器，提供 predict/correct 分离 API
pub struct KalmanFilterWrapper {
    motion_model: ConstantVelocityModel,
    observation_model: PositionObservationModel,
    current_estimate: StateAndCovariance<f64, U6>,
    config: KalmanConfig,
}

impl KalmanFilterWrapper {
    pub fn new(config: KalmanConfig) -> Result<Self, adskalman::Error> {
        let motion_model = ConstantVelocityModel::new(config.clone());
        let observation_model = PositionObservationModel::new(config.measurement_noise_scale);
        let initial_state = OVector::<f64, U6>::zeros();
        let initial_covariance = OMatrix::<f64, U6, U6>::identity() * config.initial_covariance_scale;
        let current_estimate = StateAndCovariance::new(initial_state, initial_covariance);
        Ok(Self { motion_model, observation_model, current_estimate, config })
    }

    /// 用初始位置（和可选速度）初始化滤波器
    pub fn init_with_state(&mut self, position: Vector3<f64>, velocity: Option<Vector3<f64>>) {
        let mut state = OVector::<f64, U6>::zeros();
        state[0] = position.x;
        state[1] = position.y;
        state[2] = position.z;
        if let Some(vel) = velocity {
            state[3] = vel.x;
            state[4] = vel.y;
            state[5] = vel.z;
        }
        let covariance = OMatrix::<f64, U6, U6>::identity() * self.config.initial_covariance_scale;
        self.current_estimate = StateAndCovariance::new(state, covariance);
    }

    /// 预测：将状态前推 dt 秒
    pub fn predict(&mut self, dt: f64) -> Result<(), adskalman::Error> {
        self.motion_model.set_dt(dt);
        self.current_estimate = self.motion_model.predict(&self.current_estimate);
        Ok(())
    }

    /// 修正：用观测值校正状态（不含预测步骤）
    ///
    /// 直接计算卡尔曼增益并更新，不使用 step()，避免重复预测。
    pub fn correct(&mut self, measurement: Vector3<f64>) -> Result<(), adskalman::Error> {
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

    pub fn get_state(&self) -> &OVector<f64, U6> {
        self.current_estimate.state()
    }

    pub fn get_covariance(&self) -> &OMatrix<f64, U6, U6> {
        self.current_estimate.covariance()
    }

    pub fn get_position_uncertainty(&self) -> Vector3<f64> {
        let cov = self.current_estimate.covariance();
        Vector3::new(cov[(0, 0)].sqrt(), cov[(1, 1)].sqrt(), cov[(2, 2)].sqrt())
    }

    pub fn get_velocity_uncertainty(&self) -> Vector3<f64> {
        let cov = self.current_estimate.covariance();
        Vector3::new(cov[(3, 3)].sqrt(), cov[(4, 4)].sqrt(), cov[(5, 5)].sqrt())
    }

    /// 马氏距离（用于数据关联门控）
    ///
    /// d = sqrt(innovation^T * S^-1 * innovation)
    /// 服从 χ²(3) 分布，α=0.05 时阈值为 sqrt(7.815) ≈ 2.795
    pub fn mahalanobis_distance(&self, measurement: Vector3<f64>) -> f64 {
        let innovation = self.get_innovation(measurement);
        let s = self.get_innovation_covariance();
        s.try_inverse()
            .map(|s_inv| (&innovation).dot(&(&s_inv * &innovation)).sqrt())
            .unwrap_or(f64::MAX)
    }

    pub fn get_innovation(&self, measurement: Vector3<f64>) -> Vector3<f64> {
        let s = self.current_estimate.state();
        measurement - Vector3::new(s[0], s[1], s[2])
    }

    pub fn get_innovation_covariance(&self) -> Matrix3<f64> {
        let p = self.current_estimate.covariance();
        let h = self.observation_model.H();
        let ht = self.observation_model.HT();
        let r = self.observation_model.R();
        h * p * ht + r
    }

    pub fn reset(&mut self) {
        let initial_state = OVector::<f64, U6>::zeros();
        let initial_covariance = OMatrix::<f64, U6, U6>::identity() * self.config.initial_covariance_scale;
        self.current_estimate = StateAndCovariance::new(initial_state, initial_covariance);
    }

    pub fn set_measurement_noise(&mut self, noise_scale: f64) {
        self.observation_model = PositionObservationModel::new(noise_scale);
        self.config.measurement_noise_scale = noise_scale;
    }

    pub fn set_process_noise(&mut self, noise_scale: f64) {
        self.motion_model = ConstantVelocityModel::new(KalmanConfig {
            process_noise_scale: noise_scale,
            ..self.config.clone()
        });
        self.config.process_noise_scale = noise_scale;
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
        kf.init_with_state(Vector3::new(1.0, 2.0, 3.0), Some(Vector3::new(0.1, 0.2, 0.3)));
        assert_relative_eq!(kf.get_position().x, 1.0, epsilon = 1e-10);
        assert_relative_eq!(kf.get_velocity().x, 0.1, epsilon = 1e-10);
    }

    #[test]
    fn test_predict_then_correct() {
        let mut kf = KalmanFilterWrapper::new(KalmanConfig::default()).unwrap();
        kf.init_with_state(Vector3::new(0.0, 0.0, 0.0), Some(Vector3::new(1.0, 0.0, 0.0)));

        // 预测一步 (dt=0.1)
        kf.predict(0.1).unwrap();
        let pos = kf.get_position();
        assert_relative_eq!(pos.x, 0.1, epsilon = 1e-10);

        // 用观测值修正
        kf.correct(Vector3::new(1.0, 0.0, 0.0)).unwrap();
        let pos = kf.get_position();
        assert!(pos.x > 0.1); // 观测值拉向 1.0
    }

    #[test]
    fn test_predict_correct_cycle() {
        let mut kf = KalmanFilterWrapper::new(KalmanConfig::default()).unwrap();
        kf.init_with_state(Vector3::new(0.0, 0.0, 0.0), Some(Vector3::new(1.0, 0.0, 0.0)));

        for i in 1..=10 {
            let true_pos = i as f64 * 0.1;
            kf.predict(0.1).unwrap();
            kf.correct(Vector3::new(true_pos, 0.0, 0.0)).unwrap();
        }

        let pos = kf.get_position();
        assert!((pos.x - 1.0).abs() < 0.5);
    }

    #[test]
    fn test_mahalanobis_distance() {
        let mut kf = KalmanFilterWrapper::new(KalmanConfig::default()).unwrap();
        kf.init_with_state(Vector3::new(0.0, 0.0, 0.0), None);
        let d = kf.mahalanobis_distance(Vector3::new(1.0, 0.0, 0.0));
        assert!(d > 0.0);
    }

    #[test]
    fn test_reset() {
        let mut kf = KalmanFilterWrapper::new(KalmanConfig::default()).unwrap();
        kf.init_with_state(Vector3::new(10.0, 20.0, 30.0), Some(Vector3::new(1.0, 2.0, 3.0)));
        kf.reset();
        assert_relative_eq!(kf.get_position().x, 0.0, epsilon = 1e-10);
        assert_relative_eq!(kf.get_velocity().x, 0.0, epsilon = 1e-10);
    }
}
