use ndarray::{Array1, Array2};
use ndarray_linalg::Inverse;
use std::f64;

/// 卡尔曼滤波器的错误类型定义
#[derive(Debug)]
pub enum KalmanError {
    MatrixError(String),
    InvalidInput(String),
}

impl std::fmt::Display for KalmanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KalmanError::MatrixError(msg) => write!(f, "Matrix error: {}", msg),
            KalmanError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
        }
    }
}

impl std::error::Error for KalmanError {}

/// 卡尔曼滤波器结构体
pub struct KalmanFilter {
    /// 状态向量 [x, y, z, vx, vy, vz] - 位置和速度
    state: Array1<f64>,
    /// 状态协方差矩阵
    covariance: Array2<f64>,
    /// 状态转移矩阵
    transition_matrix: Array2<f64>,
    /// 观测矩阵
    observation_matrix: Array2<f64>,
    /// 过程噪声协方差
    process_noise: Array2<f64>,
    /// 测量噪声协方差
    measurement_noise: Array2<f64>,
    /// 控制输入矩阵 (暂时不使用)
    _control_matrix: Option<Array2<f64>>,
    /// 初始化标志
    initialized: bool,
}

impl KalmanFilter {
    /// 创建一个新的卡尔曼滤波器实例
    pub fn new(state_dim: usize, measurement_dim: usize) -> Result<Self, KalmanError> {
        // 默认使用6维状态向量 (x, y, z, vx, vy, vz)
        let state = Array1::zeros(state_dim);

        // 状态协方差矩阵，初始化为单位矩阵
        let covariance = Array2::eye(state_dim);

        // 状态转移矩阵 (假设匀速模型)
        let transition_matrix = Array2::eye(state_dim);
        // 在匀速模型中，位置会根据速度变化

        // 观测矩阵 (只观测位置信息)
        let mut observation_matrix = Array2::zeros((measurement_dim, state_dim));
        for i in 0..measurement_dim {
            observation_matrix[[i, i]] = 1.0;
        }

        // 过程噪声协方差
        let process_noise = Array2::eye(state_dim) * 0.1;

        // 测量噪声协方差
        let measurement_noise = Array2::eye(measurement_dim) * 0.1;

        Ok(KalmanFilter {
            state,
            covariance,
            transition_matrix,
            observation_matrix,
            process_noise,
            measurement_noise,
            _control_matrix: None,
            initialized: false,
        })
    }

    /// 使用指定的参数初始化卡尔曼滤波器
    pub fn init(
        &mut self,
        initial_state: Array1<f64>,
        initial_covariance: Array2<f64>,
        delta_time: f64,
    ) -> Result<(), KalmanError> {
        if initial_state.len() != self.state.len() {
            return Err(KalmanError::InvalidInput(
                "Initial state dimension mismatch".to_string(),
            ));
        }

        if initial_covariance.shape()[0] != initial_covariance.shape()[1]
            || initial_covariance.shape()[0] != self.state.len()
        {
            return Err(KalmanError::InvalidInput(
                "Initial covariance matrix dimension mismatch".to_string(),
            ));
        }

        // 更新状态转移矩阵，基于delta_time
        self.update_transition_matrix(delta_time);

        self.state = initial_state;
        self.covariance = initial_covariance;
        self.initialized = true;

        Ok(())
    }

    /// 更新状态转移矩阵，基于时间增量
    fn update_transition_matrix(&mut self, dt: f64) {
        // 对于匀速模型，位置会根据速度变化: pos_new = pos_old + vel * dt
        let state_dim = self.state.len();

        // 重置为单位矩阵
        self.transition_matrix = Array2::eye(state_dim);

        // 设置位置对速度的转移关系 (前3个位置元素受后3个速度元素影响)
        for i in 0..3 {
            self.transition_matrix[[i, i + 3]] = dt;
        }
    }

    /// 预测步骤：基于系统模型预测下一状态
    pub fn predict(&mut self, dt: f64) -> Result<(), KalmanError> {
        if !self.initialized {
            return Err(KalmanError::InvalidInput(
                "Kalman filter not initialized".to_string(),
            ));
        }

        // 更新状态转移矩阵
        self.update_transition_matrix(dt);

        // 预测状态: x_pred = F * x
        let predicted_state = self.transition_matrix.dot(&self.state);

        // 预测协方差: P_pred = F * P * F^T + Q
        let temp = self.transition_matrix.dot(&self.covariance);
        let predicted_covariance = temp.dot(&self.transition_matrix.t()) + &self.process_noise;

        self.state = predicted_state;
        self.covariance = predicted_covariance;

        Ok(())
    }

    /// 更新步骤：使用测量值更新状态估计
    pub fn update(&mut self, measurement: &Array1<f64>) -> Result<(), KalmanError> {
        if !self.initialized {
            return Err(KalmanError::InvalidInput(
                "Kalman filter not initialized".to_string(),
            ));
        }

        let measurement_dim = self.observation_matrix.shape()[0];
        if measurement.len() != measurement_dim {
            return Err(KalmanError::InvalidInput(format!(
                "Measurement dimension mismatch: expected {}, got {}",
                measurement_dim,
                measurement.len()
            )));
        }

        // 计算卡尔曼增益
        // S = H * P * H^T + R
        let temp = self
            .observation_matrix
            .dot(&self.covariance)
            .dot(&self.observation_matrix.t());
        let innovation_covariance = temp + &self.measurement_noise;

        // K = P * H^T * S^(-1)
        let kalman_gain = match innovation_covariance.inv() {
            Ok(inv) => {
                let gain = self.covariance.dot(&self.observation_matrix.t()).dot(&inv);
                gain
            }
            Err(_) => {
                return Err(KalmanError::MatrixError(
                    "Cannot compute inverse of innovation covariance".to_string(),
                ));
            }
        };

        // 创新 (测量残差): y = z - H * x_pred
        let predicted_measurement = self.observation_matrix.dot(&self.state);
        let innovation = measurement - &predicted_measurement;

        // 更新状态: x_upd = x_pred + K * y
        let state_update = kalman_gain.dot(&innovation);
        self.state = &self.state + &state_update;

        // 更新协方差: P_upd = (I - K * H) * P_pred
        let gain_h_product = kalman_gain.dot(&self.observation_matrix);
        let identity = Array2::eye(self.state.len());
        let updated_covariance = (&identity - &gain_h_product).dot(&self.covariance);

        self.covariance = updated_covariance;

        Ok(())
    }

    /// 获取当前状态估计
    pub fn get_state(&self) -> &Array1<f64> {
        &self.state
    }

    /// 获取当前协方差矩阵
    pub fn get_covariance(&self) -> &Array2<f64> {
        &self.covariance
    }

    /// 获取位置信息 (前三个维度)
    pub fn get_position(&self) -> (f64, f64, f64) {
        (self.state[0], self.state[1], self.state[2])
    }

    /// 获取速度信息 (后三个维度)
    pub fn get_velocity(&self) -> (f64, f64, f64) {
        (self.state[3], self.state[4], self.state[5])
    }
}

#[cfg(test)]
mod tests {
    use ndarray::arr1;

    use super::*;

    #[test]
    fn test_kalman_filter_creation() {
        let kf = KalmanFilter::new(6, 3).expect("Failed to create Kalman filter");
        assert_eq!(kf.state.len(), 6);
        assert_eq!(kf.observation_matrix.shape()[0], 3);
    }

    #[test]
    fn test_kalman_filter_init() {
        let mut kf = KalmanFilter::new(6, 3).unwrap();
        let initial_state = arr1(&[1.0, 2.0, 3.0, 0.1, 0.2, 0.3]);
        let initial_covariance = Array2::eye(6) * 0.1;

        kf.init(initial_state, initial_covariance, 0.1).unwrap();
        assert!(kf.initialized);
    }

    #[test]
    fn test_predict_and_update() {
        let mut kf = KalmanFilter::new(6, 3).unwrap();
        let initial_state = arr1(&[1.0, 2.0, 3.0, 0.1, 0.2, 0.3]);
        let initial_covariance = Array2::eye(6) * 0.1;

        kf.init(initial_state, initial_covariance, 0.1).unwrap();

        // 预测
        kf.predict(0.1).unwrap();

        // 更新
        let measurement = arr1(&[1.01, 2.02, 3.03]);
        kf.update(&measurement).unwrap();

        assert!(kf.get_position().0.is_finite());
        assert!(kf.get_velocity().0.is_finite());
    }
}
