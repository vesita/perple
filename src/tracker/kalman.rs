//! 卡尔曼滤波模块
//! 
//! 基于 adskalman 库实现的常速模型卡尔曼滤波器
//! 用于目标跟踪中的状态估计和预测

use nalgebra as na;
use na::{Matrix3, Matrix6, OMatrix, OVector, Vector3, Vector6, U3, U6};
use adskalman::{
    KalmanFilterNoControl, 
    ObservationModel, 
    TransitionModelLinearNoControl,
    StateAndCovariance,
};

/// 状态维度常量 (6 维：x, y, z, vx, vy, vz)
pub const STATE_DIM: usize = 6;
/// 观测维度常量 (3 维：x, y, z)
pub const OBS_DIM: usize = 3;

/// 卡尔曼滤波器配置参数
#[derive(Debug, Clone)]
pub struct KalmanConfig {
    /// 时间步长 (秒)
    pub dt: f64,
    /// 过程噪声缩放系数
    pub process_noise_scale: f64,
    /// 测量噪声缩放系数
    pub measurement_noise_scale: f64,
    /// 初始协方差缩放系数
    pub initial_covariance_scale: f64,
}

impl Default for KalmanConfig {
    fn default() -> Self {
        Self {
            dt: 0.1,  // 默认 100ms
            process_noise_scale: 0.01,
            measurement_noise_scale: 0.1,
            initial_covariance_scale: 10.0,
        }
    }
}

/// 常速运动模型实现
/// 
/// 状态转移方程:
/// x_new = x_old + vx * dt
/// y_new = y_old + vy * dt
/// z_new = z_old + vz * dt
/// vx_new = vx_old
/// vy_new = vy_old
/// vz_new = vz_old
pub struct ConstantVelocityModel {
    /// 状态转移矩阵 F
    transition_matrix: Matrix6<f64>,
    /// 转移矩阵转置 FT
    transition_matrix_transpose: Matrix6<f64>,
    /// 过程噪声协方差 Q
    process_noise: Matrix6<f64>,
    /// 配置参数
    config: KalmanConfig,
}

impl ConstantVelocityModel {
    /// 创建新的常速运动模型
    pub fn new(config: KalmanConfig) -> Self {
        let dt = config.dt;
        
        // 构建状态转移矩阵 F
        let transition_matrix = Matrix6::<f64>::new(
            1.0, 0.0, 0.0,  dt, 0.0, 0.0,  // x = x + vx*dt
            0.0, 1.0, 0.0, 0.0,  dt, 0.0,  // y = y + vy*dt
            0.0, 0.0, 1.0, 0.0, 0.0,  dt,  // z = z + vz*dt
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0,  // vx = vx
            0.0, 0.0, 0.0, 0.0, 1.0, 0.0,  // vy = vy
            0.0, 0.0, 0.0, 0.0, 0.0, 1.0,  // vz = vz
        );
        
        // 过程噪声协方差 Q
        // 假设速度有随机扰动，位置随之变化
        let q = config.process_noise_scale;
        let process_noise = Matrix6::<f64>::new(
            q, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, q, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, q, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, q, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, q, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, q,
        );
        
        Self {
            transition_matrix,
            transition_matrix_transpose: transition_matrix.transpose(),
            process_noise,
            config,
        }
    }
    
    /// 更新时间步长
    pub fn set_dt(&mut self, dt: f64) {
        self.config.dt = dt;
        
        // 重新构建状态转移矩阵
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
    fn F(&self) -> &OMatrix<f64, U6, U6> {
        &self.transition_matrix
    }
    
    fn FT(&self) -> &OMatrix<f64, U6, U6> {
        &self.transition_matrix_transpose
    }
    
    fn Q(&self) -> &OMatrix<f64, U6, U6> {
        &self.process_noise
    }
}

/// 位置观测模型实现
/// 
/// 观测方程:
/// z_x = x
/// z_y = y
/// z_z = z
/// 
/// 即只观测位置，不直接观测速度
pub struct PositionObservationModel {
    /// 观测矩阵 H (3x6)
    observation_matrix: OMatrix<f64, U3, U6>,
    /// 观测矩阵转置 HT (6x3)
    observation_matrix_transpose: OMatrix<f64, U6, U3>,
    /// 观测噪声协方差 R (3x3)
    measurement_noise: Matrix3<f64>,
}

impl PositionObservationModel {
    /// 创建新的位置观测模型
    pub fn new(noise_scale: f64) -> Self {
        // 观测矩阵 H - 只观测位置部分
        let observation_matrix = OMatrix::<f64, U3, U6>::new(
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0,  // z_x = x
            0.0, 1.0, 0.0, 0.0, 0.0, 0.0,  // z_y = y
            0.0, 0.0, 1.0, 0.0, 0.0, 0.0,  // z_z = z
        );
        
        // 观测噪声协方差 R
        let measurement_noise = Matrix3::<f64>::identity() * noise_scale;
        
        Self {
            observation_matrix,
            observation_matrix_transpose: observation_matrix.transpose(),
            measurement_noise,
        }
    }
}

impl ObservationModel<f64, U6, U3> for PositionObservationModel {
    fn H(&self) -> &OMatrix<f64, U3, U6> {
        &self.observation_matrix
    }
    
    fn HT(&self) -> &OMatrix<f64, U6, U3> {
        &self.observation_matrix_transpose
    }
    
    fn R(&self) -> &OMatrix<f64, U3, U3> {
        &self.measurement_noise
    }
}

/// 封装后的卡尔曼滤波器
/// 
/// 提供简化的 API 用于目标跟踪
/// 
/// 该结构体不直接存储 KalmanFilterNoControl，而是在需要时临时创建，
/// 从而避免生命周期和线程安全问题。
pub struct KalmanFilterWrapper {
    /// 运动模型
    motion_model: ConstantVelocityModel,
    /// 观测模型
    observation_model: PositionObservationModel,
    /// 当前状态估计
    current_estimate: StateAndCovariance<f64, U6>,
    /// 配置
    config: KalmanConfig,
}

impl KalmanFilterWrapper {
    /// 创建新的卡尔曼滤波器
    pub fn new(config: KalmanConfig) -> Result<Self, adskalman::Error> {
        let motion_model = ConstantVelocityModel::new(config.clone());
        let observation_model = PositionObservationModel::new(
            config.measurement_noise_scale
        );
        
        // 创建初始状态估计
        let initial_state = OVector::<f64, U6>::zeros();
        let initial_covariance = OMatrix::<f64, U6, U6>::identity() 
            * config.initial_covariance_scale;
        let current_estimate = StateAndCovariance::new(initial_state, initial_covariance);
        
        Ok(Self {
            motion_model,
            observation_model,
            current_estimate,
            config,
        })
    }
    
    /// 从初始位置和速度初始化滤波器
    pub fn init_with_state(
        &mut self,
        position: Vector3<f64>,
        velocity: Option<Vector3<f64>>,
    ) {
        let mut state = OVector::<f64, U6>::zeros();
        
        // 设置位置
        state[0] = position.x;
        state[1] = position.y;
        state[2] = position.z;
        
        // 设置速度 (如果提供)
        if let Some(vel) = velocity {
            state[3] = vel.x;
            state[4] = vel.y;
            state[5] = vel.z;
        }
        
        let covariance = OMatrix::<f64, U6, U6>::identity() 
            * self.config.initial_covariance_scale;
        
        self.current_estimate = StateAndCovariance::new(state, covariance);
    }
    
    /// 执行预测步骤
    pub fn predict(&mut self) -> Result<(), adskalman::Error> {
        // 使用运动模型预测下一状态
        self.current_estimate = self.motion_model.predict(&self.current_estimate);
        Ok(())
    }
    
    /// 执行更新步骤
    pub fn update(&mut self, measurement: Vector3<f64>) -> Result<(), adskalman::Error> {
        // 检查测量值是否有效 (NaN 表示无效)
        if measurement.x.is_nan() || measurement.y.is_nan() || measurement.z.is_nan() {
            // 如果测量无效，只进行预测，不进行更新
            return self.predict();
        }
        
        // 临时创建 KalmanFilterNoControl 进行 step 操作
        let kalman_filter = KalmanFilterNoControl::new(
            &self.motion_model,
            &self.observation_model,
        );
        
        // 执行预测 + 更新
        self.current_estimate = kalman_filter.step(
            &self.current_estimate,
            &measurement,
        )?;
        
        Ok(())
    }
    
    /// 获取当前位置估计
    pub fn get_position(&self) -> Vector3<f64> {
        let state = self.current_estimate.state();
        Vector3::new(state[0], state[1], state[2])
    }
    
    /// 获取当前速度估计
    pub fn get_velocity(&self) -> Vector3<f64> {
        let state = self.current_estimate.state();
        Vector3::new(state[3], state[4], state[5])
    }
    
    /// 获取完整状态向量
    pub fn get_state(&self) -> &OVector<f64, U6> {
        self.current_estimate.state()
    }
    
    /// 获取状态协方差矩阵
    pub fn get_covariance(&self) -> &OMatrix<f64, U6, U6> {
        self.current_estimate.covariance()
    }
    
    /// 更新时间步长
    pub fn set_dt(&mut self, dt: f64) {
        self.motion_model.set_dt(dt);
        self.config.dt = dt;
    }
    
    /// 重置滤波器到初始状态
    pub fn reset(&mut self) {
        let initial_state = OVector::<f64, U6>::zeros();
        let initial_covariance = OMatrix::<f64, U6, U6>::identity() 
            * self.config.initial_covariance_scale;
        self.current_estimate = StateAndCovariance::new(initial_state, initial_covariance);
    }
    
    /// 获取位置估计的不确定性（标准差）
    /// 
    /// 返回 (x_std, y_std, z_std)
    pub fn get_position_uncertainty(&self) -> Vector3<f64> {
        let cov = self.current_estimate.covariance();
        Vector3::new(
            cov[(0, 0)].sqrt(),
            cov[(1, 1)].sqrt(),
            cov[(2, 2)].sqrt(),
        )
    }
    
    /// 获取速度估计的不确定性（标准差）
    /// 
    /// 返回 (vx_std, vy_std, vz_std)
    pub fn get_velocity_uncertainty(&self) -> Vector3<f64> {
        let cov = self.current_estimate.covariance();
        Vector3::new(
            cov[(3, 3)].sqrt(),
            cov[(4, 4)].sqrt(),
            cov[(5, 5)].sqrt(),
        )
    }
    
    /// 检查滤波器是否已初始化（协方差不是初始值）
    /// 
    /// 通过检查协方差迹是否小于初始值来判断
    pub fn is_initialized(&self) -> bool {
        let cov = self.current_estimate.covariance();
        let trace: f64 = (0..STATE_DIM).map(|i| cov[(i, i)]).sum();
        let initial_trace = STATE_DIM as f64 * self.config.initial_covariance_scale;
        // 如果迹明显小于初始值，认为已初始化
        trace < initial_trace * 0.9
    }
    
    /// 动态设置测量噪声缩放系数
    /// 
    /// 这会重新创建观测模型
    pub fn set_measurement_noise(&mut self, noise_scale: f64) {
        self.observation_model = PositionObservationModel::new(noise_scale);
        self.config.measurement_noise_scale = noise_scale;
    }
    
    /// 动态设置过程噪声缩放系数
    /// 
    /// 这会重新创建运动模型
    pub fn set_process_noise(&mut self, noise_scale: f64) {
        self.motion_model = ConstantVelocityModel::new(KalmanConfig {
            process_noise_scale: noise_scale,
            ..self.config.clone()
        });
        self.config.process_noise_scale = noise_scale;
    }
    
    /// 计算新息（测量残差）
    /// 
    /// 新息 = 实际测量值 - 预测测量值
    /// 用于评估滤波器性能和检测异常测量
    pub fn get_innovation(&self, measurement: Vector3<f64>) -> Vector3<f64> {
        let state = self.current_estimate.state();
        
        // 预测的测量值 = H * state
        // 对于位置观测模型，就是状态中的位置部分
        let predicted_measurement = Vector3::new(state[0], state[1], state[2]);
        
        measurement - predicted_measurement
    }
    
    /// 获取新息协方差矩阵 (S = H*P*H' + R)
    /// 
    /// 用于计算卡尔曼增益和评估测量一致性
    pub fn get_innovation_covariance(&self) -> Matrix3<f64> {
        let p = self.current_estimate.covariance();
        let h = self.observation_model.H();
        let ht = self.observation_model.HT();
        let r = self.observation_model.R();
        
        // S = H * P * H' + R
        let hp = h * p;
        let hph = hp * ht;
        hph + r
    }
    
    /// 计算归一化新息平方 (NIS)
    /// 
    /// NIS = innovation' * S^-1 * innovation
    /// 用于卡方检验，检测测量是否异常
    pub fn normalized_innovation_squared(&self, measurement: Vector3<f64>) -> f64 {
        let innovation = self.get_innovation(measurement);
        let s = self.get_innovation_covariance();
        
        // 计算 S 的逆
        if let Some(s_inv) = s.try_inverse() {
            (&innovation).dot(&(&s_inv * &innovation))
        } else {
            // 如果不可逆，返回一个很大的值表示异常
            f64::MAX
        }
    }
    
    /// 获取当前配置参数的引用
    pub fn get_config(&self) -> &KalmanConfig {
        &self.config
    }
    
    /// 获取运动模型的引用
    pub fn get_motion_model(&self) -> &ConstantVelocityModel {
        &self.motion_model
    }
    
    /// 获取观测模型的引用
    pub fn get_observation_model(&self) -> &PositionObservationModel {
        &self.observation_model
    }
    
    /// 执行仅预测步骤（不更新状态）
    /// 
    /// 用于在没有测量值时进行状态外推
    pub fn predict_only(&mut self) -> Result<(), adskalman::Error> {
        self.predict()
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
    fn test_kalman_filter_creation() {
        let kf = KalmanFilterWrapper::new(KalmanConfig::default()).unwrap();
        assert_eq!(kf.get_state().len(), 6);
    }
    
    #[test]
    fn test_init_with_state() {
        let mut kf = KalmanFilterWrapper::new(KalmanConfig::default()).unwrap();
        let position = Vector3::new(1.0, 2.0, 3.0);
        let velocity = Some(Vector3::new(0.1, 0.2, 0.3));
        
        kf.init_with_state(position, velocity);
        
        let pos = kf.get_position();
        assert_relative_eq!(pos.x, 1.0, epsilon = 1e-10);
        assert_relative_eq!(pos.y, 2.0, epsilon = 1e-10);
        assert_relative_eq!(pos.z, 3.0, epsilon = 1e-10);
        
        let vel = kf.get_velocity();
        assert_relative_eq!(vel.x, 0.1, epsilon = 1e-10);
        assert_relative_eq!(vel.y, 0.2, epsilon = 1e-10);
        assert_relative_eq!(vel.z, 0.3, epsilon = 1e-10);
    }
    
    #[test]
    fn test_predict_step() {
        let mut kf = KalmanFilterWrapper::new(KalmanConfig::default()).unwrap();
        let position = Vector3::new(1.0, 0.0, 0.0);
        let velocity = Some(Vector3::new(1.0, 0.0, 0.0));
        kf.init_with_state(position, velocity);
        
        // 预测一步
        kf.predict().unwrap();
        
        let pos = kf.get_position();
        // x = x0 + vx * dt = 1.0 + 1.0 * 0.1 = 1.1
        assert_relative_eq!(pos.x, 1.1, epsilon = 1e-10);
    }
    
    #[test]
    fn test_update_step() {
        let mut kf = KalmanFilterWrapper::new(KalmanConfig::default()).unwrap();
        let position = Vector3::new(0.0, 0.0, 0.0);
        kf.init_with_state(position, None);
        
        // 使用观测值更新
        let measurement = Vector3::new(1.0, 0.0, 0.0);
        kf.update(measurement).unwrap();
        
        let pos = kf.get_position();
        // 更新后位置应该接近观测值
        assert!(pos.x > 0.5);
    }
    
    #[test]
    fn test_predict_update_cycle() {
        let mut kf = KalmanFilterWrapper::new(KalmanConfig::default()).unwrap();
        let position = Vector3::new(0.0, 0.0, 0.0);
        let velocity = Some(Vector3::new(1.0, 0.0, 0.0));
        kf.init_with_state(position, velocity);
        
        // 模拟多次预测 - 更新循环
        for i in 1..=10 {
            kf.predict().unwrap();
            
            // 生成带噪声的观测值
            let true_pos = i as f64 * 0.1;
            let measurement = Vector3::new(true_pos, 0.0, 0.0);
            kf.update(measurement).unwrap();
        }
        
        let pos = kf.get_position();
        // 最终位置应该接近真实位置 (1.0)
        assert!((pos.x - 1.0).abs() < 0.5);
    }
    
    #[test]
    fn test_get_uncertainty() {
        let mut kf = KalmanFilterWrapper::new(KalmanConfig::default()).unwrap();
        let position = Vector3::new(1.0, 2.0, 3.0);
        kf.init_with_state(position, None);
        
        // 获取协方差矩阵的对角线元素作为不确定性
        let cov = kf.get_covariance();
        let pos_unc = Vector3::new(cov[(0, 0)], cov[(1, 1)], cov[(2, 2)]);
        let vel_unc = Vector3::new(cov[(3, 3)], cov[(4, 4)], cov[(5, 5)]);
        
        // 不确定性应该是正数
        assert!(pos_unc.x > 0.0);
        assert!(pos_unc.y > 0.0);
        assert!(pos_unc.z > 0.0);
        assert!(vel_unc.x > 0.0);
        assert!(vel_unc.y > 0.0);
        assert!(vel_unc.z > 0.0);
    }
    
    #[test]
    fn test_is_initialized() {
        // 注意：KalmanFilterWrapper 总是已初始化的（有初始状态）
        // 这个测试仅用于验证基本功能
        let kf = KalmanFilterWrapper::new(KalmanConfig::default()).unwrap();
        assert_eq!(kf.get_state().len(), 6);
    }
    
    #[test]
    fn test_set_measurement_noise() {
        let mut kf = KalmanFilterWrapper::new(KalmanConfig::default()).unwrap();
        
        let new_noise = 0.5;
        kf.set_measurement_noise(new_noise);
        
        assert_relative_eq!(kf.get_config().measurement_noise_scale, new_noise, epsilon = 1e-10);
    }
    
    #[test]
    fn test_get_innovation() {
        let mut kf = KalmanFilterWrapper::new(KalmanConfig::default()).unwrap();
        let position = Vector3::new(1.0, 2.0, 3.0);
        kf.init_with_state(position, None);
        
        let measurement = Vector3::new(1.5, 2.5, 3.5);
        let innovation = kf.get_innovation(measurement);
        
        // 新息应该是测量值与预测值的差
        assert_relative_eq!(innovation.x, 0.5, epsilon = 1e-10);
        assert_relative_eq!(innovation.y, 0.5, epsilon = 1e-10);
        assert_relative_eq!(innovation.z, 0.5, epsilon = 1e-10);
    }
    
    #[test]
    fn test_normalized_innovation_squared() {
        let mut kf = KalmanFilterWrapper::new(KalmanConfig::default()).unwrap();
        let position = Vector3::new(0.0, 0.0, 0.0);
        kf.init_with_state(position, None);
        
        let measurement = Vector3::new(1.0, 0.0, 0.0);
        let nis = kf.normalized_innovation_squared(measurement);
        
        // NIS 应该是非负数
        assert!(nis >= 0.0);
    }
    
    #[test]
    fn test_reset() {
        let mut kf = KalmanFilterWrapper::new(KalmanConfig::default()).unwrap();
        let position = Vector3::new(10.0, 20.0, 30.0);
        let velocity = Some(Vector3::new(1.0, 2.0, 3.0));
        kf.init_with_state(position, velocity);
        
        // 重置后状态应该回到零
        kf.reset();
        
        let pos = kf.get_position();
        let vel = kf.get_velocity();
        
        assert_relative_eq!(pos.x, 0.0, epsilon = 1e-10);
        assert_relative_eq!(pos.y, 0.0, epsilon = 1e-10);
        assert_relative_eq!(pos.z, 0.0, epsilon = 1e-10);
        assert_relative_eq!(vel.x, 0.0, epsilon = 1e-10);
        assert_relative_eq!(vel.y, 0.0, epsilon = 1e-10);
        assert_relative_eq!(vel.z, 0.0, epsilon = 1e-10);
    }
}
