use nalgebra as na;
use na::{Matrix2, OMatrix, SVector, Vector2, Vector3, U4};
use adskalman::{ObservationModel, TransitionModelLinearNoControl, StateAndCovariance};

/// 4D CV 模型配置：状态 [x, y, vx, vy]
#[derive(Debug, Clone)]
pub struct KalmanConfigCA {
    pub dt: f64,
    pub process_noise_pos: f64,
    pub process_noise_vel: f64,
    pub measurement_noise_pos: f64,
    pub measurement_noise_vel: f64,
    pub initial_covariance_scale: f64,
}

impl Default for KalmanConfigCA {
    fn default() -> Self {
        Self {
            dt: 0.04,
            process_noise_pos: 0.1,
            process_noise_vel: 0.05,
            measurement_noise_pos: 0.3,
            measurement_noise_vel: 0.8,
            initial_covariance_scale: 1.0,
        }
    }
}

struct ConstantVelocityModel4 {
    transition_matrix: OMatrix<f64, U4, U4>,
    transition_matrix_transpose: OMatrix<f64, U4, U4>,
    process_noise: OMatrix<f64, U4, U4>,
    config: KalmanConfigCA,
}

impl ConstantVelocityModel4 {
    fn new(config: KalmanConfigCA) -> Self {
        let dt = config.dt;
        let mut f = OMatrix::<f64, U4, U4>::identity();
        f[(0, 2)] = dt;
        f[(1, 3)] = dt;
        let q = OMatrix::<f64, U4, U4>::from_diagonal(&SVector::<f64, 4>::from_column_slice(&[
            config.process_noise_pos,
            config.process_noise_pos,
            config.process_noise_vel,
            config.process_noise_vel,
        ]));
        Self {
            transition_matrix: f,
            transition_matrix_transpose: f.transpose(),
            process_noise: q,
            config,
        }
    }

    fn set_dt(&mut self, dt: f64) {
        self.config.dt = dt;
        let mut f = OMatrix::<f64, U4, U4>::identity();
        f[(0, 2)] = dt;
        f[(1, 3)] = dt;
        self.transition_matrix = f;
        self.transition_matrix_transpose = self.transition_matrix.transpose();
        self.process_noise = OMatrix::<f64, U4, U4>::from_diagonal(&SVector::<f64, 4>::from_column_slice(&[
            self.config.process_noise_pos * dt,
            self.config.process_noise_pos * dt,
            self.config.process_noise_vel * dt,
            self.config.process_noise_vel * dt,
        ]));
    }
}

impl TransitionModelLinearNoControl<f64, U4> for ConstantVelocityModel4 {
    fn F(&self) -> &OMatrix<f64, U4, U4> { &self.transition_matrix }
    fn FT(&self) -> &OMatrix<f64, U4, U4> { &self.transition_matrix_transpose }
    fn Q(&self) -> &OMatrix<f64, U4, U4> { &self.process_noise }
}

struct FullStateObservationModel4 {
    observation_matrix: OMatrix<f64, U4, U4>,
    observation_matrix_transpose: OMatrix<f64, U4, U4>,
    measurement_noise: OMatrix<f64, U4, U4>,
}

impl FullStateObservationModel4 {
    fn new(noise_pos: f64, noise_vel: f64) -> Self {
        let h = OMatrix::<f64, U4, U4>::identity();
        let r = OMatrix::<f64, U4, U4>::from_diagonal(&SVector::<f64, 4>::from_column_slice(&[
            noise_pos, noise_pos, noise_vel, noise_vel,
        ]));
        Self { observation_matrix: h, observation_matrix_transpose: h.transpose(), measurement_noise: r }
    }
}

impl ObservationModel<f64, U4, U4> for FullStateObservationModel4 {
    fn H(&self) -> &OMatrix<f64, U4, U4> { &self.observation_matrix }
    fn HT(&self) -> &OMatrix<f64, U4, U4> { &self.observation_matrix_transpose }
    fn R(&self) -> &OMatrix<f64, U4, U4> { &self.measurement_noise }
}

/// 4D 恒速度卡尔曼滤波器 [x, y, vx, vy]
pub struct KalmanFilterCA {
    motion_model: ConstantVelocityModel4,
    observation_model: FullStateObservationModel4,
    current_estimate: StateAndCovariance<f64, U4>,
    config: KalmanConfigCA,
}

impl KalmanFilterCA {
    pub fn new(config: KalmanConfigCA) -> Result<Self, adskalman::Error> {
        let motion_model = ConstantVelocityModel4::new(config.clone());
        let observation_model = FullStateObservationModel4::new(
            config.measurement_noise_pos,
            config.measurement_noise_vel,
        );
        let initial_state = SVector::<f64, 4>::zeros();
        let initial_covariance = OMatrix::<f64, U4, U4>::identity() * config.initial_covariance_scale;
        let current_estimate = StateAndCovariance::new(initial_state, initial_covariance);
        Ok(Self { motion_model, observation_model, current_estimate, config })
    }

    pub fn init_with_state(&mut self, state: SVector<f64, 4>) {
        let cov = OMatrix::<f64, U4, U4>::identity() * self.config.initial_covariance_scale;
        self.current_estimate = StateAndCovariance::new(state, cov);
    }

    pub fn predict(&mut self, dt: f64) -> Result<(), adskalman::Error> {
        self.motion_model.set_dt(dt);
        self.current_estimate = self.motion_model.predict(&self.current_estimate);
        Ok(())
    }

    pub fn correct(&mut self, measurement: SVector<f64, 4>) -> Result<(), adskalman::Error> {
        let x = self.current_estimate.state();
        let p = self.current_estimate.covariance();
        let h = self.observation_model.H();
        let r = self.observation_model.R();
        let y = measurement - (h * x);
        let s = h * p * h.transpose() + r;
        let si = s.try_inverse().ok_or(adskalman::Error::CovarianceNotPositiveSemiDefinite)?;
        let k = p * h.transpose() * si;
        let new_x = x + &k * y;
        let i = OMatrix::<f64, U4, U4>::identity();
        let new_p = (i - &k * h) * p;
        self.current_estimate = StateAndCovariance::new(new_x, new_p);
        Ok(())
    }

    pub fn correct_position(&mut self, measurement: Vector2<f64>) -> Result<(), adskalman::Error> {
        let x = self.current_estimate.state();
        let p = self.current_estimate.covariance();
        let r_val = self.config.measurement_noise_pos;
        let innovation = Vector2::new(measurement.x - x[0], measurement.y - x[1]);
        let s = p.fixed_view::<2, 2>(0, 0) + Matrix2::identity() * r_val;
        let s_inv = s.try_inverse().ok_or(adskalman::Error::CovarianceNotPositiveSemiDefinite)?;
        let k = p.columns(0, 2) * s_inv;
        let new_x = x + &k * innovation;
        let hp = p.fixed_view::<2, 4>(0, 0);
        let new_p = p - k * hp;
        self.current_estimate = StateAndCovariance::new(new_x, new_p);
        Ok(())
    }

    pub fn get_position(&self) -> Vector3<f64> {
        let s = self.current_estimate.state();
        Vector3::new(s[0], s[1], 0.0)
    }

    pub fn get_velocity(&self) -> Vector3<f64> {
        let s = self.current_estimate.state();
        Vector3::new(s[2], s[3], 0.0)
    }

    pub fn get_full_state(&self) -> SVector<f64, 4> {
        self.current_estimate.state().clone_owned()
    }

    pub fn get_covariance(&self) -> &OMatrix<f64, U4, U4> { self.current_estimate.covariance() }

    pub fn mahalanobis_distance(&self, measurement: Vector2<f64>) -> f64 {
        let s = self.current_estimate.state();
        let innovation = Vector2::new(measurement.x - s[0], measurement.y - s[1]);
        let cov = self.current_estimate.covariance();
        let h = self.observation_model.H();
        let r = self.observation_model.R();
        let s_full = h * cov * h.transpose() + r;
        let s_pos = s_full.fixed_view::<2, 2>(0, 0);
        s_pos.try_inverse()
            .map(|s_inv| (&innovation).dot(&(&s_inv * &innovation)).sqrt())
            .unwrap_or(f64::MAX)
    }

    pub fn correct_with_gating(&mut self, measurement: SVector<f64, 4>, gate_threshold: f64) -> Result<(), adskalman::Error> {
        let x = self.current_estimate.state();
        let p = self.current_estimate.covariance();

        let innovation_pos = Vector2::new(measurement[0] - x[0], measurement[1] - x[1]);
        let s_pos = p.fixed_view::<2, 2>(0, 0) + Matrix2::identity() * self.config.measurement_noise_pos;
        let s_inv = match s_pos.try_inverse() {
            Some(inv) => inv,
            None => return Err(adskalman::Error::CovarianceNotPositiveSemiDefinite),
        };
        let mahal = (innovation_pos.dot(&(&s_inv * &innovation_pos))).sqrt();

        if mahal > gate_threshold {
            let r_val = self.config.measurement_noise_pos;
            let innovation = Vector2::new(measurement[0] - x[0], measurement[1] - x[1]);
            let s = p.fixed_view::<2, 2>(0, 0) + Matrix2::identity() * r_val;
            let s_inv = s.try_inverse().ok_or(adskalman::Error::CovarianceNotPositiveSemiDefinite)?;
            let k = p.columns(0, 2) * s_inv;
            let new_x = x + &k * innovation;
            let hp = p.fixed_view::<2, 4>(0, 0);
            let new_p = p - k * hp;
            self.current_estimate = StateAndCovariance::new(new_x, new_p);
            return Ok(());
        }

        let h = self.observation_model.H();
        let r = self.observation_model.R();
        let y = measurement - (h * x);
        let s = h * p * h.transpose() + r;
        let si = s.try_inverse().ok_or(adskalman::Error::CovarianceNotPositiveSemiDefinite)?;
        let k = p * h.transpose() * si;
        let new_x = x + &k * y;
        let i = OMatrix::<f64, U4, U4>::identity();
        let new_p = (i - &k * h) * p;
        self.current_estimate = StateAndCovariance::new(new_x, new_p);
        Ok(())
    }

    pub fn adjust_noise_for_distance(&mut self, distance: f64) {
        let scale = 1.0 + distance / 10.0;
        let noise_pos = self.config.measurement_noise_pos * scale;
        let noise_vel = self.config.measurement_noise_vel * scale.min(3.0);
        self.observation_model = FullStateObservationModel4::new(noise_pos, noise_vel);
    }

    pub fn adjust_noise_for_confidence(&mut self, distance: f64, confidence: f32) {
        let dist_scale = 1.0 + distance / 10.0;
        let conf_scale = 1.0 + (1.0 - confidence as f64) * 3.0;
        let scale = dist_scale * conf_scale;
        let noise_pos = self.config.measurement_noise_pos * scale;
        let noise_vel = self.config.measurement_noise_vel * scale.min(3.0);
        self.observation_model = FullStateObservationModel4::new(noise_pos, noise_vel);
    }

    pub fn clamp_state(&mut self, max_speed: f64, _max_accel: f64, _min_size: f64, _max_size: f64) {
        let mut state = self.current_estimate.state().clone_owned();
        state[2] = state[2].clamp(-max_speed, max_speed);
        state[3] = state[3].clamp(-max_speed, max_speed);
        self.current_estimate = StateAndCovariance::new(state, self.current_estimate.covariance().clone());
    }
}
