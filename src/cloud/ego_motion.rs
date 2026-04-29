use std::time::Instant;

use crate::swapl::global_swapl;
use crate::utils::stream::{Eap, Stream};

/// 基于帧间地面平面方程追踪的自车速度估计器
///
/// 原理：小车前进时，LiDAR 帧中地面平面位置发生偏移。
/// 追踪平面方程 [a,b,c,d] 的变化 → 自车速度。
pub struct EgoMotion {
    ground_plane_stream: Eap<Stream<[f32; 4]>>,
    prev_plane: Option<[f32; 4]>,
    prev_time: Option<Instant>,
    velocity: [f32; 3],
    velocity_smooth: [f32; 3],
}

impl EgoMotion {
    pub fn new() -> Self {
        let swapl = global_swapl();
        Self {
            ground_plane_stream: swapl.ground_plane.clone(),
            prev_plane: None,
            prev_time: None,
            velocity: [0.0; 3],
            velocity_smooth: [0.0; 3],
        }
    }

    /// 每帧调用：读取当前地面平面，更新自车速度估计
    /// 返回平滑后的自车速度 [vx, vy, vz]（LiDAR 帧，单位 m/s）
    pub fn update(&mut self) -> [f32; 3] {
        let now = Instant::now();

        let current = {
            let stream = self.ground_plane_stream.blocking_lock();
            stream.peek_latest()
        };

        if let Some(plane) = current {
            if let Some(prev) = self.prev_plane {
                let dt = now
                    .duration_since(self.prev_time.unwrap())
                    .as_secs_f32();
                if dt > 0.001 && dt < 1.0 {
                    // 取平面上离原点最近点：p = -d * [a, b, c]
                    let p_curr = [
                        -plane[3] * plane[0],
                        -plane[3] * plane[1],
                        -plane[3] * plane[2],
                    ];
                    let p_prev = [
                        -prev[3] * prev[0],
                        -prev[3] * prev[1],
                        -prev[3] * prev[2],
                    ];

                    // 地面表观位移（表观 = 自车运动的相反方向）
                    let dx = p_curr[0] - p_prev[0];
                    let dy = p_curr[1] - p_prev[1];
                    let dz = p_curr[2] - p_prev[2];

                    // 自车速度 = -表观位移 / dt
                    self.velocity = [-dx / dt, -dy / dt, -dz / dt];
                }
            }
            self.prev_plane = Some(plane);
            self.prev_time = Some(now);
        }

        // 指数平滑
        let alpha = 0.3;
        self.velocity_smooth = [
            self.velocity_smooth[0] * (1.0 - alpha) + self.velocity[0] * alpha,
            self.velocity_smooth[1] * (1.0 - alpha) + self.velocity[1] * alpha,
            self.velocity_smooth[2] * (1.0 - alpha) + self.velocity[2] * alpha,
        ];

        self.velocity_smooth
    }

    /// 原始（未平滑）速度
    pub fn get_raw_velocity(&self) -> [f32; 3] {
        self.velocity
    }

    /// 平滑后的速度
    pub fn get_velocity(&self) -> [f32; 3] {
        self.velocity_smooth
    }

    /// 自车速率
    pub fn get_speed(&self) -> f32 {
        let v = self.velocity_smooth;
        (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
    }

    pub fn reset(&mut self) {
        self.prev_plane = None;
        self.prev_time = None;
        self.velocity = [0.0; 3];
        self.velocity_smooth = [0.0; 3];
    }
}

impl Default for EgoMotion {
    fn default() -> Self {
        Self::new()
    }
}
