//! 外参偏差监测工具
//!
//! 利用 fuse 的 2D→3D 匹配结果，统计投影残差，通过子 Kalman 滤波器
//! 估计外参微小偏移 [δrx, δry, δrz, δtx, δty, δtz]。
//! 输出：偏差估计 + 投影残差统计 → CSV 文件。

use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::Path;

use nalgebra::{Matrix2, Matrix2x6, Matrix3, Matrix4, Matrix6, Vector2, Vector4, Vector6};

use crate::cloud::CldBud;
use crate::color::output::ClrBud;
use crate::config::fixif;
use crate::swapl::global_swapl;
use crate::utils::boxes::Box2D;

/// 单帧匹配记录的投影残差
#[allow(dead_code)]
struct MatchResidual {
    class_name: String,
    iou: f32,
    /// 3D 框中心投影到像素的坐标
    proj_u: f32,
    proj_v: f32,
    /// 2D 检测框中心
    det_u: f32,
    det_v: f32,
    /// 残差（像素）
    du: f32,
    dv: f32,
    /// 3D 框中心在相机帧的深度 Z
    cam_z: f32,
}

/// 外参偏差监测器
pub struct ExtrinsicMonitor {
    intrinsic: Matrix3<f32>,
    cam_from_lidar: Matrix4<f32>,
    frame_count: u32,
    csv_writer: Option<BufWriter<File>>,
    /// 当前偏差估计 [δrx, δry, δrz, δtx, δty, δtz]
    /// 单位：旋转弧度，平移米
    state: Vector6<f64>,
    /// 状态协方差
    cov: Matrix6<f64>,
    /// 过程噪声系数
    process_noise_scale: f64,
    /// 测量噪声系数（像素）
    measurement_noise_scale: f64,
    /// 各帧残差历史（用于可视化/分析）
    residuals_history: Vec<Vec<MatchResidual>>,
}

impl ExtrinsicMonitor {
    pub fn new() -> Self {
        let config = fixif();
        let cam_from_lidar = Matrix4::from(config.camera.extrinsic);
        Self {
            intrinsic: Matrix3::from(config.camera.intrinsic),
            cam_from_lidar,
            frame_count: 0,
            csv_writer: None,
            state: Vector6::zeros(),
            cov: Matrix6::identity() * 1.0,  // 初始协方差：1 rad / 1 m
            process_noise_scale: 1e-6,
            measurement_noise_scale: 5.0,     // 5 像素噪声
            residuals_history: Vec::new(),
        }
    }

    /// 设置 CSV 输出路径
    pub fn set_csv_path<P: AsRef<Path>>(&mut self, path: P) -> std::io::Result<()> {
        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(path.as_ref())?;
        let mut writer = BufWriter::new(file);
        // CSV 表头
        writeln!(writer, "frame,n_matches,delta_rx,delta_ry,delta_rz,delta_tx,delta_ty,delta_tz,avg_du,avg_dv,std_du,std_dv,max_du,max_dv")?;
        self.csv_writer = Some(writer);
        Ok(())
    }

    /// 每帧调用：读取检测结果，计算匹配残差，更新 Kalman 估计
    pub fn update(&mut self) {
        let swapl = global_swapl();

        let clr_buds: Vec<ClrBud> = swapl.clr_objs.producer().lock().unwrap().clone();
        if clr_buds.is_empty() {
            return;
        }
        let cld_buds: Vec<CldBud> = match swapl.cld_objs.lock().unwrap().get_at(0) {
            Some(buds) => buds,
            None => return,
        };

        if clr_buds.is_empty() || cld_buds.is_empty() {
            return;
        }

        self.frame_count += 1;
        let fx = self.intrinsic[(0, 0)];
        let fy = self.intrinsic[(1, 1)];
        let cx = self.intrinsic[(0, 2)];
        let cy = self.intrinsic[(1, 2)];

        // ── 对每个 3D 检测，投影并找最佳 2D 匹配 ──
        let mut frame_residuals: Vec<MatchResidual> = Vec::new();

        for cld in &cld_buds {
            let verts = cld.the_box.vertices();
            let center = cld.the_box.center();

            // 计算 3D 框中心在相机帧的位置
            let cam_center = self.cam_from_lidar * Vector4::new(center.x, center.y, center.z, 1.0);
            if cam_center.z <= 0.0 {
                continue;
            }
            let center_u = fx * cam_center.x / cam_center.z + cx;
            let center_v = fy * cam_center.y / cam_center.z + cy;

            // 投影 3D 框 8 顶点到 2D 求包围盒
            let (mut l, mut t, mut r, mut b) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
            for v in &verts {
                let cam = self.cam_from_lidar * Vector4::new(v.x, v.y, v.z, 1.0);
                if cam.z <= 0.0 {
                    continue;
                }
                let u = fx * cam.x / cam.z + cx;
                let v_ = fy * cam.y / cam.z + cy;
                l = l.min(u);
                t = t.min(v_);
                r = r.max(u);
                b = b.max(v_);
            }
            if l == f32::MAX {
                continue;
            }
            let proj_box = Box2D::new(l, t, r, b);
            if !proj_box.is_valid() {
                continue;
            }

            // 找最佳匹配
            let mut best_iou = 0.2;
            let mut matched_clr: Option<&ClrBud> = None;
            for clr in &clr_buds {
                let iou = proj_box.iou(&clr.the_box);
                if iou > best_iou {
                    best_iou = iou;
                    matched_clr = Some(clr);
                }
            }

            if let Some(clr) = matched_clr {
                let det_u = (clr.the_box.x1 + clr.the_box.x2) / 2.0;
                let det_v = (clr.the_box.y1 + clr.the_box.y2) / 2.0;
                let du = center_u - det_u;
                let dv = center_v - det_v;

                frame_residuals.push(MatchResidual {
                    class_name: clr.class_name.clone(),
                    iou: best_iou,
                    proj_u: center_u,
                    proj_v: center_v,
                    det_u,
                    det_v,
                    du,
                    dv,
                    cam_z: cam_center.z,
                });
            }
        }

        if frame_residuals.is_empty() {
            return;
        }

        // ── 计算平均残差 ──
        let n = frame_residuals.len();
        let avg_du: f32 = frame_residuals.iter().map(|r| r.du).sum::<f32>() / n as f32;
        let avg_dv: f32 = frame_residuals.iter().map(|r| r.dv).sum::<f32>() / n as f32;

        // 标准差
        let var_du = frame_residuals.iter().map(|r| (r.du - avg_du).powi(2)).sum::<f32>() / n as f32;
        let var_dv = frame_residuals.iter().map(|r| (r.dv - avg_dv).powi(2)).sum::<f32>() / n as f32;
        let std_du = var_du.sqrt();
        let std_dv = var_dv.sqrt();
        let max_du = frame_residuals.iter().map(|r| r.du).fold(f32::MIN, f32::max);
        let max_dv = frame_residuals.iter().map(|r| r.dv).fold(f32::MIN, f32::max);

        // ── 计算平均 3D 点在相机帧的坐标（用于 Jacobian） ──
        let avg_cam_z: f32 = frame_residuals.iter().map(|r| r.cam_z).sum::<f32>() / n as f32;
        let avg_proj_u: f32 = frame_residuals.iter().map(|r| r.proj_u).sum::<f32>() / n as f32;
        let avg_proj_v: f32 = frame_residuals.iter().map(|r| r.proj_v).sum::<f32>() / n as f32;
        let avg_cam_x = (avg_proj_u - cx) * avg_cam_z / fx;
        let avg_cam_y = (avg_proj_v - cy) * avg_cam_z / fy;

        // ── Kalman 更新 ──
        // H = ∂[du, dv] / ∂[δrx, δry, δrz, δtx, δty, δtz]
        self.kalman_update(
            avg_du as f64, avg_dv as f64,
            avg_cam_x as f64, avg_cam_y as f64, avg_cam_z as f64,
            fx as f64, fy as f64,
        );

        // ── 写入 CSV ──
        if let Some(ref mut writer) = self.csv_writer {
            let _ = writeln!(
                writer,
                "{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
                self.frame_count,
                n,
                self.state[0], self.state[1], self.state[2],
                self.state[3], self.state[4], self.state[5],
                avg_du, avg_dv,
                std_du, std_dv,
                max_du, max_dv,
            );
        }

        self.residuals_history.push(frame_residuals);
        // 保留最近 1000 帧
        if self.residuals_history.len() > 1000 {
            self.residuals_history.remove(0);
        }
    }

    /// Kalman predict + correct for 6-DOF bias
    fn kalman_update(
        &mut self,
        du: f64, dv: f64,
        cam_x: f64, cam_y: f64, cam_z: f64,
        fx: f64, fy: f64,
    ) {
        // ── Predict (constant bias model, small random walk) ──
        // x_pred = x
        // P_pred = P + Q
        let q = Matrix6::identity() * self.process_noise_scale;
        let p_pred = self.cov + q;

        // ── 计算 Jacobian H (2×6) ──
        // 公式推导见模块文档：∂[du,dv]/∂[δrx,δry,δrz,δtx,δty,δtz]
        if cam_z.abs() < 0.001 {
            return;
        }
        let z = cam_z;
        let x = cam_x;
        let y = cam_y;
        let z2 = z * z;

        // H rows: [du/dδrx, du/dδry, du/dδrz, du/dδtx, du/dδty, du/dδtz]
        //         [dv/dδrx, dv/dδry, dv/dδrz, dv/dδtx, dv/dδty, dv/dδtz]
        let h = Matrix2x6::new(
            -fx * x * y / z2,   fx * (1.0 + x * x / z2),   -fx * y / z,   fx / z,   0.0,   -fx * x / z2,
            -fy * (1.0 + y * y / z2),   fy * x * y / z2,    fy * x / z,   0.0,   fy / z,   -fy * y / z2,
        );

        // ── Innovation ──
        // z_meas = [du, dv] (the actual pixel residual)
        // H * x_pred is the predicted residual given current bias
        // For small bias, H * x_pred ≈ 0 (no bias → no systematic residual)
        let innovation = Vector2::new(du, dv) - &h * &self.state;

        // ── Innovation covariance ──
        let r = Matrix2::identity() * self.measurement_noise_scale;
        let s = &h * &p_pred * h.transpose() + r;

        // ── Kalman gain ──
        let s_inv = match s.try_inverse() {
            Some(inv) => inv,
            None => return,
        };
        let k = &p_pred * h.transpose() * s_inv;

        // ── Correct ──
        self.state = &self.state + &k * innovation;
        let i6 = Matrix6::identity();
        self.cov = (&i6 - &k * &h) * &p_pred;
    }

    /// 当前偏差估计 [δrx, δry, δrz, δtx, δty, δtz]
    pub fn get_bias(&self) -> [f64; 6] {
        [
            self.state[0], self.state[1], self.state[2],
            self.state[3], self.state[4], self.state[5],
        ]
    }

    /// 当前帧的平均像素残差
    pub fn get_latest_residual_stats(&self) -> Option<(f32, f32, usize)> {
        self.residuals_history.last().map(|r| {
            let n = r.len();
            let avg_du = r.iter().map(|m| m.du).sum::<f32>() / n as f32;
            let avg_dv = r.iter().map(|m| m.dv).sum::<f32>() / n as f32;
            (avg_du, avg_dv, n)
        })
    }

    pub fn reset(&mut self) {
        self.state = Vector6::zeros();
        self.cov = Matrix6::identity() * 1.0;
        self.frame_count = 0;
        self.residuals_history.clear();
    }
}

impl Default for ExtrinsicMonitor {
    fn default() -> Self {
        Self::new()
    }
}
