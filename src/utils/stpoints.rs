//! STPoints 标定文件加载模块
//!
//! 加载 STPoints（OpenCV 标定工具）输出的 JSON 标定文件，
//! 转换为 perple 的 CameraConfig 格式。
//!
//! STPoints JSON 格式（OpenCV 约定）：
//! - `intrinsic`: 9 floats, 3×3 行主序 → `[fx, 0, cx, 0, fy, cy, 0, 0, 1]`
//! - `extrinsic`: 16 floats, 4×4 行主序 → `[R|t; 0 0 0 1]`
//! - `dist_coeffs`: N floats (通常 5: k1, k2, p1, p2, k3)
//!
//! 使用的 OpenCV 函数：
//! - `cv2.calibrateCamera` → intrinsic + dist_coeffs
//! - `cv2.solvePnP` / `cv2.solvePnPRansac` → extrinsic
//! - `cv2.undistort` → 使用 intrinsic + dist_coeffs 去畸变

use std::fs;
use std::path::Path;

use serde::Deserialize;

use crate::config::CameraConfig;

/// STPoints JSON 标定文件结构
#[derive(Deserialize, Debug)]
struct StPointsCalib {
    /// 3×3 内参矩阵，9 个 float（行主序）
    intrinsic: Vec<f32>,
    /// 4×4 外参矩阵，16 个 float（行主序）
    extrinsic: Vec<f32>,
    /// 畸变系数 [k1, k2, p1, p2, k3]，可选
    dist_coeffs: Option<Vec<f32>>,
}

/// 从 STPoints JSON 文件加载标定参数
///
/// # 参数
/// * `path` - JSON 文件路径
///
/// # 返回值
/// 返回 `CameraConfig`，包含嵌套矩阵格式的内参、外参和畸变系数
///
/// # 错误
/// - 文件不存在或无法读取
/// - JSON 格式不正确
/// - 数组长度不符合预期（intrinsic=9, extrinsic=16）
pub fn load_stpoints(path: impl AsRef<Path>) -> Result<CameraConfig, String> {
    let content = fs::read_to_string(&path)
        .map_err(|e| format!("读取标定文件失败: {}", e))?;
    parse_stpoints(&content)
}

/// 从 JSON 字符串解析标定参数
pub fn parse_stpoints(json: &str) -> Result<CameraConfig, String> {
    let calib: StPointsCalib = serde_json::from_str(json)
        .map_err(|e| format!("解析 JSON 失败: {}", e))?;

    // 校验 intrinsic: 9 floats → 3×3
    if calib.intrinsic.len() != 9 {
        return Err(format!(
            "intrinsic 长度错误: 期望 9，实际 {}", calib.intrinsic.len()
        ));
    }
    let mut intrinsic = [[0.0f32; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            intrinsic[r][c] = calib.intrinsic[r * 3 + c];
        }
    }

    // 校验 extrinsic: 16 floats → 4×4
    if calib.extrinsic.len() != 16 {
        return Err(format!(
            "extrinsic 长度错误: 期望 16，实际 {}", calib.extrinsic.len()
        ));
    }
    let mut extrinsic = [[0.0f32; 4]; 4];
    for r in 0..4 {
        for c in 0..4 {
            extrinsic[r][c] = calib.extrinsic[r * 4 + c];
        }
    }

    // 畸变系数：默认 5 个 (k1, k2, p1, p2, k3)
    let dist_coeffs = match calib.dist_coeffs {
        Some(ref coeffs) if coeffs.len() >= 5 => {
            Some([coeffs[0], coeffs[1], coeffs[2], coeffs[3], coeffs[4]])
        }
        Some(ref coeffs) => {
            // 不足 5 个时补零
            let mut arr = [0.0f32; 5];
            for (i, &v) in coeffs.iter().enumerate().take(5) {
                arr[i] = v;
            }
            Some(arr)
        }
        None => None,
    };

    Ok(CameraConfig {
        intrinsic,
        extrinsic,
        dist_coeffs,
    })
}

/// 将 CameraConfig 序列化为 TOML 格式的 camera 段
///
/// 用于生成可直接粘贴到 `config/default.toml` 的文本
pub fn to_toml_string(config: &CameraConfig) -> String {
    let mut out = String::new();

    // intrinsic
    out.push_str("[camera]\n");
    out.push_str("intrinsic = [\n");
    for r in 0..3 {
        out.push_str(&format!("  [{}, {}, {}]", config.intrinsic[r][0], config.intrinsic[r][1], config.intrinsic[r][2]));
        if r < 2 { out.push(','); }
        out.push('\n');
    }
    out.push_str("]\n\n");

    // extrinsic
    out.push_str("extrinsic = [\n");
    for r in 0..4 {
        out.push_str(&format!(
            "  [{}, {}, {}, {}]",
            config.extrinsic[r][0], config.extrinsic[r][1],
            config.extrinsic[r][2], config.extrinsic[r][3]
        ));
        if r < 3 { out.push(','); }
        out.push('\n');
    }
    out.push_str("]\n");

    // dist_coeffs
    if let Some(ref d) = config.dist_coeffs {
        out.push_str(&format!(
            "\ndist_coeffs = [{}, {}, {}, {}, {}]\n",
            d[0], d[1], d[2], d[3], d[4]
        ));
    }

    out
}

/// OpenCV 去畸变：使用内参和畸变系数校正像素坐标
///
/// 对单个 2D 点进行去畸变（Brown-Conrady 模型）。
/// 适用于在投影/反投影前校正 YOLO 检测框中心等关键点。
///
/// # 参数
/// * `point` - 畸变图像上的 (u, v) 像素坐标
/// * `intrinsic` - 3×3 内参矩阵
/// * `dist` - 畸变系数 [k1, k2, p1, p2, k3]
///
/// # 返回值
/// 去畸变后的 (u, v) 像素坐标
pub fn undistort_point(point: (f32, f32), intrinsic: &[[f32; 3]; 3], dist: &[f32; 5]) -> (f32, f32) {
    let fx = intrinsic[0][0];
    let fy = intrinsic[1][1];
    let cx = intrinsic[0][2];
    let cy = intrinsic[1][2];

    // 像素 → 归一化坐标
    let mut x = (point.0 - cx) / fx;
    let mut y = (point.1 - cy) / fy;

    let k1 = dist[0]; let k2 = dist[1];
    let p1 = dist[2]; let p2 = dist[3];
    let k3 = dist[4];

    // 迭代去畸变（Brown-Conrady）
    for _ in 0..5 {
        let r2 = x * x + y * y;
        let r4 = r2 * r2;
        let r6 = r4 * r2;
        let radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;
        let dx = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
        let dy = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;

        // 理想归一化坐标
        let x_ideal = (point.0 - cx) / fx / radial - dx / radial;
        let y_ideal = (point.1 - cy) / fy / radial - dy / radial;

        x = x_ideal;
        y = y_ideal;
    }

    // 归一化坐标 → 像素
    (x * fx + cx, y * fy + cy)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_stpoints() {
        let json = r#"{
            "intrinsic": [523.08, 0, 327.82, 0, 523.60, 209.39, 0, 0, 1],
            "extrinsic": [
                0.1393, 0.9817, 0.1294, -0.4347,
                0.0749, -0.1408, 0.9872, -0.6424,
                0.9874, -0.1279, -0.0932, 0.1997,
                0, 0, 0, 1
            ],
            "dist_coeffs": [-0.498, 0.306, -0.0056, -0.0019, -0.100]
        }"#;

        let config = parse_stpoints(json).unwrap();
        assert!((config.intrinsic[0][0] - 523.08).abs() < 0.01);
        assert!((config.intrinsic[1][1] - 523.60).abs() < 0.01);
        assert!((config.intrinsic[0][2] - 327.82).abs() < 0.01);
        assert!(config.dist_coeffs.is_some());
        assert!((config.dist_coeffs.unwrap()[0] - (-0.498)).abs() < 0.001);
    }

    #[test]
    fn test_parse_no_distortion() {
        let json = r#"{
            "intrinsic": [500, 0, 320, 0, 500, 240, 0, 0, 1],
            "extrinsic": [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]
        }"#;

        let config = parse_stpoints(json).unwrap();
        assert!(config.dist_coeffs.is_none());
    }

    #[test]
    fn test_undistort_center() {
        // 图像中心点去畸变后应该不变
        let intrinsic = [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]];
        let dist = [0.0, 0.0, 0.0, 0.0, 0.0];
        let (u, v) = undistort_point((320.0, 240.0), &intrinsic, &dist);
        assert!((u - 320.0).abs() < 0.01);
        assert!((v - 240.0).abs() < 0.01);
    }
}
