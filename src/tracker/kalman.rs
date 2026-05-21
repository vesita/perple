//! 卡尔曼滤波模块
//!
//! 使用 4D CV 模型 [`KalmanFilterCA`]：`[x, y, vx, vy]`
//!
//! 测量值（LV-DOT 风格）：全状态观测 H=I
//! - 位置 (x,y) 直接观测量测值
//! - 速度 (vx,vy) 通过 k 帧位置差计算
//!
//! 关键设计：
//! - predict() 和 correct() 分离，避免重复预测
//! - 动态 dt：基于帧间隔时间戳实时计算
//! - correct_with_gating() 新息门控：马氏距离超过阈值时降级到位置-only修正
//!
//! 调参建议（通过 eval_ablation --tracker-toml）：
//!
//! | 参数 | 默认值 | 调大效果 | 调小效果 |
//! |------|--------|----------|----------|
//! | kf_process_noise_vel | 0.05 | 速度跟踪更灵活 | 速度更平滑 |
//! | kf_measurement_noise_vel | 0.8 | 速度更平滑(信任预测) | 速度响应更快 |
//! | kf_measurement_noise_pos | 0.3 | 位置更平滑 | 位置响应更快 |
//! | kf_gate_threshold | 3.5 | 更多测量被接受 | 更多被门控拒绝 |

pub mod ca;

pub use ca::{KalmanConfigCA, KalmanFilterCA};
