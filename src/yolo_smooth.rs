//! YOLO 检测结果标签平滑
//!
//! 在 Camera 写入检测结果后、Fuse 融合前介入。
//! 利用帧间位置关联 + 标签动量，对 "person" 标签做时间域平滑，
//! 消除 YOLO 间歇性漏检导致的标签闪烁。
//!
//! 使用方式（在 swap 后、fuse.act() 前调用）：
//! ```ignore
//! let mut smoother = YoloSmoother::new();
//! // 每帧：
//! swapl.clr_objs.swap();
//! smoother.smooth(&mut *swapl.clr_objs.consumer().lock().unwrap());
//! fuse.act().await;
//! ```

use crate::color::ClrBud;

/// 帧间检测关联 + 标签动量
pub struct YoloSmoother {
    /// 上一帧的检测（按 2D 中心位置 + 动量）
    prev: Vec<PriorDetection>,
    /// 每帧 YOLO 标 "person" 时动量增量
    person_boost: f32,
    /// 每帧 YOLO 未标 "person" 时动量衰减量
    decay_amount: f32,
    /// 动量超过此阈值 → 输出 "person"（即使 YOLO 当前未标）
    momentum_threshold: f32,
    /// 帧间关联的 2D 中心距离阈值（像素）
    assoc_pixels: f32,
}

struct PriorDetection {
    u: f32,
    v: f32,
    momentum: f32,
}

impl YoloSmoother {
    pub fn new() -> Self {
        Self {
            prev: Vec::new(),
            person_boost: 0.25,
            decay_amount: 0.12,
            momentum_threshold: 0.15,
            assoc_pixels: 60.0,
        }
    }

    /// 自定义参数
    pub fn with_params(person_boost: f32, decay_amount: f32, momentum_threshold: f32, assoc_pixels: f32) -> Self {
        Self {
            prev: Vec::new(),
            person_boost,
            decay_amount,
            momentum_threshold,
            assoc_pixels,
        }
    }

    /// 对当前帧的 YOLO 检测结果进行标签平滑（原地修改 class_name）
    pub fn smooth(&mut self, detections: &mut Vec<ClrBud>) {
        let assoc_sq = self.assoc_pixels * self.assoc_pixels;
        let mut used_prev = vec![false; self.prev.len()];
        let mut next_prev: Vec<PriorDetection> = Vec::with_capacity(detections.len());

        for det in detections.iter_mut() {
            let cu = (det.the_box.x1 + det.the_box.x2) * 0.5;
            let cv = (det.the_box.y1 + det.the_box.y2) * 0.5;
            let is_person = det.class_name == "person";

            // 贪心关联：找最近的上一帧检测
            let mut best_idx = None;
            let mut best_dist = assoc_sq;
            for (pi, pd) in self.prev.iter().enumerate() {
                if used_prev[pi] { continue; }
                let dx = pd.u - cu;
                let dy = pd.v - cv;
                let d = dx * dx + dy * dy;
                if d < best_dist {
                    best_dist = d;
                    best_idx = Some(pi);
                }
            }

            let momentum = if let Some(pi) = best_idx {
                used_prev[pi] = true;
                let pd = &self.prev[pi];
                if is_person {
                    (pd.momentum + self.person_boost).min(1.0)
                } else {
                    (pd.momentum - self.decay_amount).max(0.0)
                }
            } else {
                if is_person { self.person_boost } else { 0.0 }
            };

            // 应用平滑：动量超过阈值且 YOLO 当前未标 person → 补标
            if !is_person && momentum > self.momentum_threshold {
                det.class_name = "person".to_string();
            }

            next_prev.push(PriorDetection { u: cu, v: cv, momentum });
        }

        self.prev = next_prev;
    }

    /// 重置内部状态（切换场景时调用）
    pub fn reset(&mut self) {
        self.prev.clear();
    }
}
