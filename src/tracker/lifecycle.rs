use crate::tracker::object::{TargetClass, TrackedObject};

/// 点云投票分析（信号产生，不修改分类）
///
/// 对每个轨迹：当前帧点云 vs skip_frames 帧前点云
/// 分类状态机
///
/// 变迁规则：
///   Static ←→ Floating（滞后：上浮无门槛，沉淀需 30 帧）
///   Floating ──> Moving（点云投票 + 速度 + 连续帧）
///   Moving ←──→ Movable（同层往返，速度决定）
///   一旦 confirmed_moving=true，永不回到 Static/Floating
pub(crate) fn apply_state_machine(
    obj: &mut TrackedObject,
    in_static_cluster: bool,
    voting_active: bool,
    speed: f32,
    moving_speed_threshold: f32,
    floating_to_static_frames: usize,
    voting_consistency_frames: usize,
    class_cooldown_frames: u32,
) {
    match obj.classification {
        TargetClass::Static => {
            if !in_static_cluster {
                // Static → Floating：带迟滞，连续 N 帧偏离静态簇后才转换
                obj.static_miss_count += 1;
                if obj.static_miss_count >= 10 {
                    obj.classification = TargetClass::Floating;
                    obj.floating_static_count = 0;
                    obj.static_miss_count = 0;
                }
            } else {
                obj.static_miss_count = 0;
            }
        }
        TargetClass::Floating => {
            if in_static_cluster {
                // 持续在静态簇中 → 累积计数
                obj.floating_static_count += 1;
                if obj.floating_static_count >= floating_to_static_frames as u32 {
                    obj.classification = TargetClass::Static;
                    obj.floating_static_count = 0;
                }
            } else {
                obj.floating_static_count = 0;
            }

            // 点云投票 → Floating → Moving
            if voting_active && speed >= 0.2 {
                obj.voting_streak += 1;
                if obj.voting_streak >= voting_consistency_frames as u32 {
                    obj.classification = TargetClass::Moving;
                    obj.confirmed_moving = true;
                    obj.voting_streak = 0;
                }
            } else {
                obj.voting_streak = 0;
            }
        }
        TargetClass::Moving | TargetClass::Movable => {
            obj.confirmed_moving = true;
            // Moving ↔ Movable：带冷却的迟滞转换
            let want_move = speed > moving_speed_threshold;
            let is_move = obj.classification == TargetClass::Moving;

            if want_move == is_move {
                // 已在目标状态，重置冷却
                obj.class_cooldown = 0;
            } else {
                // 需要切换状态，累积冷却
                obj.class_cooldown += 1;
                if obj.class_cooldown >= class_cooldown_frames {
                    obj.classification = if want_move { TargetClass::Moving } else { TargetClass::Movable };
                    obj.class_cooldown = 0;
                }
            }
        }
    }
}

