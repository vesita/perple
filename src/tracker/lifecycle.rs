use crate::tracker::object::{TargetClass, TrackedObject};

/// 分类状态机
///
/// 用 `Transitioner<bool>` 替代手动计数器管理：
///
///   Static
///     └─ `static_leaver.feed(!in_static_cluster)` → 10 帧后 → Floating
///
///   Floating
///     ├─ `floating_settler.feed(in_static_cluster)` → N 帧后 → Static
///     └─ `voting_promoter.feed(voting_active && speed ≥ 0.2)` → N 帧后 → Moving
///
///   Moving ←──→ Movable
///     └─ `class_transitioner.feed(speed > moving_speed_threshold)`
///        ON  → Moving，OFF → Movable
///
///   一旦 confirmed_moving=true，永不回到 Static/Floating
pub(crate) fn apply_state_machine(
    obj: &mut TrackedObject,
    in_static_cluster: bool,
    voting_active: bool,
    speed: f32,
    moving_speed_threshold: f32,
) {
    match obj.classification {
        // ════════════════════════════════════════════════════════════════
        //  Static → Floating
        // ════════════════════════════════════════════════════════════════
        TargetClass::Static => {
            if obj.static_leaver.feed(&(!in_static_cluster)) {
                obj.classification = TargetClass::Floating;
                obj.floating_settler.reset();
            }
        }

        // ════════════════════════════════════════════════════════════════
        //  Floating → Static / Moving
        // ════════════════════════════════════════════════════════════════
        TargetClass::Floating => {
            if obj.floating_settler.feed(&in_static_cluster) {
                obj.classification = TargetClass::Static;
            }
            if obj.voting_promoter.feed(&(voting_active && speed >= 0.2)) {
                obj.classification = TargetClass::Moving;
                obj.confirmed_moving = true;
            }
        }

        // ════════════════════════════════════════════════════════════════
        //  Moving ↔ Movable
        // ════════════════════════════════════════════════════════════════
        TargetClass::Moving | TargetClass::Movable => {
            obj.confirmed_moving = true;

            let want_move = speed > moving_speed_threshold;
            // 检测双向翻转（ON→OFF 或 OFF→ON），feed() 返回翻转后的状态
            let prev = obj.class_transitioner.state();
            let next = obj.class_transitioner.feed(&want_move);
            if prev != next {
                obj.classification = if next {
                    TargetClass::Moving
                } else {
                    TargetClass::Movable
                };
            }
        }
    }
}

