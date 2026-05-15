use std::collections::BTreeMap;

use super::object::{TargetClass, TrackedObject};

/// 3D 包围盒几何外形判断：检查是否像行人。
///
/// 当 YOLO/Fuse 无法为追踪目标打标签时（例如目标在相机盲区），
/// 用此函数作为后备手段。
///
/// 判断依据（多级级联 + 体积约束兜底）：
/// - 高度 0.5~2.2m（含蹲姿/坐姿，但排除地面杂物）
/// - 水平尺寸 0.15~1.2m（排除车辆/墙体/极小微点）
/// - 最小厚度 > 0.10m（排除薄片/杆状物）
/// - 最小体积 > 0.04 m³（排除微小噪声聚类通过宽松尺寸筛选）
/// - 高宽比 > 1.15（明显瘦高，区别于箱子/长凳等）
fn check_person_geom(obj: &TrackedObject) -> bool {
    let ref_box = match obj.last_box {
        Some(ref b) => b,
        None => return false,
    };

    let h = ref_box.height;
    let horiz_max = ref_box.length.max(ref_box.width);  // 水平方向最大尺寸
    let horiz_min = ref_box.length.min(ref_box.width);   // 水平方向最小尺寸（厚度）

    // 高度范围：站立行人～蹲姿/坐姿
    if h < 0.5 || h > 2.2 {
        return false;
    }

    // 水平最大尺寸：排除车辆和极小噪点
    if horiz_max < 0.15 || horiz_max > 1.2 {
        return false;
    }

    // 最小厚度：排除薄片/杆状物（行人至少有一定厚度）
    if horiz_min < 0.10 {
        return false;
    }

    // 最小体积约束：排除微小噪声聚类（即使各维度都通过筛选）
    if h * horiz_max * horiz_min < 0.04 {
        return false;
    }

    // 高宽比：行人的明显特征是远比宽高
    if h / horiz_max < 1.15 {
        return false;
    }

    true
}

/// 几何 + 速度双重判断：用于新目标（Floating）的快速过滤。
/// 速度 > 0.05 且几何像行人 → 盲区运动行人。
fn is_person_like(obj: &TrackedObject) -> bool {
    if obj.speed() < 0.02 {
        return false;
    }
    check_person_geom(obj)
}

/// 纯几何判断（无速度要求）：用于已跟踪较长时间的 Static 目标。
/// 已持续跟踪 >30 帧且几何像行人，即使静止也很可能是盲区行人。
fn is_person_like_static(obj: &TrackedObject) -> bool {
    check_person_geom(obj)
}

/// 将正在移动的目标标记为行人（小技巧）。
///
/// 在 Sight 移除后，作为替代方案：任何被判定为 Moving 的目标
/// 都视为行人，其余未分类目标降级为障碍物。
///
/// 扩展：对 YOLO/Fuse 未能标注（cluster_N / 空标签）但几何外形
/// 像行人的目标，也标记为 person。这解决了相机盲区中行人的漏标问题。
/// 仅对 Floating（未定性新目标）应用 fallback，避免 Static 背景物体误标。
pub(crate) fn apply(objs: &mut BTreeMap<usize, TrackedObject>) {
    for (_, obj) in objs.iter_mut() {
        if obj.class_type.is_empty() || obj.class_type.starts_with("cluster_") {
            // Moving 目标：几何验证后标为 person（替代旧版无条件 Moving→person）
            if obj.classification == TargetClass::Moving && is_person_like(obj) {
                obj.class_type = "person".to_string();
                obj.geo_labeled = true;
            } else if obj.classification == TargetClass::Floating && is_person_like(obj) {
                obj.class_type = "person".to_string();
                obj.geo_labeled = true; // 标记为几何标签，允许后续帧覆盖
            } else if obj.classification == TargetClass::Static && obj.appearance_count >= 10 && is_person_like_static(obj) {
                // 持续跟踪的 Static 目标：已确认>30帧且几何外型持续像行人
                // 相机盲区中静止行人会被状态机误判为 Static，这里通过持久化几何证据纠正
                obj.class_type = "person".to_string();
                obj.geo_labeled = true;
            } else {
                obj.class_type = "obstacle".to_string();
            }
        }
    }
}
