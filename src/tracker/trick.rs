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

/// 基于累计几何验证 + 速度激活的复合跟踪后端行人标记。
///
/// 对 YOLO/Fuse 未能标注（cluster_N / 空标签）或已被几何后端标记为 person 的目标，
/// 使用两条路径 OR 决定标签：
///
/// 1. 几何累计路径：连续几何通过 >= geo_pass_threshold → person
/// 2. 速度激活路径：平滑速度 > geo_speed_threshold → person（单帧即时激活）
///
/// 回退：连续几何失败 >= geo_fail_threshold → obstacle
/// 运动中清空失败计数，避免运动行人因偶发几何异常被降级。
pub(crate) fn apply(
    objs: &mut BTreeMap<usize, TrackedObject>,
    geo_pass_threshold: u32,
    geo_fail_threshold: u32,
    geo_speed_threshold: f32,
) {
    for (_, obj) in objs.iter_mut() {
        let is_unlabeled = obj.class_type.is_empty() || obj.class_type.starts_with("cluster_");
        let is_geo_person = obj.class_type == "person" && obj.geo_labeled;

        // 仅处理未标注或几何标记的目标（YOLO 标注的直接跳过）
        if !is_unlabeled && !is_geo_person {
            continue;
        }

        let geom_pass = check_person_geom(obj);
        let speed = obj.speed();

        // 更新累计计数
        if geom_pass {
            obj.geo_pass_streak += 1;
            obj.geo_fail_streak = 0;
        } else {
            obj.geo_fail_streak += 1;
            obj.geo_pass_streak = 0;
        }

        // 运动中清空失败计数：避免运动行人偶发几何异常被降级
        if obj.classification == TargetClass::Moving || obj.confirmed_moving {
            obj.geo_fail_streak = 0;
        }

        // 决策：几何累计通过 OR 速度激活 → person
        let is_person = obj.geo_pass_streak >= geo_pass_threshold || speed > geo_speed_threshold;

        if is_person {
            obj.class_type = "person".to_string();
            obj.geo_labeled = true;
        // 回退：几何累计失败达到阈值 → obstacle
        } else if obj.geo_fail_streak >= geo_fail_threshold {
            obj.class_type = "obstacle".to_string();
            obj.geo_labeled = false;
        } else if is_unlabeled {
            // 累计未达阈值，临时标为 obstacle（下一帧仍可检查）
            obj.class_type = "obstacle".to_string();
        }
        // geo_person 且两条阈值都未达到 → 保持 person 不动
    }
}
