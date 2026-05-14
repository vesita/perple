use std::collections::HashMap;

use super::core::{TargetClass, TrackedObject};

/// 将正在移动的目标标记为行人（小技巧）。
///
/// 在 Sight 移除后，作为替代方案：任何被判定为 Moving 的目标
/// 都视为行人，其余未分类目标降级为障碍物。
pub(crate) fn apply(objs: &mut HashMap<usize, TrackedObject>) {
    for (_, obj) in objs.iter_mut() {
        if obj.classification == TargetClass::Moving {
            obj.class_type = "person".to_string();
        } else if obj.class_type.is_empty() || obj.class_type.starts_with("cluster_") {
            obj.class_type = "obstacle".to_string();
        }
    }
}
