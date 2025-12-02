//! 坐标转换模块
//! 
//! 该模块提供在不同坐标系之间进行转换的功能，特别适用于将Perple模块输出的Z-up坐标系
//! 转换为游戏引擎常用的Y-up坐标系。

use bevy::prelude::*;

/// 将Z-up坐标系转换为Y-up坐标系
/// 
/// 在Z-up坐标系中，Z轴指向上方
/// 在Y-up坐标系中，Y轴指向上方
/// 
/// 转换规则：
/// - X轴保持不变
/// - Y轴 = -Z轴（原来的Z轴变为新的Y轴，符号取反)
/// - Z轴 = Y轴（原来的Y轴变为新的Z轴）
/// 
/// # 参数
/// * `position` - Z-up坐标系中的位置
/// 
/// # 返回值
/// Y-up坐标系中的位置
pub fn z_up_to_y_up(position: Vec3) -> Vec3 {
    Vec3::new(
        position.x,     // X轴保持不变
        -position.z,    // Y = -Z
        position.y,     // Z = Y
    )
}

/// 将Y-up坐标系转换为Z-up坐标系
/// 
/// 转换规则：
/// - X轴保持不变
/// - Z轴 = -Y轴（原来的Y轴变为新的Z轴，符号取反)
/// - Y轴 = Z轴（原来的Z轴变为新的Y轴）
/// 
/// # 参数
/// * `position` - Y-up坐标系中的位置
/// 
/// # 返回值
/// Z-up坐标系中的位置
pub fn y_up_to_z_up(position: Vec3) -> Vec3 {
    Vec3::new(
        position.x,     // X轴保持不变
        position.z,     // Y = Z
        -position.y,    // Z = -Y
    )
}

/// 转换包围盒坐标从Z-up到Y-up
/// 
/// # 参数
/// * `min` - 包围盒最小点 (Z-up)
/// * `max` - 包围盒最大点 (Z-up)
/// 
/// # 返回值
/// 转换后的包围盒最小点和最大点 (Y-up)
pub fn bounds_z_up_to_y_up(min: Vec3, max: Vec3) -> (Vec3, Vec3) {
    // 先转换两个对角点
    let min_y_up = z_up_to_y_up(min);
    let max_y_up = z_up_to_y_up(max);
    
    // 计算新的包围盒
    let new_min = min_y_up.min(max_y_up);
    let new_max = min_y_up.max(max_y_up);
    
    (new_min, new_max)
}

/// 转换包围盒坐标从Y-up到Z-up
/// 
/// # 参数
/// * `min` - 包围盒最小点 (Y-up)
/// * `max` - 包围盒最大点 (Y-up)
/// 
/// # 返回值
/// 转换后的包围盒最小点和最大点 (Z-up)
pub fn bounds_y_up_to_z_up(min: Vec3, max: Vec3) -> (Vec3, Vec3) {
    // 先转换两个对角点
    let min_z_up = y_up_to_z_up(min);
    let max_z_up = y_up_to_z_up(max);
    
    // 计算新的包围盒
    let new_min = min_z_up.min(max_z_up);
    let new_max = min_z_up.max(max_z_up);
    
    (new_min, new_max)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_z_up_to_y_up() {
        // 测试点 (1, 2, 3) 在Z-up坐标系中
        // 转换后应为 (1, -3, 2) 在Y-up坐标系中
        let z_up_point = Vec3::new(1.0, 2.0, 3.0);
        let y_up_point = z_up_to_y_up(z_up_point);
        let expected = Vec3::new(1.0, -3.0, 2.0);
        
        assert!((y_up_point.x - expected.x).abs() < f32::EPSILON);
        assert!((y_up_point.y - expected.y).abs() < f32::EPSILON);
        assert!((y_up_point.z - expected.z).abs() < f32::EPSILON);
    }
    
    #[test]
    fn test_y_up_to_z_up() {
        // 测试点 (1, 2, 3) 在Y-up坐标系中
        // 转换后应为 (1, 3, -2) 在Z-up坐标系中
        let y_up_point = Vec3::new(1.0, 2.0, 3.0);
        let z_up_point = y_up_to_z_up(y_up_point);
        let expected = Vec3::new(1.0, 3.0, -2.0);
        
        assert!((z_up_point.x - expected.x).abs() < f32::EPSILON);
        assert!((z_up_point.y - expected.y).abs() < f32::EPSILON);
        assert!((z_up_point.z - expected.z).abs() < f32::EPSILON);
    }
    
    #[test]
    fn test_inverse_conversion() {
        let original = Vec3::new(1.0, 2.0, 3.0);
        let converted = z_up_to_y_up(original);
        let back = y_up_to_z_up(converted);
        
        assert!((original.x - back.x).abs() < f32::EPSILON);
        assert!((original.y - back.y).abs() < f32::EPSILON);
        assert!((original.z - back.z).abs() < f32::EPSILON);
    }
}