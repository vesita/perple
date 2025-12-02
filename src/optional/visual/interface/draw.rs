use bevy::prelude::*;
use smooth_bevy_cameras::{
    controllers::fps::{FpsCameraBundle, FpsCameraController},
};

use crate::{optional::visual::{VisResource, utils::{wirefra::{spawn_wireframe_cube, WireframeCube}, coordinate::y_up_to_z_up}}};

// 添加标记组件，用于标识我们创建的可视化对象
#[derive(Component)]
pub struct VisualizedObject;

// 设置场景的基本元素：相机和光源
pub fn setup_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {    
    // 添加点光源
    commands.spawn((
        PointLight {
            intensity: 1500.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(4.0, 8.0, 4.0),
    ));
    
    // 添加环境光
    commands.insert_resource(AmbientLight {
        color: Color::WHITE,
        brightness: 100.0,
        ..default()
    });
    
    // 添加一个圆形基座以便更好地观察3D空间
    commands.spawn((
        Mesh3d(meshes.add(Circle::new(4.0))),
        MeshMaterial3d(materials.add(Color::WHITE)),
        Transform::from_rotation(Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2)),
    ));
    
    // 使用FPS相机控制器
    commands
        .spawn(Camera3d::default())
        .insert(FpsCameraBundle::new(
            FpsCameraController {
                mouse_rotate_sensitivity: Vec2::new(0.2, 0.2),
                ..Default::default()
            },
            Vec3::new(-2.5, 4.5, 9.0),  // 相机位置
            Vec3::new(0.0, 0.0, 0.0),    // 看向的目标点
            Vec3::Y,
        ));

    println!("场景已设置：相机和光源已添加");
}


// 更新可视化内容的系统
pub fn update_visualization(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    resource: Res<VisResource>,
    // 查询当前已有的可视化对象并删除它们
    mut query: Query<Entity, With<VisualizedObject>>,
    // 查询线框立方体
    _wireframe_query: Query<Entity, With<WireframeCube>>,
) {

    let swapl = &resource.swapl;
    
    // 尝试获取targets数据
    match swapl.targets.lock() {
        Ok(mut targets_lock) => {
            // 使用read_indexed读取数据和索引
            if let Some((targets, idx)) = targets_lock.read_indexed() {
                // 尝试获取点云数据
                match swapl.clouds.lock() {
                    Ok(clouds) => {
                        // 使用索引获取对应的点云数据
                        if let Some(cloud) = clouds.get_at(idx) {
                            println!("正在绘制{}个点和{}个检测框", cloud.len(), targets.len());
                            for (idx, point) in cloud.iter().enumerate() {
                                // 将Y-up坐标转换为Z-up坐标
                                let point_vec3 = y_up_to_z_up(Vec3::new(point[0], point[1], point[2]));
                                
                                commands.spawn((
                                    VisualizedObject,
                                    Mesh3d(meshes.add(Sphere::new(0.05).mesh())),
                                    MeshMaterial3d(materials.add(StandardMaterial {
                                        base_color: Color::srgb(1.0, 0.0, 1.0),
                                        ..default()
                                    })),
                                    Transform::from_xyz(point_vec3.x, point_vec3.y, point_vec3.z),
                                ));
                                
                                // 每1000个点打印一次进度
                                if idx % 1000 == 0 && idx > 0 {
                                    println!("已绘制{}个点", idx);
                                }
                            }

                            // 绘制处理后的检测框（使用手动线框实现）
                            for (i, tar) in targets.iter().enumerate() {
                                let center = y_up_to_z_up(Vec3::new(
                                    (tar.the_box.x_max + tar.the_box.x_min) / 2.0,
                                    (tar.the_box.y_max + tar.the_box.y_min) / 2.0,
                                    (tar.the_box.z_max + tar.the_box.z_min) / 2.0
                                ));
                                
                                let size = y_up_to_z_up(Vec3::new(
                                    tar.the_box.x_max - tar.the_box.x_min,
                                    tar.the_box.y_max - tar.the_box.y_min,
                                    tar.the_box.z_max - tar.the_box.z_min
                                ));
                                
                                let size = Vec3::new(size.x.abs(), size.y.abs(), size.z.abs());
                                
                                // 确保尺寸合理
                                if size.x > 0.0 && size.y > 0.0 && size.z > 0.0 {
                                    spawn_wireframe_cube(
                                        &mut commands,
                                        &mut meshes,
                                        &mut materials,
                                        center,
                                        size,
                                        Color::srgb(0.0, 0.0, 1.0), // Blue
                                    );
                                    
                                    println!("绘制检测框{}: 中心({:.2}, {:.2}, {:.2}), 尺寸({:.2}, {:.2}, {:.2})", 
                                             i, center.x, center.y, center.z, size.x, size.y, size.z);
                                }
                            }
                            // 删除之前的可视化对象
                            for entity in query.iter_mut() {
                                commands.entity(entity).despawn();
                            }
                            println!("完成绘制: {}个点, {}个检测框", cloud.len(), targets.len());
                        } else {
                            println!("未能获取索引{}处的点云数据", idx);
                        }
                    }
                    Err(e) => {
                        println!("无法锁定点云数据: {}", e);
                    }
                }
            }
        }
        Err(e) => {
            println!("无法锁定目标数据: {}", e);
        }
    }
}