use bevy::{input::mouse::MouseMotion, prelude::*};

use crate::{optional::visual::VisResource};

// 添加标记组件，用于标识我们创建的可视化对象
#[derive(Component)]
pub struct VisualizedObject;

// 相机控制组件
#[derive(Component)]
pub struct CameraController {
    pub move_speed: f32,
    pub look_speed: f32,
    pub yaw: f32,
    pub pitch: f32,
}

impl Default for CameraController {
    fn default() -> Self {
        Self {
            move_speed: 5.0,
            look_speed: 0.5,
            yaw: 0.0,
            pitch: 0.0,
        }
    }
}

// 设置场景的基本元素：相机和光源
pub fn setup_scene(
    mut commands: Commands,
) {
    // 添加3D相机，带有相机控制器
    commands.spawn((
        Camera3d::default(),
        CameraController::default(),
        Transform::from_xyz(-2.0, 2.5, 5.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
    
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
    
    println!("场景已设置：相机和光源已添加");
}

// 相机控制系统
pub fn camera_controller(
    time: Res<Time>,
    keyboard_input: Res<ButtonInput<KeyCode>>,
    mut mouse_motion: MessageReader<MouseMotion>,
    mut query: Query<(&mut Transform, &mut CameraController)>,
) {
    let delta_time = time.delta_secs();
    
    // 处理鼠标视角控制
    let mut delta_look = Vec2::ZERO;
    for motion in mouse_motion.read() {
        delta_look += motion.delta;
    }
    
    // 应用变换到相机
    for (mut transform, mut controller) in query.iter_mut() {
        // 处理键盘移动输入
        let mut delta_movement = Vec3::ZERO;
        if keyboard_input.pressed(KeyCode::KeyW) {
            delta_movement += transform.forward().as_vec3();
        }
        if keyboard_input.pressed(KeyCode::KeyS) {
            delta_movement += transform.back().as_vec3();
        }
        if keyboard_input.pressed(KeyCode::KeyA) {
            delta_movement += transform.left().as_vec3();
        }
        if keyboard_input.pressed(KeyCode::KeyD) {
            delta_movement += transform.right().as_vec3();
        }
        if keyboard_input.pressed(KeyCode::Space) {
            delta_movement += Vec3::Y;
        }
        if keyboard_input.pressed(KeyCode::ShiftLeft) {
            delta_movement += Vec3::NEG_Y;
        }
        
        // 移动相机
        transform.translation += delta_movement * controller.move_speed * delta_time;
        
        // 视角旋转
        if delta_look.length_squared() > 0.0 {
            controller.yaw -= delta_look.x * controller.look_speed * delta_time;
            controller.pitch -= delta_look.y * controller.look_speed * delta_time;
            
            // 限制俯仰角度
            controller.pitch = controller.pitch.clamp(-std::f32::consts::FRAC_PI_2 + 0.01, std::f32::consts::FRAC_PI_2 - 0.01);
            
            transform.rotation = Quat::from_euler(EulerRot::YXZ, controller.yaw, controller.pitch, 0.0);
        }
    }
}

// 更新可视化内容的系统
pub fn update_visualization(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    resource: Res<VisResource>,
    // 查询当前已有的可视化对象并删除它们
    mut query: Query<Entity, With<VisualizedObject>>,
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
                                commands.spawn((
                                    VisualizedObject,
                                    Mesh3d(meshes.add(Sphere::new(0.05).mesh())),
                                    MeshMaterial3d(materials.add(StandardMaterial {
                                        base_color: Color::srgb(1.0, 0.0, 1.0),
                                        ..default()
                                    })),
                                    Transform::from_xyz(point[0], point[1], point[2]),
                                ));
                                
                                // 每1000个点打印一次进度
                                if idx % 1000 == 0 && idx > 0 {
                                    println!("已绘制{}个点", idx);
                                }
                            }

                            // 绘制处理后的检测框
                            for (i, tar) in targets.iter().enumerate() {
                                let center = [
                                    (tar.the_box.x_max + tar.the_box.x_min) / 2.0,
                                    (tar.the_box.y_max + tar.the_box.y_min) / 2.0,
                                    (tar.the_box.z_max + tar.the_box.z_min) / 2.0
                                ];
                                
                                let size = [
                                    tar.the_box.x_max - tar.the_box.x_min,
                                    tar.the_box.y_max - tar.the_box.y_min,
                                    tar.the_box.z_max - tar.the_box.z_min
                                ];
                                
                                // 确保尺寸合理
                                if size[0] > 0.0 && size[1] > 0.0 && size[2] > 0.0 {
                                    commands.spawn((
                                        VisualizedObject,
                                        Mesh3d(meshes.add(Cuboid::new(size[0], size[1], size[2]))),
                                        MeshMaterial3d(materials.add(StandardMaterial {
                                            base_color: Color::srgb(0.0, 0.0, 1.0), // Blue
                                            alpha_mode: AlphaMode::Blend,
                                            ..default()
                                        })),
                                        Transform::from_xyz(center[0], center[1], center[2]),
                                    ));
                                    println!("绘制检测框{}: 中心({:.2}, {:.2}, {:.2}), 尺寸({:.2}, {:.2}, {:.2})", 
                                             i, center[0], center[1], center[2], size[0], size[1], size[2]);
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