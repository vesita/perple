// use bevy::prelude::*;
// use smooth_bevy_cameras::{
//     controllers::fps::{FpsCameraBundle, FpsCameraController},
// };

// use crate::{optional::visual::utils::{coordinate::z_up_to_y_up, wirefra::spawn_wireframe_cube}, swapl::global_swapl};

// // 设置场景的基本元素：相机和光源
// pub fn setup_scene(
//     mut commands: Commands,
// ) {    
//     // 添加点光源
//     commands.spawn((
//         PointLight {
//             intensity: 1500.0,
//             shadows_enabled: true,
//             ..default()
//         },
//         Transform::from_xyz(4.0, 8.0, 4.0),
//     ));
    
//     // 添加环境光
//     commands.insert_resource(AmbientLight {
//         color: Color::WHITE,
//         brightness: 100.0,
//         ..default()
//     });
    
//     // 使用FPS相机控制器
//     commands
//         .spawn(Camera3d::default())
//         .insert(FpsCameraBundle::new(
//             FpsCameraController {
//                 mouse_rotate_sensitivity: Vec2::new(0.2, 0.2),
//                 ..Default::default()
//             },
//             Vec3::new(-2.5, 4.5, 9.0),  // 相机位置
//             Vec3::new(0.0, 0.0, 0.0),    // 看向的目标点
//             Vec3::Y,
//         ));

//     println!("场景已设置：相机和光源已添加");
// }


// // 更新可视化内容的系统
// pub fn update_visualization(
//     mut commands: Commands,
//     mut meshes: ResMut<Assets<Mesh>>,
//     mut materials: ResMut<Assets<StandardMaterial>>,
// ) {
//     let swapl = global_swapl();
//     // 尝试获取targets数据
//     match swapl.targets.lock() {
//         Ok(mut targets_lock) => {
//             // 使用read_indexed读取数据和索引
//             if let Some((targets, idx)) = targets_lock.read_indexed() {
//                 // 使用lidar实例获取经过坐标变换的点云数据
//                 match swapl.cloud_in_world.lock() {
//                     Ok(cloud_stream) => {
//                         if let Some(cloud) = cloud_stream.get_at(idx) {
//                             println!("正在绘制{}个点和{}个检测框", cloud.len(), targets.len());
                                                    
//                             for (point_idx, point) in cloud.iter().enumerate() {
//                                 // 点云数据是Z-up坐标系，需要转换为Y-up坐标系以适配Bevy
//                                 let point_vec3 = z_up_to_y_up(Vec3::new(point[0], point[1], point[2]));
                                
//                                 commands.spawn((
//                                     Mesh3d(meshes.add(Sphere::new(0.05).mesh())),
//                                     MeshMaterial3d(materials.add(StandardMaterial {
//                                         base_color: Color::srgb(1.0, 0.0, 1.0),
//                                         ..default()
//                                     })),
//                                     Transform::from_xyz(point_vec3.x, point_vec3.y, point_vec3.z),
//                                 ));
                                
//                                 // 每4000个点打印一次进度
//                                 if point_idx % 4000 == 0 && point_idx > 0 {
//                                     println!("已绘制{}个点", point_idx);
//                                 }
//                             }

//                             // 绘制处理后的检测框（使用手动线框实现）
//                             for tar in &targets {
//                                 // Box3D也是Z-up坐标系，需要转换为Y-up坐标系
//                                 let center = tar.the_box.center_y_up();
//                                 let size = tar.the_box.shape_y_up();
//                                 let size = Vec3::new(size.x.abs(), size.y.abs(), size.z.abs());
                                
//                                 spawn_wireframe_cube(
//                                     &mut commands,
//                                     &mut meshes,
//                                     &mut materials,
//                                     center,
//                                     size,
//                                     Color::srgb(0.0, 0.0, 1.0), // Blue
//                                 );
//                             }
//                             println!("完成绘制: {}个点, {}个检测框", cloud.len(), targets.len());
//                         } else {
//                             println!("未能获取索引{}处的点云数据", idx);
//                         }
//                     }
//                     Err(e) => {
//                         println!("无法锁定cloud_in_world实例: {}", e);
//                     }
//                 }
//             }
//         }
//         Err(e) => {
//             println!("无法锁定目标数据: {}", e);
//         }
//     }
// }