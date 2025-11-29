use bevy::prelude::*;
use bevy::color::palettes::css::*;

use crate::{cloud, optional::visual::VisResource};

pub fn draw_setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    resource: Res<VisResource>
) {
    let swapl_guard = resource.swapl.lock().unwrap();
    let targets_lock = swapl_guard.targets.lock();
    if let Ok(mut targets) = targets_lock {
        if let Some((target, idx)) = targets.read_indexed() {
            if let Ok(clouds) = swapl_guard.clouds.lock() {
                if let Some(cloud) = clouds.get_at(idx) {
                    print!("具有{}个检测框", target.len());
                    // 绘制雷达点云
                    for point in cloud {
                        commands.spawn((
                            Mesh3d(meshes.add(Sphere::new(0.1).mesh())),
                            MeshMaterial3d(materials.add(StandardMaterial {
                                base_color: DEEP_PINK.into(),
                                ..default()
                            })),
                            Transform::from_xyz(point[0], point[1], point[2]),
                        ));
                    }
                    
                    // 绘制处理后的检测框
                    for tar in target {
                        let trans = [
                            (tar.the_box.x_max + tar.the_box.x_min) / 2.0,
                            (tar.the_box.y_max + tar.the_box.y_min) / 2.0,
                            (tar.the_box.z_max + tar.the_box.z_min) / 2.0
                        ];
                        commands.spawn((
                            Mesh3d(meshes.add(Rectangle::new(2.0, 0.5))),
                            MeshMaterial3d(materials.add(StandardMaterial {
                                perceptual_roughness: 1.0,
                                alpha_mode: AlphaMode::Mask(0.5),
                                cull_mode: None,
                                ..default()
                            })),
                            Transform::from_xyz(trans[0], trans[1], trans[2]),
                        ));

                    }
                }
            }
        }
    }
}