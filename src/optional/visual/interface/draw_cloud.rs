use bevy::prelude::*;
use crate::cloud::Lifra;

/// 点云可视化组件，存储点云数据
#[derive(Component, Clone)]
pub struct PointCloud {
    pub points: Vec<[f32; 3]>,
    pub color: Color,
    pub point_size: f32,
}

impl Default for PointCloud {
    fn default() -> Self {
        Self {
            points: Vec::new(),
            color: Color::srgb(0.0, 1.0, 0.0), // 默认绿色
            point_size: 0.02,
        }
    }
}

impl PointCloud {
    /// 创建新的点云组件
    pub fn new(points: Vec<[f32; 3]>) -> Self {
        Self {
            points,
            color: Color::srgb(0.0, 1.0, 0.0), // 默认绿色
            point_size: 0.02,
        }
    }
    
    /// 创建带颜色的点云组件
    pub fn with_color(mut self, color: Color) -> Self {
        self.color = color;
        self
    }
    
    /// 设置点的大小
    pub fn with_point_size(mut self, size: f32) -> Self {
        self.point_size = size;
        self
    }
}

/// 点云实体标记组件
#[derive(Component)]
pub struct PointCloudRoot;

/// 点云子点标记组件
#[derive(Component)]
pub struct PointEntity;

/// 系统函数，用于渲染点云
pub fn render_point_cloud(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    point_clouds: Query<(Entity, &PointCloud), Added<PointCloud>>,
) {
    for (entity, point_cloud) in point_clouds.iter() {
        // 为每个点创建实体
        commands.entity(entity).with_children(|parent| {
            for point in &point_cloud.points {
                parent.spawn((
                    PointEntity,
                    Mesh3d(meshes.add(Sphere::new(point_cloud.point_size).mesh().ico(3).expect("Failed to create ico sphere mesh"))),
                    MeshMaterial3d(materials.add(point_cloud.color)),
                    Transform::from_translation(Vec3::new(point[0], point[1], point[2])),
                    GlobalTransform::default(),
                ));
            }
        });
    }
}

/// 创建点云束
#[derive(Bundle)]
pub struct PointCloudBundle {
    pub point_cloud: PointCloud,
    pub transform: Transform,
    pub global_transform: GlobalTransform,
    pub visibility: Visibility,
    pub inherited_visibility: InheritedVisibility,
    pub view_visibility: ViewVisibility,
    pub marker: PointCloudRoot,
}

impl PointCloudBundle {
    pub fn from_lifra(lifra: Lifra) -> Self {
        Self {
            point_cloud: PointCloud::new(lifra.points().clone()),
            transform: Transform::default(),
            global_transform: GlobalTransform::default(),
            visibility: Visibility::default(),
            inherited_visibility: InheritedVisibility::default(),
            view_visibility: ViewVisibility::default(),
            marker: PointCloudRoot,
        }
    }
    
    pub fn from_lifra_with_color(lifra: Lifra, color: Color) -> Self {
        Self {
            point_cloud: PointCloud::new(lifra.points().clone()).with_color(color),
            transform: Transform::default(),
            global_transform: GlobalTransform::default(),
            visibility: Visibility::default(),
            inherited_visibility: InheritedVisibility::default(),
            view_visibility: ViewVisibility::default(),
            marker: PointCloudRoot,
        }
    }
}

/// 简单的绘制点云函数
pub fn draw_point_cloud(mut commands: Commands, cloud: Lifra) {
    commands.spawn(PointCloudBundle::from_lifra(cloud));
}

/// 简单的绘制带颜色点云函数
pub fn draw_colored_point_cloud(mut commands: Commands, cloud: Lifra, color: Color) {
    commands.spawn(PointCloudBundle::from_lifra_with_color(cloud, color));
}

/// 简单的绘制点函数
pub fn draw_point(mut commands: Commands, point: [f32; 3]) {
    let lifra = Lifra::from_points(vec![point]);
    commands.spawn(PointCloudBundle::from_lifra(lifra));
}