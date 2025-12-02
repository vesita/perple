use bevy::prelude::*;

/// 线框立方体组件，用于标识线框立方体实体
#[derive(Component)]
pub struct WireframeCube;

/// 创建线框立方体的顶点列表
/// 返回立方体的8个顶点坐标
fn cube_vertices(width: f32, height: f32, depth: f32) -> Vec<[f32; 3]> {
    let half_width = width / 2.0;
    let half_height = height / 2.0;
    let half_depth = depth / 2.0;
    
    vec![
        // 底面四个顶点
        [-half_width, -half_height, -half_depth],  // 0: 左前下
        [half_width, -half_height, -half_depth],   // 1: 右前下
        [half_width, -half_height, half_depth],    // 2: 右后下
        [-half_width, -half_height, half_depth],   // 3: 左后下
        
        // 顶面四个顶点
        [-half_width, half_height, -half_depth],   // 4: 左前上
        [half_width, half_height, -half_depth],    // 5: 右前上
        [half_width, half_height, half_depth],     // 6: 右后上
        [-half_width, half_height, half_depth],    // 7: 左后上
    ]
}

/// 创建线框立方体的索引列表
/// 定义立方体12条边的顶点连接关系
fn cube_indices() -> Vec<u32> {
    vec![
        // 底面四条边
        0, 1,  1, 2,  2, 3,  3, 0,
        // 顶面四条边
        4, 5,  5, 6,  6, 7,  7, 4,
        // 垂直四条边
        0, 4,  1, 5,  2, 6,  3, 7,
    ]
}

/// 创建线框立方体网格
pub fn create_wireframe_cube_mesh(
    meshes: &mut Assets<Mesh>,
    width: f32,
    height: f32,
    depth: f32,
) -> Mesh3d {
    let vertices = cube_vertices(width, height, depth);
    let indices = cube_indices();
    
    // 创建线段顶点列表（每条线段两个顶点）
    let mut positions: Vec<[f32; 3]> = Vec::new();
    for i in (0..indices.len()).step_by(2) {
        positions.push(vertices[indices[i] as usize]);
        positions.push(vertices[indices[i + 1] as usize]);
    }
    
    // 使用Bevy提供的正确API创建线段网格
    let mut mesh = Mesh::new(
        bevy::render::render_resource::PrimitiveTopology::LineList,
        Default::default()
    );
    
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    
    Mesh3d(meshes.add(mesh))
}

/// 生成线框立方体
pub fn spawn_wireframe_cube(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    position: Vec3,
    size: Vec3,
    color: Color,
) -> Entity {
    let mesh = create_wireframe_cube_mesh(
        meshes.as_mut(),
        size.x,
        size.y,
        size.z,
    );
    
    let material = MeshMaterial3d(materials.add(StandardMaterial {
        base_color: color,
        unlit: true,
        ..default()
    }));
    
    commands.spawn((
        WireframeCube,
        mesh,
        material,
        Transform::from_translation(position),
        Visibility::default(),
    )).id()
}

/// 更新现有线框立方体的大小和位置
pub fn update_wireframe_cube(
    mut cube_query: Query<(&mut Mesh3d, &mut Transform), With<WireframeCube>>,
    meshes: &mut ResMut<Assets<Mesh>>,
    entity: Entity,
    position: Vec3,
    size: Vec3,
) {
    if let Ok((mut mesh_handle, mut transform)) = cube_query.get_mut(entity) {
        // 更新网格
        let new_mesh = create_wireframe_cube_mesh(
            meshes.as_mut(),
            size.x,
            size.y,
            size.z,
        );
        *mesh_handle = new_mesh;
        
        // 更新位置
        transform.translation = position;
    }
}