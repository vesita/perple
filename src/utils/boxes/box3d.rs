use nalgebra::{Matrix4, Point3, Vector3};

use super::bev_iou::{clip_polygon_2d, convex_hull_2d, polygon_area_2d};
use super::obb_iou::{clip_triangle_by_plane, triangle_mesh_volume};

#[derive(Debug, Clone)]
pub struct Box3D {
    /// 位姿矩阵：表示包围盒在世界坐标系中的位置和方向
    pub pose: Matrix4<f32>,
    /// 长度（x方向）
    pub length: f32,
    /// 宽度（y方向）
    pub width: f32,
    /// 高度（z方向）
    pub height: f32,
}

impl Box3D {
    pub fn new(pose: Matrix4<f32>, length: f32, width: f32, height: f32) -> Self {
        Box3D {
            pose,
            length,
            width,
            height,
        }
    }

    /// 到目标点的 XY 平面距离（忽略 Z）
    pub fn xy_distance_to(&self, target: [f32; 3]) -> f32 {
        let c = self.center();
        let dx = c.x - target[0];
        let dy = c.y - target[1];
        (dx * dx + dy * dy).sqrt()
    }

    /// 中心是否在以 origin 为圆心、max_range 为半径的 XY 圆内
    pub fn is_in_xy_range(&self, origin: [f32; 3], max_range: f32) -> bool {
        self.xy_distance_to(origin) <= max_range
    }

    /// 从平移和欧拉角创建Tag3D
    ///
    /// # 参数
    /// * `x, y, z` - 平移分量
    /// * `rx, ry, rz` - 欧拉角（弧度）
    /// * `length, width, height` - 包围盒尺寸
    ///
    /// # 返回值
    /// 新创建的Tag3D实例
    pub fn from_position_and_angles(
        x: f32,
        y: f32,
        z: f32,
        rx: f32,
        ry: f32,
        rz: f32,
        length: f32,
        width: f32,
        height: f32,
    ) -> Self {
        let translation = Matrix4::new_translation(&Vector3::new(x, y, z));
        let rotation = Matrix4::from_euler_angles(rx, ry, rz);
        let pose = translation * rotation;
        Box3D::new(pose, length, width, height)
    }

    /// 创建空的包围盒
    pub fn empty_box() -> Self {
        Box3D {
            pose: Matrix4::identity(),
            length: 0.0,
            width: 0.0,
            height: 0.0,
        }
    }

    /// 检查点是否在包围盒内
    pub fn contains(&self, point: &[f32; 3]) -> bool {
        let local_point = self.to_local(point);
        local_point.x >= -self.length / 2.0
            && local_point.x <= self.length / 2.0
            && local_point.y >= -self.width / 2.0
            && local_point.y <= self.width / 2.0
            && local_point.z >= -self.height / 2.0
            && local_point.z <= self.height / 2.0
    }

    /// 扩展包围盒以包含指定点
    pub fn expand(&mut self, point: &[f32; 3]) -> Result<(), String> {
        if self.length == 0.0 && self.width == 0.0 && self.height == 0.0 {
            let position = Vector3::new(point[0], point[1], point[2]);
            self.pose = Matrix4::new_translation(&position);
            return Ok(());
        }

        let point_world = Point3::new(point[0], point[1], point[2]);
        let inv_pose = self.pose.try_inverse().ok_or("矩阵不可求逆".to_string())?;
        let point_local = inv_pose.transform_point(&point_world);

        let x_min = point_local.x.min(-self.length / 2.0);
        let x_max = point_local.x.max(self.length / 2.0);
        let y_min = point_local.y.min(-self.width / 2.0);
        let y_max = point_local.y.max(self.width / 2.0);
        let z_min = point_local.z.min(-self.height / 2.0);
        let z_max = point_local.z.max(self.height / 2.0);

        self.length = x_max - x_min;
        self.width = y_max - y_min;
        self.height = z_max - z_min;

        let new_center_local = Point3::new(
            (x_min + x_max) / 2.0,
            (y_min + y_max) / 2.0,
            (z_min + z_max) / 2.0,
        );

        let new_center_world = self.pose.transform_point(&new_center_local);

        let mut new_pose = self.pose;
        new_pose[(0, 3)] = new_center_world.x;
        new_pose[(1, 3)] = new_center_world.y;
        new_pose[(2, 3)] = new_center_world.z;
        self.pose = new_pose;

        Ok(())
    }

    /// 从点云计算 AABB（轴对齐包围盒），自动 clamp 最小边长到 `min_edge`（米）
    pub fn from_cloud_aabb(cloud3d: &[[f32; 3]], min_edge: f32) -> Self {
        if cloud3d.is_empty() {
            return Self::empty_box();
        }

        let mut x_min = cloud3d[0][0];
        let mut x_max = cloud3d[0][0];
        let mut y_min = cloud3d[0][1];
        let mut y_max = cloud3d[0][1];
        let mut z_min = cloud3d[0][2];
        let mut z_max = cloud3d[0][2];

        for p in &cloud3d[1..] {
            x_min = x_min.min(p[0]);
            x_max = x_max.max(p[0]);
            y_min = y_min.min(p[1]);
            y_max = y_max.max(p[1]);
            z_min = z_min.min(p[2]);
            z_max = z_max.max(p[2]);
        }

        let cx = (x_min + x_max) * 0.5;
        let cy = (y_min + y_max) * 0.5;
        let cz = (z_min + z_max) * 0.5;

        Box3D {
            pose: Matrix4::new_translation(&Vector3::new(cx, cy, cz)),
            length: (x_max - x_min).max(min_edge),
            width:  (y_max - y_min).max(min_edge),
            height: (z_max - z_min).max(min_edge),
        }
    }

    /// 从点云创建包围盒（旧接口，委托给 `from_cloud_aabb`）
    pub fn cloud2box(&mut self, cloud3d: &Vec<[f32; 3]>) {
        *self = Self::from_cloud_aabb(cloud3d, 0.0);
    }

    /// 从2D点云创建包围盒（俯视视角）
    pub fn look_down(&mut self, cloud2d: &Vec<[f32; 2]>) {
        if cloud2d.is_empty() {
            return;
        }

        let mut x_min = cloud2d[0][0];
        let mut x_max = cloud2d[0][0];
        let mut y_min = cloud2d[0][1];
        let mut y_max = cloud2d[0][1];

        for point in cloud2d {
            x_min = x_min.min(point[0]);
            x_max = x_max.max(point[0]);
            y_min = y_min.min(point[1]);
            y_max = y_max.max(point[1]);
        }

        let center_x = (x_min + x_max) / 2.0;
        let center_y = (y_min + y_max) / 2.0;

        self.pose = Matrix4::new_translation(&Vector3::new(center_x, center_y, 0.0));

        self.length = x_max - x_min;
        self.width = y_max - y_min;
        self.height = f32::MAX;
    }

    /// 获取包围盒中心点（世界坐标系）
    pub fn center(&self) -> Vector3<f32> {
        Vector3::new(self.pose[(0, 3)], self.pose[(1, 3)], self.pose[(2, 3)])
    }

    /// 获取包围盒尺寸（长、宽、高）
    pub fn shape(&self) -> Vector3<f32> {
        Vector3::new(self.length, self.width, self.height)
    }

    pub fn center_single(&self) -> [f32; 3] {
        [self.pose[(0, 3)], self.pose[(1, 3)], self.pose[(2, 3)]]
    }

    /// 计算点到包围盒中心的距离
    pub fn distance_to_point(&self, point: &[f32; 3]) -> f32 {
        let center = self.center();
        let dx = point[0] - center.x;
        let dy = point[1] - center.y;
        let dz = point[2] - center.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// 获取包围盒的八个顶点
    pub fn vertices(&self) -> [Point3<f32>; 8] {
        let half_length = self.length / 2.0;
        let half_width = self.width / 2.0;
        let half_height = self.height / 2.0;

        let local_vertices = [
            Point3::new(-half_length, -half_width, -half_height),
            Point3::new(half_length, -half_width, -half_height),
            Point3::new(half_length, half_width, -half_height),
            Point3::new(-half_length, half_width, -half_height),
            Point3::new(-half_length, -half_width, half_height),
            Point3::new(half_length, -half_width, half_height),
            Point3::new(half_length, half_width, half_height),
            Point3::new(-half_length, half_width, half_height),
        ];

        let mut world_vertices = [Point3::origin(); 8];
        for (i, local_vertex) in local_vertices.iter().enumerate() {
            world_vertices[i] = self.pose.transform_point(local_vertex);
        }

        world_vertices
    }

    pub fn edges(&self) -> Vec<[[f32; 3]; 2]> {
        let vertices = self.vertices();
        let cube_edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7],
        ];
        let mut result = Vec::new();
        for &edge in cube_edges.iter() {
            let vertex1 = [
                vertices[edge[0]].x,
                vertices[edge[0]].y,
                vertices[edge[0]].z,
            ];
            let vertex2 = [
                vertices[edge[1]].x,
                vertices[edge[1]].y,
                vertices[edge[1]].z,
            ];
            result.push([vertex1, vertex2]);
        }
        result
    }

    pub fn edges_z_up(&self) -> Vec<[[f32; 3]; 2]> {
        let vertices = self.vertices();
        let cube_edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7],
        ];
        let mut result = Vec::new();
        for &edge in cube_edges.iter() {
            let vertex1 = [
                vertices[edge[0]].x,
                vertices[edge[0]].z,
                vertices[edge[0]].y,
            ];
            let vertex2 = [
                vertices[edge[1]].x,
                vertices[edge[1]].z,
                vertices[edge[1]].y,
            ];
            result.push([vertex1, vertex2]);
        }
        result
    }

    /// 检查点是否在包围盒附近（在一定距离内）
    pub fn near(&self, point: &[f32; 3], distance: f32) -> bool {
        let center = self.center();
        let center_distance_sq = (point[0] - center.x).powi(2)
            + (point[1] - center.y).powi(2)
            + (point[2] - center.z).powi(2);

        let max_radius = ((self.length / 2.0).powi(2)
            + (self.width / 2.0).powi(2)
            + (self.height / 2.0).powi(2))
        .sqrt();

        if center_distance_sq > (max_radius + distance).powi(2) {
            return false;
        }

        if max_radius >= distance && center_distance_sq <= (max_radius - distance).powi(2) {
            return true;
        }

        let vertices = self.vertices();

        let mut x_min = vertices[0].x;
        let mut x_max = vertices[0].x;
        let mut y_min = vertices[0].y;
        let mut y_max = vertices[0].y;
        let mut z_min = vertices[0].z;
        let mut z_max = vertices[0].z;

        for vertex in &vertices {
            x_min = x_min.min(vertex.x);
            x_max = x_max.max(vertex.x);
            y_min = y_min.min(vertex.y);
            y_max = y_max.max(vertex.y);
            z_min = z_min.min(vertex.z);
            z_max = z_max.max(vertex.z);
        }

        let x = point[0].clamp(x_min, x_max);
        let y = point[1].clamp(y_min, y_max);
        let z = point[2].clamp(z_min, z_max);

        let dx = x - point[0];
        let dy = y - point[1];
        let dz = z - point[2];

        dx * dx + dy * dy + dz * dz <= distance * distance
    }

    /// 将世界坐标系中的点转换到包围盒的局部坐标系中
    pub fn to_local(&self, point: &[f32; 3]) -> Point3<f32> {
        let point = Point3::new(point[0], point[1], point[2]);
        if let Some(inv_pose) = self.pose.try_inverse() {
            inv_pose.transform_point(&point)
        } else {
            panic!("矩阵不可求逆: {}", self.pose);
        }
    }

    pub fn to_y_up(&self, point: &Point3<f32>) -> Point3<f32> {
        Point3::new(point.x, point.z, point.y)
    }

    /// 将局部坐标系中的点转换到世界坐标系中
    pub fn to_world(&self, point: &Point3<f32>) -> [f32; 3] {
        let world_point = self.pose.transform_point(point);
        [world_point.x, world_point.y, world_point.z]
    }

    /// 计算两个Box3D的交并比(IOU)
    pub fn iou(&self, other: &Self) -> f32 {
        let vertices1 = self.vertices();
        let vertices2 = other.vertices();

        let mut x_min1 = vertices1[0].x;
        let mut x_max1 = vertices1[0].x;
        let mut y_min1 = vertices1[0].y;
        let mut y_max1 = vertices1[0].y;
        let mut z_min1 = vertices1[0].z;
        let mut z_max1 = vertices1[0].z;

        for vertex in &vertices1 {
            x_min1 = x_min1.min(vertex.x);
            x_max1 = x_max1.max(vertex.x);
            y_min1 = y_min1.min(vertex.y);
            y_max1 = y_max1.max(vertex.y);
            z_min1 = z_min1.min(vertex.z);
            z_max1 = z_max1.max(vertex.z);
        }

        let mut x_min2 = vertices2[0].x;
        let mut x_max2 = vertices2[0].x;
        let mut y_min2 = vertices2[0].y;
        let mut y_max2 = vertices2[0].y;
        let mut z_min2 = vertices2[0].z;
        let mut z_max2 = vertices2[0].z;

        for vertex in &vertices2 {
            x_min2 = x_min2.min(vertex.x);
            x_max2 = x_max2.max(vertex.x);
            y_min2 = y_min2.min(vertex.y);
            y_max2 = y_max2.max(vertex.y);
            z_min2 = z_min2.min(vertex.z);
            z_max2 = z_max2.max(vertex.z);
        }

        let inter_x_min = x_min1.max(x_min2);
        let inter_x_max = x_max1.min(x_max2);
        let inter_y_min = y_min1.max(y_min2);
        let inter_y_max = y_max1.min(y_max2);
        let inter_z_min = z_min1.max(z_min2);
        let inter_z_max = z_max1.min(z_max2);

        if inter_x_min >= inter_x_max || inter_y_min >= inter_y_max || inter_z_min >= inter_z_max {
            return 0.0;
        }

        let intersection_volume =
            (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min) * (inter_z_max - inter_z_min);

        let volume1 = (x_max1 - x_min1) * (y_max1 - y_min1) * (z_max1 - z_min1);
        let volume2 = (x_max2 - x_min2) * (y_max2 - y_min2) * (z_max2 - z_min2);

        let union_volume = volume1 + volume2 - intersection_volume;

        if union_volume == 0.0 { 0.0 } else { intersection_volume / union_volume }
    }

    /// 计算包围盒体积
    pub fn volume(&self) -> f32 {
        self.length * self.width * self.height
    }

    /// 获取局部坐标轴在世界坐标系中的方向向量
    fn axes(&self) -> [Vector3<f32>; 3] {
        [
            Vector3::new(self.pose[(0, 0)], self.pose[(1, 0)], self.pose[(2, 0)]),
            Vector3::new(self.pose[(0, 1)], self.pose[(1, 1)], self.pose[(2, 1)]),
            Vector3::new(self.pose[(0, 2)], self.pose[(1, 2)], self.pose[(2, 2)]),
        ]
    }

    /// 获取 6 个面平面（向内法线 + 平面常数 d），满足 normal·p + d ≥ 0 表示在盒内
    fn face_planes(&self) -> [(Vector3<f32>, f32); 6] {
        let axes = self.axes();
        let c = self.center();
        let hl = self.length / 2.0;
        let hw = self.width / 2.0;
        let hh = self.height / 2.0;
        [
            (-axes[0],  axes[0].dot(&c) + hl),
            ( axes[0], -axes[0].dot(&c) + hl),
            (-axes[1],  axes[1].dot(&c) + hw),
            ( axes[1], -axes[1].dot(&c) + hw),
            (-axes[2],  axes[2].dot(&c) + hh),
            ( axes[2], -axes[2].dot(&c) + hh),
        ]
    }

    /// 获取 12 个三角形（6 个面各 2 个），顶点为世界坐标，从外部看逆时针绕向
    fn triangles(&self) -> Vec<[Point3<f32>; 3]> {
        let hl = self.length / 2.0;
        let hw = self.width / 2.0;
        let hh = self.height / 2.0;
        let lv: [Point3<f32>; 8] = [
            Point3::new(-hl, -hw, -hh),
            Point3::new( hl, -hw, -hh),
            Point3::new( hl,  hw, -hh),
            Point3::new(-hl,  hw, -hh),
            Point3::new(-hl, -hw,  hh),
            Point3::new( hl, -hw,  hh),
            Point3::new( hl,  hw,  hh),
            Point3::new(-hl,  hw,  hh),
        ];
        let v: Vec<Point3<f32>> = lv.iter().map(|p| self.pose.transform_point(p)).collect();
        vec![
            [v[4], v[5], v[6]], [v[4], v[6], v[7]],
            [v[1], v[0], v[3]], [v[1], v[3], v[2]],
            [v[1], v[2], v[6]], [v[1], v[6], v[5]],
            [v[0], v[4], v[7]], [v[0], v[7], v[3]],
            [v[3], v[7], v[6]], [v[3], v[6], v[2]],
            [v[0], v[1], v[5]], [v[0], v[5], v[4]],
        ]
    }

    /// 真 3D OBB 交并比（通过三角形网格裁剪计算交集体积）
    pub fn obb_iou(&self, other: &Self) -> f32 {
        let tri_b = other.triangles();
        let planes_a = self.face_planes();
        let planes_b = other.face_planes();

        let mut current: Vec<[Point3<f32>; 3]> = tri_b;
        let mut next = Vec::new();

        for (n, d) in &planes_a {
            next.clear();
            for &tri in &current {
                clip_triangle_by_plane(tri, n, *d, &mut next);
            }
            if next.is_empty() {
                return 0.0;
            }
            std::mem::swap(&mut current, &mut next);
        }

        for (n, d) in &planes_b {
            next.clear();
            for &tri in &current {
                clip_triangle_by_plane(tri, n, *d, &mut next);
            }
            if next.is_empty() {
                return 0.0;
            }
            std::mem::swap(&mut current, &mut next);
        }

        let intersection_vol = triangle_mesh_volume(&current);
        let vol_a = self.volume();
        let vol_b = other.volume();
        let union_vol = vol_a + vol_b - intersection_vol;

        if union_vol <= 1e-12 { 0.0 } else { intersection_vol / union_vol }
    }

    /// BEV (Bird's Eye View) 2D IoU — 投影到 XY 平面计算 2D 交并比
    pub fn bev_iou(&self, other: &Self) -> f32 {
        let poly1 = self.bev_projection();
        let poly2 = other.bev_projection();

        let intersection = clip_polygon_2d(&poly1, &poly2);
        if intersection.len() < 3 {
            return 0.0;
        }

        let inter_area = polygon_area_2d(&intersection);
        let area1 = polygon_area_2d(&poly1);
        let area2 = polygon_area_2d(&poly2);
        let union_area = area1 + area2 - inter_area;

        if union_area <= 1e-12 { 0.0 } else { inter_area / union_area }
    }

    /// 将 Box3D 投影到 XY 平面，返回 2D 凸包顶点（逆时针绕向）
    fn bev_projection(&self) -> Vec<(f32, f32)> {
        let verts = self.vertices();
        let projected: Vec<(f32, f32)> = verts.iter().map(|p| (p.x, p.y)).collect();
        convex_hull_2d(&projected)
    }

    /// 合并两个Box3D对象
    pub fn merge(&self, other: &Self) -> Self {
        let vertices1 = self.vertices();
        let vertices2 = other.vertices();

        let mut all_vertices = Vec::new();
        all_vertices.extend_from_slice(&vertices1);
        all_vertices.extend_from_slice(&vertices2);

        let mut x_min = all_vertices[0].x;
        let mut x_max = all_vertices[0].x;
        let mut y_min = all_vertices[0].y;
        let mut y_max = all_vertices[0].y;
        let mut z_min = all_vertices[0].z;
        let mut z_max = all_vertices[0].z;

        for vertex in &all_vertices {
            x_min = x_min.min(vertex.x);
            x_max = x_max.max(vertex.x);
            y_min = y_min.min(vertex.y);
            y_max = y_max.max(vertex.y);
            z_min = z_min.min(vertex.z);
            z_max = z_max.max(vertex.z);
        }

        let mut merged_box = Box3D::empty_box();
        let cloud_points = vec![[x_min, y_min, z_min], [x_max, y_max, z_max]];
        merged_box.cloud2box(&cloud_points);

        merged_box
    }
}
