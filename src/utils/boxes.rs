use nalgebra::{Matrix4, Point3, Vector3};

/// 2D边界框结构
///
/// 表示一个矩形边界框，用于包围检测到的目标。
#[derive(Debug, Clone, Default, Copy, PartialEq)]
pub struct Box2D {
    /// 左上角x坐标
    pub x1: f32,
    /// 左上角y坐标
    pub y1: f32,
    /// 右下角x坐标
    pub x2: f32,
    /// 右下角y坐标
    pub y2: f32,
}

impl Box2D {
    /// 创建一个新的边界框
    pub fn new(x1: f32, y1: f32, x2: f32, y2: f32) -> Self {
        Self { x1, y1, x2, y2 }
    }

    /// 计算边界框的宽度
    pub fn width(&self) -> f32 {
        (self.x2 - self.x1).abs()
    }

    /// 计算边界框的高度
    pub fn height(&self) -> f32 {
        (self.y2 - self.y1).abs()
    }

    /// 计算边界框的面积
    pub fn area(&self) -> f32 {
        self.width() * self.height()
    }

    /// 检查边界框是否有效（宽度和高度都大于0）
    pub fn is_valid(&self) -> bool {
        self.width() > 0.0 && self.height() > 0.0
    }

    /// 计算与另一个 2D 框的 IoU（交并比）
    pub fn iou(&self, other: &Self) -> f32 {
        let inter_x1 = self.x1.max(other.x1);
        let inter_y1 = self.y1.max(other.y1);
        let inter_x2 = self.x2.min(other.x2);
        let inter_y2 = self.y2.min(other.y2);

        let inter_w = (inter_x2 - inter_x1).max(0.0);
        let inter_h = (inter_y2 - inter_y1).max(0.0);
        let inter_area = inter_w * inter_h;

        let self_area = self.area();
        let other_area = other.area();
        let union_area = self_area + other_area - inter_area;

        if union_area <= 0.0 { 0.0 } else { inter_area / union_area }
    }
}

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
    ///
    /// # 参数
    /// * `point` - 要检查的世界坐标系中的点
    ///
    /// # 返回值
    /// 如果点在包围盒内则返回true，否则返回false
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
    ///
    /// # 参数
    /// * `point` - 要包含的世界坐标系中的点
    ///
    /// # 返回值
    /// 成功时返回Ok(())，失败时返回错误信息
    pub fn expand(&mut self, point: &[f32; 3]) -> Result<(), String> {
        // 如果包围盒为空，则初始化为一个点
        if self.length == 0.0 && self.width == 0.0 && self.height == 0.0 {
            let position = Vector3::new(point[0], point[1], point[2]);
            self.pose = Matrix4::new_translation(&position);
            return Ok(());
        }

        // 将点转换到包围盒的局部坐标系中
        let point_world = Point3::new(point[0], point[1], point[2]);
        let inv_pose = self.pose.try_inverse().ok_or("矩阵不可求逆".to_string())?;
        let point_local = inv_pose.transform_point(&point_world);

        // 更新局部坐标系中的边界
        let x_min = point_local.x.min(-self.length / 2.0);
        let x_max = point_local.x.max(self.length / 2.0);
        let y_min = point_local.y.min(-self.width / 2.0);
        let y_max = point_local.y.max(self.width / 2.0);
        let z_min = point_local.z.min(-self.height / 2.0);
        let z_max = point_local.z.max(self.height / 2.0);

        // 更新尺寸
        self.length = x_max - x_min;
        self.width = y_max - y_min;
        self.height = z_max - z_min;

        // 更新位置（在局部坐标系中）
        let new_center_local = Point3::new(
            (x_min + x_max) / 2.0,
            (y_min + y_max) / 2.0,
            (z_min + z_max) / 2.0,
        );

        // 将新的中心点转换回世界坐标系
        let new_center_world = self.pose.transform_point(&new_center_local);

        // 更新位姿矩阵的平移部分
        let mut new_pose = self.pose;
        new_pose[(0, 3)] = new_center_world.x;
        new_pose[(1, 3)] = new_center_world.y;
        new_pose[(2, 3)] = new_center_world.z;
        self.pose = new_pose;

        Ok(())
    }

    /// 从点云计算 AABB（轴对齐包围盒）
    ///
    /// 接受任意 slice，自动 clamp 最小边长到 `min_edge`（米），
    /// 避免退化为零体积盒导致渲染异常。
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

        // 设置包围盒的位姿矩阵（在XY平面上，Z为0）
        self.pose = Matrix4::new_translation(&Vector3::new(center_x, center_y, 0.0));

        // 设置尺寸
        self.length = x_max - x_min;
        self.width = y_max - y_min;
        self.height = f32::MAX; // 在Z方向上无限延伸
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

    // pub fn center_z_up(&self) -> Vec3 {
    //     Vec3::new(self.pose[(0, 3)], self.pose[(1, 3)], self.pose[(2, 3)])
    // }

    // pub fn shape_z_up(&self) -> Vec3 {
    //     Vec3::new(self.length, self.width, self.height)
    // }

    // pub fn center_y_up(&self) -> Vec3 {
    //     let center = self.center_z_up();
    //     Vec3::new(
    //         center.x,
    //         center.z,
    //         center.y,
    //     )
    // }

    // pub fn shape_y_up(&self) -> Vec3 {
    //     let shape = self.shape_z_up();
    //     Vec3::new(
    //         shape.x,
    //         shape.z,
    //         shape.y,
    //     )
    // }

    /// 计算点到包围盒中心的距离
    ///
    /// # 参数
    /// * `point` - 世界坐标系中的点
    ///
    /// # 返回值
    /// 点到包围盒中心的欧几里得距离
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
            // 下底面的边
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
            // 上顶面的边
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 4],
            // 垂直连接上下底面的边
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],
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
            // 下底面的边
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
            // 上顶面的边
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 4],
            // 垂直连接上下底面的边
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],
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
        // 首先通过中心点距离进行快速筛选
        let center = self.center();
        let center_distance_sq = (point[0] - center.x).powi(2)
            + (point[1] - center.y).powi(2)
            + (point[2] - center.z).powi(2);

        // 估算包围盒的最大半径（从中心到顶点的最大距离）
        let max_radius = ((self.length / 2.0).powi(2)
            + (self.width / 2.0).powi(2)
            + (self.height / 2.0).powi(2))
        .sqrt();

        // 如果点到中心的距离大于(最大半径+距离阈值)，则肯定不相交
        if center_distance_sq > (max_radius + distance).powi(2) {
            return false;
        }

        // 如果点到中心的距离小于等于(最大半径-距离阈值)，则肯定相交
        if max_radius >= distance && center_distance_sq <= (max_radius - distance).powi(2) {
            return true;
        }

        // 如果快速判断无法确定结果，则进行精确计算
        // 获取包围盒的AABB边界
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

        // 使用 clamp 找到点在边界框上的最近点
        let x = point[0].clamp(x_min, x_max);
        let y = point[1].clamp(y_min, y_max);
        let z = point[2].clamp(z_min, z_max);

        // 计算欧几里得距离的平方
        let dx = x - point[0];
        let dy = y - point[1];
        let dz = z - point[2];

        dx * dx + dy * dy + dz * dz <= distance * distance
    }

    /// 将世界坐标系中的点转换到包围盒的局部坐标系中
    ///
    /// # 参数
    /// * `point` - 世界坐标系中的点
    ///
    /// # 返回值
    /// 局部坐标系中的点
    ///
    /// # Panics
    /// 当位姿矩阵不可逆时会panic
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
    ///
    /// # 参数
    /// * `point` - 局部坐标系中的点
    ///
    /// # 返回值
    /// 世界坐标系中的点
    pub fn to_world(&self, point: &Point3<f32>) -> [f32; 3] {
        let world_point = self.pose.transform_point(point);
        [world_point.x, world_point.y, world_point.z]
    }

    /// 计算两个Box3D的交并比(IOU)
    ///
    /// # 参数
    /// * `other` - 另一个Box3D对象
    ///
    /// # 返回值
    /// 交并比值，范围在0.0到1.0之间
    pub fn iou(&self, other: &Self) -> f32 {
        // 获取两个包围盒的顶点以计算AABB边界
        let vertices1 = self.vertices();
        let vertices2 = other.vertices();

        // 计算第一个包围盒的AABB边界
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

        // 计算第二个包围盒的AABB边界
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

        // 计算交集区域的边界
        let inter_x_min = x_min1.max(x_min2);
        let inter_x_max = x_max1.min(x_max2);
        let inter_y_min = y_min1.max(y_min2);
        let inter_y_max = y_max1.min(y_max2);
        let inter_z_min = z_min1.max(z_min2);
        let inter_z_max = z_max1.min(z_max2);

        // 检查是否有交集
        if inter_x_min >= inter_x_max || inter_y_min >= inter_y_max || inter_z_min >= inter_z_max {
            return 0.0;
        }

        // 计算交集体积
        let intersection_volume =
            (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min) * (inter_z_max - inter_z_min);

        // 计算两个盒子的体积
        let volume1 = (x_max1 - x_min1) * (y_max1 - y_min1) * (z_max1 - z_min1);
        let volume2 = (x_max2 - x_min2) * (y_max2 - y_min2) * (z_max2 - z_min2);

        // 计算并集体积
        let union_volume = volume1 + volume2 - intersection_volume;

        // 返回交并比
        if union_volume == 0.0 {
            0.0
        } else {
            intersection_volume / union_volume
        }
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
            (-axes[0],  axes[0].dot(&c) + hl),   // +x 面（向内 -x）
            ( axes[0], -axes[0].dot(&c) + hl),   // -x 面（向内 +x）
            (-axes[1],  axes[1].dot(&c) + hw),   // +y 面（向内 -y）
            ( axes[1], -axes[1].dot(&c) + hw),   // -y 面（向内 +y）
            (-axes[2],  axes[2].dot(&c) + hh),   // +z 面（向内 -z）
            ( axes[2], -axes[2].dot(&c) + hh),   // -z 面（向内 +z）
        ]
    }

    /// 获取 12 个三角形（6 个面各 2 个），顶点为世界坐标，从外部看逆时针绕向
    fn triangles(&self) -> Vec<[Point3<f32>; 3]> {
        let hl = self.length / 2.0;
        let hw = self.width / 2.0;
        let hh = self.height / 2.0;
        // 局部顶点
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
        // 变换到世界坐标
        let v: Vec<Point3<f32>> = lv.iter().map(|p| self.pose.transform_point(p)).collect();
        vec![
            [v[4], v[5], v[6]], [v[4], v[6], v[7]], // +z
            [v[1], v[0], v[3]], [v[1], v[3], v[2]], // -z
            [v[1], v[2], v[6]], [v[1], v[6], v[5]], // +x
            [v[0], v[4], v[7]], [v[0], v[7], v[3]], // -x
            [v[3], v[7], v[6]], [v[3], v[6], v[2]], // +y
            [v[0], v[1], v[5]], [v[0], v[5], v[4]], // -y
        ]
    }

    /// 真 3D OBB 交并比（通过三角形网格裁剪计算交集体积）
    ///
    /// 使用 Sutherland-Hodgman 风格裁剪：
    /// 1. 将 `other` 的 12 个三角形依次裁剪到 `self` 的 6 个面内
    /// 2. 再裁剪到 `other` 的 6 个面内（保证交集位于两盒内）
    /// 3. 计算剩余三角形网格的封闭体积
    pub fn obb_iou(&self, other: &Self) -> f32 {
        let tri_b = other.triangles();
        let planes_a = self.face_planes();
        let planes_b = other.face_planes();

        let mut current: Vec<[Point3<f32>; 3]> = tri_b;
        let mut next = Vec::new();

        // 用 A 的 6 个面裁剪 B 的三角形
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

        // 再用 B 的 6 个面裁剪（保证交集位于 B 内）
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
    ///
    /// 将两个 OBB 的 8 个顶点投影到 XY 平面，取 2D 凸包，
    /// 用 Sutherland-Hodgman 计算交集多边形面积。
    /// 行人检测推荐使用 BEV IoU 而非 3D IoU（行人体积小，3D IoU 过于敏感）。
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
    ///
    /// # 参数
    /// * `other` - 另一个Box3D对象
    ///
    /// # 返回值
    /// 合并后的新Box3D对象
    pub fn merge(&self, other: &Self) -> Self {
        // 获取两个包围盒的所有顶点
        let vertices1 = self.vertices();
        let vertices2 = other.vertices();

        // 合并所有顶点
        let mut all_vertices = Vec::new();
        all_vertices.extend_from_slice(&vertices1);
        all_vertices.extend_from_slice(&vertices2);

        // 计算合并后的AABB边界
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

        // 创建一个新的Box3D来表示合并结果
        let mut merged_box = Box3D::empty_box();
        let cloud_points = vec![[x_min, y_min, z_min], [x_max, y_max, z_max]];
        merged_box.cloud2box(&cloud_points);

        merged_box
    }
}

// ─── OBB IoU 辅助函数 ─────────────────────────────────────────

/// 用一个半平面裁剪一个三角形，输出 0/1/2 个新三角形。
///
/// 半平面定义为 `normal·p + d ≥ 0`（内侧），normal 为向内法线。
fn clip_triangle_by_plane(
    tri: [Point3<f32>; 3],
    normal: &Vector3<f32>,
    d: f32,
    out: &mut Vec<[Point3<f32>; 3]>,
) {
    let dists = [
        normal.dot(&tri[0].coords) + d,
        normal.dot(&tri[1].coords) + d,
        normal.dot(&tri[2].coords) + d,
    ];

    let inside = [dists[0] >= -1e-9, dists[1] >= -1e-9, dists[2] >= -1e-9];
    let n_inside = inside.iter().filter(|&&x| x).count();

    match n_inside {
        3 => out.push(tri),
        0 => {}
        1 => {
            let i = inside.iter().position(|&x| x).unwrap();
            let i1 = (i + 1) % 3;
            let i2 = (i + 2) % 3;
            let p1 = intersect_edge(tri[i], tri[i1], dists[i], dists[i1]);
            let p2 = intersect_edge(tri[i], tri[i2], dists[i], dists[i2]);
            out.push([tri[i], p1, p2]);
        }
        2 => {
            let o = inside.iter().position(|&x| !x).unwrap();
            let i1 = (o + 1) % 3;
            let i2 = (o + 2) % 3;
            let p1 = intersect_edge(tri[i1], tri[o], dists[i1], dists[o]);
            let p2 = intersect_edge(tri[i2], tri[o], dists[i2], dists[o]);
            out.push([tri[i1], tri[i2], p1]);
            out.push([tri[i2], p2, p1]);
        }
        _ => unreachable!(),
    }
}

/// 计算线段上两点与半平面交点（内→外的插值参数）。
fn intersect_edge(
    inside: Point3<f32>,
    outside: Point3<f32>,
    d_inside: f32,
    d_outside: f32,
) -> Point3<f32> {
    let t = (d_inside / (d_inside - d_outside)).clamp(0.0, 1.0);
    inside + (outside - inside) * t
}

/// 计算封闭三角形网格体积（散度定理）。
fn triangle_mesh_volume(triangles: &[[Point3<f32>; 3]]) -> f32 {
    let mut volume = 0.0;
    for tri in triangles {
        let v0 = tri[0].coords;
        let v1 = tri[1].coords;
        let v2 = tri[2].coords;
        volume += v0.dot(&v1.cross(&v2));
    }
    (volume / 6.0).abs()
}

// ─── BEV IoU 2D 辅助函数 ────────────────────────────────────

/// 计算 2D 凸多边形面积（Shoelace 公式）
fn polygon_area_2d(poly: &[(f32, f32)]) -> f32 {
    let n = poly.len();
    if n < 3 {
        return 0.0;
    }
    let mut area = 0.0;
    for i in 0..n {
        let j = (i + 1) % n;
        area += poly[i].0 * poly[j].1;
        area -= poly[j].0 * poly[i].1;
    }
    area.abs() / 2.0
}

/// 2D 线段交点（参数 t，沿 ab 方向）
fn intersect_2d(
    a: (f32, f32), b: (f32, f32),
    c: (f32, f32), d: (f32, f32),
) -> (f32, f32) {
    let denom = (b.0 - a.0) * (d.1 - c.1) - (b.1 - a.1) * (d.0 - c.0);
    if denom.abs() < 1e-12 {
        return ((a.0 + b.0) / 2.0, (a.1 + b.1) / 2.0); // fallback
    }
    let t = ((c.0 - a.0) * (d.1 - c.1) - (c.1 - a.1) * (d.0 - c.0)) / denom;
    let t = t.clamp(0.0, 1.0);
    (a.0 + t * (b.0 - a.0), a.1 + t * (b.1 - a.1))
}

/// 2D Sutherland-Hodgman：用 clipping 多边形裁剪 subject 多边形（均为凸多边形，CCW）
fn clip_polygon_2d(subject: &[(f32, f32)], clipping: &[(f32, f32)]) -> Vec<(f32, f32)> {
    let mut output = subject.to_vec();
    if output.is_empty() {
        return output;
    }

    let n = clipping.len();
    for i in 0..n {
        if output.is_empty() {
            return output;
        }
        let input = output;
        output = Vec::new();

        let p1 = clipping[i];
        let p2 = clipping[(i + 1) % n];
        let edge_x = p2.0 - p1.0;
        let edge_y = p2.1 - p1.1;

        for j in 0..input.len() {
            let curr = input[j];
            let prev = input[(j + input.len() - 1) % input.len()];

            let curr_inside = edge_x * (curr.1 - p1.1) - edge_y * (curr.0 - p1.0) >= 0.0;
            let prev_inside = edge_x * (prev.1 - p1.1) - edge_y * (prev.0 - p1.0) >= 0.0;

            if curr_inside {
                if !prev_inside {
                    output.push(intersect_2d(prev, curr, p1, p2));
                }
                output.push(curr);
            } else if prev_inside {
                output.push(intersect_2d(prev, curr, p1, p2));
            }
        }
    }
    output
}

/// 2D 凸包（Monotone Chain / Andrew 算法）
fn convex_hull_2d(points: &[(f32, f32)]) -> Vec<(f32, f32)> {
    if points.len() <= 1 {
        return points.to_vec();
    }

    let mut pts = points.to_vec();
    pts.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    });

    // 下凸包
    let mut lower: Vec<(f32, f32)> = Vec::new();
    for &p in &pts {
        while lower.len() >= 2 {
            let a = lower[lower.len() - 2];
            let b = lower[lower.len() - 1];
            let cross = (b.0 - a.0) * (p.1 - a.1) - (b.1 - a.1) * (p.0 - a.0);
            if cross <= 0.0 {
                lower.pop();
            } else {
                break;
            }
        }
        lower.push(p);
    }

    // 上凸包
    let mut upper: Vec<(f32, f32)> = Vec::new();
    for &p in pts.iter().rev() {
        while upper.len() >= 2 {
            let a = upper[upper.len() - 2];
            let b = upper[upper.len() - 1];
            let cross = (b.0 - a.0) * (p.1 - a.1) - (b.1 - a.1) * (p.0 - a.0);
            if cross <= 0.0 {
                upper.pop();
            } else {
                break;
            }
        }
        upper.push(p);
    }

    lower.pop();
    upper.pop();
    lower.extend(upper);
    lower
}

