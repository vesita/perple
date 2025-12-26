use nalgebra::{Matrix4, Point3, Vector3, Vector4};
use redra::proto::shape::Point;

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
}

/// 2D目标检测结果
/// 
/// 表示一个检测到的对象，包括边界框、类别ID、类别名称和置信度。
#[derive(Debug, Clone, Default)]
pub struct Detection2D {
    pub the_box: Box2D,
    pub class_id: u32,
    pub class_name: String,
    pub confidence: f32,
}

impl Detection2D {
    /// 创建一个新的检测结果
    /// 
    /// # 参数
    /// * `the_box` - 边界框
    /// * `class_id` - 类别ID
    /// * `class_name` - 类别名称
    /// * `confidence` - 置信度
    pub fn new(the_box: Box2D, class_id: u32, class_name: String, confidence: f32) -> Self {
        Self { the_box, class_id, class_name, confidence }
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

    /// 从平移和欧拉角创建Tag3D
    /// 
    /// # 参数
    /// * `x, y, z` - 平移分量
    /// * `rx, ry, rz` - 欧拉角（弧度）
    /// * `length, width, height` - 包围盒尺寸
    /// 
    /// # 返回值
    /// 新创建的Tag3D实例
    pub fn from_position_and_angles(x: f32, y: f32, z: f32, rx: f32, ry: f32, rz: f32, length: f32, width: f32, height: f32) -> Self {
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
        local_point.x >= -self.length / 2.0 && local_point.x <= self.length / 2.0 &&
            local_point.y >= -self.width / 2.0 && local_point.y <= self.width / 2.0 &&
            local_point.z >= -self.height / 2.0 && local_point.z <= self.height / 2.0
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
        let new_center_local = Point3::new((x_min + x_max) / 2.0, (y_min + y_max) / 2.0, (z_min + z_max) / 2.0);
        
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

    /// 从点云创建包围盒
    pub fn cloud2box(&mut self, cloud3d: &Vec<[f32; 3]>) {
        if cloud3d.is_empty() {
            return;
        }
        
        // 简单的AABB实现 - 计算点云在世界坐标系中的边界
        let mut x_min = cloud3d[0][0];
        let mut x_max = cloud3d[0][0];
        let mut y_min = cloud3d[0][1];
        let mut y_max = cloud3d[0][1];
        let mut z_min = cloud3d[0][2];
        let mut z_max = cloud3d[0][2];
        
        for point in cloud3d {
            x_min = x_min.min(point[0]);
            x_max = x_max.max(point[0]);
            y_min = y_min.min(point[1]);
            y_max = y_max.max(point[1]);
            z_min = z_min.min(point[2]);
            z_max = z_max.max(point[2]);
        }
        
        // 计算中心点坐标
        let center_x = (x_min + x_max) / 2.0;
        let center_y = (y_min + y_max) / 2.0;
        let center_z = (z_min + z_max) / 2.0;
        
        // 设置包围盒的位姿矩阵（无旋转，仅平移）
        self.pose = Matrix4::new_translation(&Vector3::new(center_x, center_y, center_z));
        
        // 设置尺寸
        self.length = x_max - x_min;
        self.width = y_max - y_min;
        self.height = z_max - z_min;
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
    
    /// 获取包围盒的中心点
    pub fn center(&self) -> Point3<f32> {
        Point3::new(self.pose[(0, 3)], self.pose[(1, 3)], self.pose[(2, 3)])
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
    
    /// 检查点是否在包围盒附近（在一定距离内）
    pub fn near(&self, point: &[f32; 3], distance: f32) -> bool {
        // 首先通过中心点距离进行快速筛选
        let center = self.center();
        let center_distance_sq = (point[0] - center.x).powi(2) +
            (point[1] - center.y).powi(2) +
            (point[2] - center.z).powi(2);
        
        // 估算包围盒的最大半径（从中心到顶点的最大距离）
        let max_radius = (
            (self.length / 2.0).powi(2) +
            (self.width / 2.0).powi(2) +
            (self.height / 2.0).powi(2)
        ).sqrt();
        
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
        if inter_x_min >= inter_x_max || 
            inter_y_min >= inter_y_max || 
            inter_z_min >= inter_z_max {
            return 0.0;
        }

        // 计算交集体积
        let intersection_volume = (inter_x_max - inter_x_min) * 
                                (inter_y_max - inter_y_min) * 
                                (inter_z_max - inter_z_min);

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
        let cloud_points = vec![
            [x_min, y_min, z_min],
            [x_max, y_max, z_max],
        ];
        merged_box.cloud2box(&cloud_points);
        
        merged_box
    }
}

/// 3D边界框结构
/// 
/// 表示一个3D空间中的边界框，用于包围点云中的对象。
#[derive(Debug, Clone, Copy, Default)]
pub struct AABB3D {
    pub x_min: f32,
    pub x_max: f32,
    pub y_min: f32,
    pub y_max: f32,
    pub z_min: f32,
    pub z_max: f32,
}

impl AABB3D {
    /// 创建一个空的边界框
    pub fn empty_box() -> Self {
        Self {
            x_min: 0.0,
            x_max: 0.0,
            y_min: 0.0,
            y_max: 0.0,
            z_min: 0.0,
            z_max: 0.0,
        }
    }
    
    /// 创建一个新的边界框
    /// 
    /// # 参数
    /// * `x_min` - X轴最小值
    /// * `x_max` - X轴最大值
    /// * `y_min` - Y轴最小值
    /// * `y_max` - Y轴最大值
    /// * `z_min` - Z轴最小值
    /// * `z_max` - Z轴最大值
    pub fn new(x_min: f32, x_max: f32, y_min: f32, y_max: f32, z_min: f32, z_max: f32) -> Self {
        Self {
            x_min,
            x_max,
            y_min,
            y_max,
            z_min,
            z_max,
        }
    }
    
    pub fn contains(&self, point: [f32; 3]) -> bool {
        point[0] >= self.x_min && point[0] <= self.x_max &&
        point[1] >= self.y_min && point[1] <= self.y_max &&
        point[2] >= self.z_min && point[2] <= self.z_max
    }

    pub fn near(&self, point: &[f32; 3], distance: f32) -> bool {
        // 使用 clamp 找到点在边界框上的最近点
        let x = point[0].clamp(self.x_min, self.x_max);
        let y = point[1].clamp(self.y_min, self.y_max);
        let z = point[2].clamp(self.z_min, self.z_max);
        
        // 计算欧几里得距离的平方
        let dx = x - point[0];
        let dy = y - point[1];
        let dz = z - point[2];
        
        dx * dx + dy * dy + dz * dz <= distance * distance
    }

    pub fn expand(&mut self, point: &[f32; 3]) {
        self.x_min = self.x_min.min(point[0]);
        self.x_max = self.x_max.max(point[0]);
        self.y_min = self.y_min.min(point[1]);
        self.y_max = self.y_max.max(point[1]);
        self.z_min = self.z_min.min(point[2]);
        self.z_max = self.z_max.max(point[2]);
    }

    pub fn merge(&mut self, other: &Self) {
        self.x_min = self.x_min.min(other.x_min);
        self.x_max = self.x_max.max(other.x_max);
        self.y_min = self.y_min.min(other.y_min);
        self.y_max = self.y_max.max(other.y_max);
        self.z_min = self.z_min.min(other.z_min);
        self.z_max = self.z_max.max(other.z_max);
    }

    pub fn iou(&self, other: &Self) -> f32 {
        // 计算交集区域的边界
        let inter_x_min = self.x_min.max(other.x_min);
        let inter_x_max = self.x_max.min(other.x_max);
        let inter_y_min = self.y_min.max(other.y_min);
        let inter_y_max = self.y_max.min(other.y_max);
        let inter_z_min = self.z_min.max(other.z_min);
        let inter_z_max = self.z_max.min(other.z_max);

        // 检查是否有交集
        if inter_x_min >= inter_x_max || 
            inter_y_min >= inter_y_max || 
            inter_z_min >= inter_z_max {
            return 0.0;
        }

        // 计算交集体积
        let intersection_volume = (inter_x_max - inter_x_min) * 
                                (inter_y_max - inter_y_min) * 
                                (inter_z_max - inter_z_min);

        // 计算两个盒子的体积
        let self_volume = self.volume();
        let other_volume = other.volume();

        // 计算并集体积
        let union_volume = self_volume + other_volume - intersection_volume;

        // 返回交并比
        if union_volume == 0.0 {
            0.0
        } else {
            intersection_volume / union_volume
        }
    }

    pub fn volume(&self) -> f32 {
        (self.x_max - self.x_min) * 
        (self.y_max - self.y_min) * 
        (self.z_max - self.z_min)
    }
}

/// 3D目标检测结果
/// 
/// 表示一个检测到的对象，包括边界框、类别ID和类别名称。
#[derive(Clone)]
pub struct Detection3D {
    pub the_box: AABB3D,
    pub class_id: u32,
    pub class_name: String,
}

impl Detection3D {
    /// 创建一个新的检测结果
    pub fn new() -> Self {
        Self {
            the_box: AABB3D::empty_box(),
            class_id: 0,
            class_name: String::new(),
        }
    }
}

impl Default for Detection3D {
    fn default() -> Self {
        Self::new()
    }
}