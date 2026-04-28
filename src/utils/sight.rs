use crate::utils::boxes::Box3D;
use nalgebra::Vector3;

/// 视线结构体，用于表示从相机发出的视线射线
///
/// 该结构体包含视线的起点(origin)和方向(direction)
#[derive(Debug, Clone)]
pub struct Sight {
    /// 视线起点
    pub origin: Vector3<f32>,
    /// 视线方向（单位向量）
    pub direction: Vector3<f32>,
}

impl Sight {
    /// 创建一个新的视线对象
    ///
    /// # 参数
    /// * `origin` - 视线起点
    /// * `direction` - 视线方向
    ///
    /// # 返回值
    /// 返回一个新的Sight对象
    pub fn new(origin: Vector3<f32>, direction: Vector3<f32>) -> Self {
        Self { origin, direction }
    }

    /// 设置视线参数
    ///
    /// # 参数
    /// * `origin` - 视线起点
    /// * `direction` - 视线方向
    pub fn set_ray(&mut self, origin: Vector3<f32>, direction: Vector3<f32>) {
        self.origin = origin;
        self.direction = direction;
    }

    /// 计算射线上某一点到给定点的最短距离
    ///
    /// # 参数
    /// * `point` - 目标点坐标
    ///
    /// # 返回值
    /// 返回射线到目标点的最短距离
    pub fn distance_to_point(&self, point: &[f32; 3]) -> f32 {
        let target_point = Vector3::from(*point);

        // 射线上的点可以用公式表示: origin + t * direction
        // 计算射线上离目标点最近的点对应的参数t
        let oc = target_point - self.origin;

        // 计算方向向量的长度平方
        let dir_len_sq = self.direction.norm_squared();

        if dir_len_sq < f32::EPSILON {
            // 方向向量几乎为零向量
            return oc.norm();
        }

        // 计算参数t
        let t = oc.dot(&self.direction) / dir_len_sq;

        // 限制t >= 0，因为射线只能向前延伸
        let t = t.max(0.0);

        // 计算射线上最近点的坐标
        let closest_point = self.origin + t * self.direction;

        // 计算两点间距离
        let diff = target_point - closest_point;
        diff.norm()
    }

    /// 使用slab算法检测视线与3D包围盒是否相交
    ///
    /// # 参数
    /// * `target` - 3D包围盒
    ///
    /// # 返回值
    /// 如果相交返回true，否则返回false
    pub fn slab(&self, target: &Box3D) -> bool {
        // 获取包围盒的顶点
        let vertices = target.vertices();

        // 计算AABB边界
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

        let mut tmin = -f32::INFINITY;
        let mut tmax = f32::INFINITY;

        // 检查x轴
        if self.direction.x.abs() < f32::EPSILON {
            // 射线平行于x平面
            if self.origin.x < x_min || self.origin.x > x_max {
                return false;
            }
        } else {
            let inv_dir = 1.0 / self.direction.x;
            let t1 = (x_min - self.origin.x) * inv_dir;
            let t2 = (x_max - self.origin.x) * inv_dir;

            let (t_near, t_far) = if t1 > t2 { (t2, t1) } else { (t1, t2) };

            tmin = tmin.max(t_near);
            tmax = tmax.min(t_far);

            if tmin > tmax || tmax < 0.0 {
                return false;
            }
        }

        // 检查y轴
        if self.direction.y.abs() < f32::EPSILON {
            // 射线平行于y平面
            if self.origin.y < y_min || self.origin.y > y_max {
                return false;
            }
        } else {
            let inv_dir = 1.0 / self.direction.y;
            let t1 = (y_min - self.origin.y) * inv_dir;
            let t2 = (y_max - self.origin.y) * inv_dir;

            let (t_near, t_far) = if t1 > t2 { (t2, t1) } else { (t1, t2) };

            tmin = tmin.max(t_near);
            tmax = tmax.min(t_far);

            if tmin > tmax || tmax < 0.0 {
                return false;
            }
        }

        // 检查z轴
        if self.direction.z.abs() < f32::EPSILON {
            // 射线平行于z平面
            if self.origin.z < z_min || self.origin.z > z_max {
                return false;
            }
        } else {
            let inv_dir = 1.0 / self.direction.z;
            let t1 = (z_min - self.origin.z) * inv_dir;
            let t2 = (z_max - self.origin.z) * inv_dir;

            let (t_near, t_far) = if t1 > t2 { (t2, t1) } else { (t1, t2) };

            tmin = tmin.max(t_near);
            tmax = tmax.min(t_far);

            if tmin > tmax || tmax < 0.0 {
                return false;
            }
        }

        true
    }

    /// 通过计算视线到包围盒中心点的距离来模拟相交检测
    ///
    /// # 参数
    /// * `target` - 3D包围盒
    /// * `threshold` - 距离阈值，当视线到包围盒中心的距离小于此值时认为相交
    ///
    /// # 返回值
    /// 如果视线到包围盒中心的距离小于阈值返回true，否则返回false
    pub fn distance_based_intersection(&self, target: &Box3D, threshold: f32) -> bool {
        // 计算包围盒的中心点
        let center = target.center();

        // 计算视线到中心点的距离
        let to_center = center - self.origin;
        let projection_length = to_center.dot(&self.direction);

        // 如果投影长度为负，说明中心点在视线的反方向上
        if projection_length < 0.0 {
            return false;
        }

        // 计算最近点
        let closest_point = self.origin + self.direction * projection_length;

        // 计算距离
        let distance = (center - closest_point).magnitude();

        distance < threshold
    }
}
