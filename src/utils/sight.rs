
use nalgebra::Vector3;
use crate::utils::boxes::Box3D;

pub struct Sight {
    pub origin: Vector3<f32>,
    pub direction: Vector3<f32>,
}

impl Sight {
    pub fn new() -> Self {
        Sight {
            origin: Vector3::new(0.0, 0.0, 0.0),
            direction: Vector3::new(0.0, 0.0, 1.0),
        }
    }

    pub fn update(&mut self, origin: Vector3<f32>, direction: Vector3<f32>) {
        self.origin = origin;
        self.direction = direction;
    }
    
    /// 设置射线的起点和方向
    /// 
    /// # 参数
    /// * `origin` - 射线起点
    /// * `direction` - 射线方向向量
    pub fn set_ray(&mut self, origin: [f32; 3], direction: [f32; 3]) {
        self.origin = Vector3::from(origin);
        self.direction = Vector3::from(direction);
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

    /// 使用slab算法检测射线与3D包围盒是否相交
    /// 
    /// # 参数
    /// * `target` - 3D包围盒
    /// 
    /// # 返回值
    /// 如果相交返回true，否则返回false
    pub fn slab(&self, target: &Box3D) -> bool {
        let mut tmin = -f32::INFINITY;
        let mut tmax = f32::INFINITY;
        
        // 检查x轴
        if self.direction.x.abs() < f32::EPSILON {
            // 射线平行于x平面
            if self.origin.x < target.x_min || self.origin.x > target.x_max {
                return false;
            }
        } else {
            let inv_dir = 1.0 / self.direction.x;
            let t1 = (target.x_min - self.origin.x) * inv_dir;
            let t2 = (target.x_max - self.origin.x) * inv_dir;
            
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
            if self.origin.y < target.y_min || self.origin.y > target.y_max {
                return false;
            }
        } else {
            let inv_dir = 1.0 / self.direction.y;
            let t1 = (target.y_min - self.origin.y) * inv_dir;
            let t2 = (target.y_max - self.origin.y) * inv_dir;
            
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
            if self.origin.z < target.z_min || self.origin.z > target.z_max {
                return false;
            }
        } else {
            let inv_dir = 1.0 / self.direction.z;
            let t1 = (target.z_min - self.origin.z) * inv_dir;
            let t2 = (target.z_max - self.origin.z) * inv_dir;
            
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
        let center = Vector3::new(
            (target.x_min + target.x_max) / 2.0,
            (target.y_min + target.y_max) / 2.0,
            (target.z_min + target.z_max) / 2.0,
        );
        
        // 计算视线到中心点的距离
        let distance = self.distance_to_point(&[center.x, center.y, center.z]);
        
        // 如果距离小于阈值，则认为相交
        distance <= threshold
    }
}