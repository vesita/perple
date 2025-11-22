    


pub struct Box3D {
    pub x_min: f32,
    pub x_max: f32,
    pub y_min: f32,
    pub y_max: f32,
    pub z_min: f32,
    pub z_max: f32,
    pub weight: usize,
}

impl Box3D {
    
    pub fn new(x_min: f32, x_max: f32, y_min: f32, y_max: f32, z_min: f32, z_max: f32) -> Self {
        Box3D {
            x_min,
            x_max,
            y_min,
            y_max,
            z_min,
            z_max,
            weight: 0,
        }
    }

    pub fn empty_box() -> Self {
        Box3D {
            x_min: 0.0,
            x_max: 0.0,
            y_min: 0.0,
            y_max: 0.0,
            z_min: 0.0,
            z_max: 0.0,
            weight: 0,
        }
    }

    pub fn contains(&self, point: [f32; 3]) -> bool {
        point[0] >= self.x_min && point[0] <= self.x_max &&
        point[1] >= self.y_min && point[1] <= self.y_max &&
        point[2] >= self.z_min && point[2] <= self.z_max
    }

    pub fn near(&self, point: &[f32; 3], distance: f32) -> bool {
        // 找到点在边界框上的最近点
        let closest_x = point[0].max(self.x_min).min(self.x_max);
        let closest_y = point[1].max(self.y_min).min(self.y_max);
        let closest_z = point[2].max(self.z_min).min(self.z_max);
        
        // 计算欧几里得距离
        let dx = point[0] - closest_x;
        let dy = point[1] - closest_y;
        let dz = point[2] - closest_z;
        
        (dx * dx + dy * dy + dz * dz) <= distance * distance
    }

    pub fn expand(&mut self, point: &[f32; 3]) {
        self.x_min = self.x_min.min(point[0]);
        self.x_max = self.x_max.max(point[0]);
        self.y_min = self.y_min.min(point[1]);
        self.y_max = self.y_max.max(point[1]);
        self.z_min = self.z_min.min(point[2]);
        self.z_max = self.z_max.max(point[2]);
        self.weight += 1;
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