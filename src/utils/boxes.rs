use crate::config::DETECTIONS_CAPACITY;

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

/// 3D边界框结构
/// 
/// 表示一个3D空间中的边界框，用于包围点云中的对象。
#[derive(Debug, Clone, Copy, Default)]
pub struct Box3D {
    pub x_min: f32,
    pub x_max: f32,
    pub y_min: f32,
    pub y_max: f32,
    pub z_min: f32,
    pub z_max: f32,
}

impl Box3D {
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
    pub the_box: Box3D,
    pub class_id: u32,
    pub class_name: String,
}

impl Detection3D {
    /// 创建一个新的检测结果
    pub fn new() -> Self {
        Self {
            the_box: Box3D::empty_box(),
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