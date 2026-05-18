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
