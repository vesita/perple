use crate::lidar::{bounds::Box3D, tag::Tag3D};
use std::cmp::Ordering;

pub struct Claster {
    objects: Vec<Box3D>,
    patience: f32,
    merge_threshold: f32,
}

impl Claster { 
    pub fn new() -> Self {
        Claster {
            objects: Vec::new(),
            patience: 0.500,
            merge_threshold: 0.6,
        }
    }

    /// 获取聚类对象的不可变引用
    pub fn objects(&self) -> &Vec<Box3D> {
        &self.objects
    }

    pub fn submit(&mut self, point: &[f32; 3]) {
        self.objects.sort_by(|a, b| {
            // 计算点到两个边界框的距离
            let dist_a = Self::distance_to_box(point, a);
            let dist_b = Self::distance_to_box(point, b);
            
            // 使用partial_cmp进行浮点数比较，并反转结果实现升序排序，让距离近的排在后面
            dist_b.partial_cmp(&dist_a).unwrap_or(Ordering::Equal)
        });

        // 从距离最近的框（数组末尾）开始向前遍历
        let mut classed = false;
        for idx in (0..self.objects.len()).rev() {
            if self.objects[idx].near(point, self.patience) {
                self.objects[idx].expand(point);
                classed = true;
            } else {
                // 遇到第一个不near的框时，执行合并操作并退出循环
                self.merge_box();
                break;
            }
        }
        // 如果没有找到合适的box，则创建一个新的box
        if !classed {
            let box3d = Box3D::new(
                point[0] - self.patience, point[0] + self.patience,
                point[1] - self.patience, point[1] + self.patience,
                point[2] - self.patience, point[2] + self.patience,
            );
            self.objects.push(box3d);
        }
    }
    
    pub fn merge_box(&mut self) {    
        if let Some(mut current_box) = self.objects.pop() {
            for idx in (0..self.objects.len()).rev() {
                if current_box.iou(&self.objects[idx]) < self.merge_threshold {
                    break;
                }
                current_box.merge(&self.objects[idx]);
                // 从objects中移除已合并的框
                self.objects.remove(idx);
            }
            self.objects.push(current_box);
        }
    }
    
    
    // 计算点到边界框的最短欧几里得距离
    fn distance_to_box(point: &[f32; 3], box3d: &Box3D) -> f32 {
        // 找到点在边界框上的最近点
        let closest_x = point[0].max(box3d.x_min).min(box3d.x_max);
        let closest_y = point[1].max(box3d.y_min).min(box3d.y_max);
        let closest_z = point[2].max(box3d.z_min).min(box3d.z_max);
        
        // 计算欧几里得距离
        let dx = point[0] - closest_x;
        let dy = point[1] - closest_y;
        let dz = point[2] - closest_z;
        
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
    
    // 将Tag3D对象转换为Box3D对象并添加到聚类中
    pub fn add_tag3d(&mut self, tag: &Tag3D) {
        let box3d = Box3D::new(
            tag.x - tag.xl / 2.0,
            tag.x + tag.xl / 2.0,
            tag.y - tag.yl / 2.0,
            tag.y + tag.yl / 2.0,
            tag.z - tag.zl / 2.0,
            tag.z + tag.zl / 2.0,
        );
        self.objects.push(box3d);
    }
    
    // 添加Box3D对象到聚类中
    pub fn add_box3d(&mut self, box3d: Box3D) {
        self.objects.push(box3d);
    }
}