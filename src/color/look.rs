use std::sync::{Arc, Mutex};

use nalgebra::{Matrix3, Matrix4, Vector2, Vector3};
use crate::{color::ClrBud, config::fixif, utils::{sight::Sight, stream::{Cream, Eap, Stream}}};

pub struct Look {
    pub cream: Cream<Vec<ClrBud>, Vec<Sight>>,
    pub intrinsic: Matrix3<f32>,
    pub extrinsic: Matrix4<f32>,
}

impl Default for Look {
    fn default() -> Self {
        // 从全局配置中获取相机参数
        let camera_config = &fixif().camera;
        
        // 将数组转换为矩阵
        let intrinsic = Matrix3::from_iterator(camera_config.intrinsic.iter().flatten().cloned());
        let extrinsic = Matrix4::from_iterator(camera_config.extrinsic.iter().flatten().cloned());

        Self {
            cream: Cream {
                in_stream: Arc::new(Mutex::new(Stream::new())),
                out_stream: Arc::new(Mutex::new(Stream::new())),
            },
            intrinsic,
            extrinsic,
        }
    }
}

impl Look {
    
    pub fn new(
        input_stream: Eap<Stream<Vec<ClrBud>>>,
        output_stream: Eap<Stream<Vec<Sight>>>,
    ) -> Self {
        // 从全局配置中获取相机参数
        let camera_config = &fixif().camera;
        
        // 将数组转换为矩阵
        let intrinsic = Matrix3::from_iterator(camera_config.intrinsic.iter().flatten().cloned());
        let extrinsic = Matrix4::from_iterator(camera_config.extrinsic.iter().flatten().cloned());

        Self {
            cream: Cream {
                in_stream: input_stream,
                out_stream: output_stream,
            },
            intrinsic,
            extrinsic,
        }
    }

    /// 将图像上的2D点转换为3D视线向量
    /// 
    /// 根据相机投影模型: 图像点 = 内参矩阵 * 外参矩阵 * 世界坐标点
    /// 反向计算视线向量
    /// 
    /// # 参数
    /// * `dot` - 图像上的2D点 (x, y) 像素坐标
    /// 
    /// # 返回值
    /// 返回视线的方向向量（在世界坐标系中）
    pub fn look_at(&self, dot: Vector2<f32>) -> Result<Vector3<f32>, Box<dyn std::error::Error>> {
        // 将像素坐标转换为相机坐标系下的归一化坐标
        // 这里使用齐次坐标进行变换
        let pixel_homogeneous = Vector3::new(dot.x, dot.y, 1.0);
        
        // 计算内参矩阵的逆矩阵
        let intrinsic_inv = self.intrinsic.try_inverse()
            .ok_or("内参矩阵不可逆")?;
        
        // 将像素坐标转换为相机坐标系下的方向向量
        let direction_camera = intrinsic_inv * pixel_homogeneous;
        
        // 将方向向量从相机坐标系转换到世界坐标系
        // 只需要旋转部分，不需要平移
        let rotation = self.extrinsic.fixed_view::<3, 3>(0, 0);
        let direction_world = rotation * direction_camera;
        
        Ok(direction_world.normalize())
    }

    /// 基于检测目标生成视线向量
    /// 
    /// # 参数
    /// * `target` - 检测目标
    /// 
    /// # 返回值
    /// 返回视线对象
    pub fn look_target(&self, target: &ClrBud) -> Sight {
        let the_box = target.the_box;
        
        // 计算检测框的中心点
        let center_x = (the_box.x1 + the_box.x2) / 2.0;
        let center_y = (the_box.y1 + the_box.y2) / 2.0;
        
        // 创建2D点
        let image_point = Vector2::new(center_x, center_y);
        
        // 转换为视线向量
        let direction = self.look_at(image_point)
            .unwrap_or_else(|_| Vector3::new(0.0, 0.0, 1.0)); // 出错时使用默认方向
        
        // 相机在世界坐标系中的位置（外参矩阵的平移部分）
        let camera_position = Vector3::new(
            self.extrinsic[(0, 3)],
            self.extrinsic[(1, 3)],
            self.extrinsic[(2, 3)]
        );
        
        // 创建视线对象
        let sight = Sight::new(camera_position, direction);
        sight
    }

    /// 自动化流式处理
    /// 
    /// 从输入流读取检测结果，为每个检测目标生成视线向量，
    /// 然后将结果写入输出流
    pub fn act(&mut self) {
        // 从输入流读取检测结果
        if let Some(detections) = self.cream.read() {
            // 为每个检测目标生成视线向量
            let sights: Vec<Sight> = detections
                .iter()
                .map(|detection| self.look_target(detection))
                .collect();
            
            // 将结果写入输出流
            if self.cream.write(sights).is_err() {
                eprintln!("写入视线向量到输出流失败");
            }
        }
    }

    pub fn set_by_angle(&mut self, tra: Vector3<f32>, rot: Vector3<f32>) {
        let rot_rad = Vector3::new(
            rot.x.to_radians(),
            rot.y.to_radians(),
            rot.z.to_radians(),
        );
        self.extrinsic = Matrix4::new_rotation(rot_rad) * Matrix4::new_translation(&tra);
    }

    pub fn set_by_radians(&mut self, tra: Vector3<f32>, rot: Vector3<f32>) {
        self.extrinsic = Matrix4::new_rotation(rot) * Matrix4::new_translation(&tra);
    }

    pub fn set_by_matrix(&mut self, matrix: Matrix4<f32>) {
        self.extrinsic = matrix;
    }
}