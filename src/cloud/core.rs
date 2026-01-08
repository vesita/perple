use std::fmt;
use std::sync::PoisonError;
use std::time::Instant;

use log::info;
use nalgebra::{Matrix4, Vector3, Vector4};

use crate::cloud::classify::core::{Classify, ClassifyError};
use crate::config::fixif;
use crate::swapl::global_swapl;
use crate::utils::stream::{Cream, StreamError};
use crate::utils::world::OnWorld;

/// Lidar模块的错误类型
#[derive(Debug)]
pub enum LidarError {
    /// 流缓冲区相关错误
    StreamError(StreamError),
    /// 线程锁中毒错误
    PoisonError(String),
    /// 提交写入操作错误
    CommitError(String),
    /// 分类错误
    ClassifyError(String),
    /// 其他错误
    Other(String),
}

impl fmt::Display for LidarError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LidarError::StreamError(e) => write!(f, "流错误: {}", e),
            LidarError::PoisonError(e) => write!(f, "线程锁中毒错误: {}", e),
            LidarError::CommitError(e) => write!(f, "提交写入操作错误: {}", e),
            LidarError::ClassifyError(e) => write!(f, "分类错误: {}", e),
            LidarError::Other(e) => write!(f, "其他错误: {}", e),
        }
    }
}

impl std::error::Error for LidarError {}

impl From<StreamError> for LidarError {
    fn from(error: StreamError) -> Self {
        LidarError::StreamError(error)
    }
}

impl<T> From<PoisonError<T>> for LidarError {
    fn from(error: PoisonError<T>) -> Self {
        LidarError::PoisonError(format!("线程锁中毒: {:?}", error))
    }
}

impl From<ClassifyError> for LidarError {
    fn from(error: ClassifyError) -> Self {
        match error {
            ClassifyError::Error => LidarError::ClassifyError("分类器错误".to_string()),
        }
    }
}

pub struct Lidar {
    cream: Cream<Vec<[f32; 3]>, Vec<[f32; 3]>>,
    classify: Classify,
    extrinsic: Matrix4<f32>,
}

impl Lidar {
    /// 创建Lidar实例，通过全局Swapl数据中枢进行数据交互
    ///
    /// 所有数据交互都通过全局Swapl完成，实现了模块间的松耦合设计。
    /// Lidar模块内部保留指向各模块的指针，但不再需要外部传入数据流引用
    pub fn new() -> Self {
        // 获取全局数据交换中枢
        let pool = global_swapl();

        // 从全局配置中获取lidar外参
        let lidar_config = &fixif().lidar;

        // 将数组转换为矩阵
        let extrinsic = lidar_config.extrinsic.clone();

        Self {
            cream: Cream {
                in_stream: pool.clouds.clone(),
                out_stream: pool.cloud_in_world.clone(),
            },
            classify: Classify::new(),
            extrinsic: extrinsic.into(),
        }
    }

    pub async fn act(&mut self) -> Result<(), LidarError> {
        info!("Lidar模块启动");
        // 先读取并处理输入数据
        self.read_input().await?;

        // 创建一个计时器
        let start = Instant::now();
        // 使用分类器处理数据
        self.classify.act().await?;

        // 计算处理时间
        let elapsed = start.elapsed().as_millis();
        println!("点云处理耗时：{}ms", elapsed);
        Ok(())
    }

    pub async fn read_input(&mut self) -> Result<(), LidarError> {
        if let Some(mut data) = self.cream.read().await {
            for point in &mut data {
                // 使用转换矩阵将点从雷达坐标系转换到世界坐标系
                let point_vec = Vector4::new(point[0], point[1], point[2], 1.0);
                let point_world = self.extrinsic * point_vec;
                point[0] = point_world.x;
                point[1] = point_world.y;
                point[2] = point_world.z;
            }

            // 写入处理后的数据到输出流
            self.cream.write(data).await?;
            Ok(())
        } else {
            return Err(LidarError::Other("没有数据".to_string()));
        }
    }
}

impl OnWorld for Lidar {
    fn on_world(&self) -> Matrix4<f32> {
        self.extrinsic
    }

    fn set_by_angle(&mut self, tra: Vector3<f32>, rot: Vector3<f32>) {
        let rot_rad = Vector3::new(rot.x.to_radians(), rot.y.to_radians(), rot.z.to_radians());
        self.extrinsic = Matrix4::new_rotation(rot_rad) * Matrix4::new_translation(&tra);
    }

    fn set_by_radian(&mut self, tra: Vector3<f32>, rot: Vector3<f32>) {
        self.extrinsic = Matrix4::new_rotation(rot) * Matrix4::new_translation(&tra);
    }

    fn set_by_matrix(&mut self, matrix: &Matrix4<f32>) {
        self.extrinsic = *matrix;
    }
}
