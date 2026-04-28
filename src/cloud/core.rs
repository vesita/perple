use std::fmt;
use std::sync::PoisonError;
use std::time::Instant;

use log::info;
use crate::cloud::classify::core::{Classify, ClassifyError};
use crate::swapl::global_swapl;
use crate::utils::stream::{Cream, StreamError};

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
}

impl Lidar {
    /// 创建 Lidar 实例
    ///
    /// 点云数据保持在 LiDAR 原生帧，不进行坐标变换。
    pub fn new() -> Self {
        let pool = global_swapl();
        info!("Lidar 模块初始化");
        Self {
            cream: Cream {
                in_stream: pool.clouds.clone(),
                out_stream: pool.clouds_out.clone(),
            },
            classify: Classify::new(),
        }
    }

    pub fn act(&mut self) -> Result<(), LidarError> {
        self.read_input()?;

        let start = Instant::now();
        let classify_result = self.classify.act();
        if let Err(e) = classify_result {
            eprintln!("点云分类错误：{:?}", e);
        }

        let elapsed = start.elapsed().as_millis();
        println!("点云处理耗时：{}ms", elapsed);
        Ok(())
    }

    pub fn read_input(&mut self) -> Result<(), LidarError> {
        let data = {
            let mut stream = self.cream.in_stream.blocking_lock();
            match stream.read() {
                Some(data) => data,
                None => return Err(LidarError::Other("没有数据".to_string())),
            }
        };

        {
            let mut stream = self.cream.out_stream.blocking_lock();
            stream.write(data)?;
        }
        Ok(())
    }
}
