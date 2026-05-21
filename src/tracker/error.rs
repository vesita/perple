//! Tracker 模块的错误类型

use crate::utils::stream::StreamError;

/// Tracker 模块的错误类型
#[derive(Debug)]
pub enum TrackerError {
    StreamError(StreamError),
    PoisonError(String),
    KalmanError(adskalman::Error),
}

impl From<StreamError> for TrackerError {
    fn from(error: StreamError) -> Self {
        TrackerError::StreamError(error)
    }
}

impl<T> From<std::sync::PoisonError<T>> for TrackerError {
    fn from(_error: std::sync::PoisonError<T>) -> Self {
        TrackerError::PoisonError("线程锁中毒".to_string())
    }
}

impl From<adskalman::Error> for TrackerError {
    fn from(error: adskalman::Error) -> Self {
        TrackerError::KalmanError(error)
    }
}

impl std::fmt::Display for TrackerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TrackerError::StreamError(e) => write!(f, "流错误：{}", e),
            TrackerError::PoisonError(e) => write!(f, "线程锁中毒：{}", e),
            TrackerError::KalmanError(e) => write!(f, "卡尔曼滤波错误：{:?}", e),
        }
    }
}

impl std::error::Error for TrackerError {}
