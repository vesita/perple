//! Perple 模块的错误类型

use std::fmt;

/// Perple 模块的错误类型
#[derive(Debug)]
pub enum PerpleError {
    /// 循环控制相关错误
    LoopError(String),
    /// 线程锁中毒错误
    PoisonError(String),
    /// 其他错误
    Other(String),
}

impl fmt::Display for PerpleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PerpleError::LoopError(e) => write!(f, "循环控制错误: {}", e),
            PerpleError::PoisonError(e) => write!(f, "线程锁中毒错误: {}", e),
            PerpleError::Other(e) => write!(f, "其他错误: {}", e),
        }
    }
}

impl std::error::Error for PerpleError {}

impl<T> From<std::sync::PoisonError<T>> for PerpleError {
    fn from(error: std::sync::PoisonError<T>) -> Self {
        PerpleError::PoisonError(format!("线程锁中毒: {:?}", error))
    }
}
