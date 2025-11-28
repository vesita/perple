use std::sync::{Arc, Mutex};
use crate::{cloud::CldBud, tracker::target::Target, utils::{sight::Sight, stream::{Stream, StreamError}}};
use nalgebra::Vector3;

/// Tracker模块的错误类型
#[derive(Debug)]
pub enum TrackerError {
    StreamError(StreamError),
    PoisonError(String),
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

impl std::fmt::Display for TrackerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TrackerError::StreamError(e) => write!(f, "流错误: {}", e),
            TrackerError::PoisonError(e) => write!(f, "线程锁中毒: {}", e),
        }
    }
}

impl std::error::Error for TrackerError {}

pub struct Tracker {
    sight: Arc<Mutex<Stream<Vec<Sight>>>>,
    tar3d: Arc<Mutex<Stream<Vec<CldBud>>>>,
    target: Arc<Mutex<Stream<Vec<Target>>>>,
    next_id: usize,
}

impl Tracker {
    pub fn new(
        sight: Arc<Mutex<Stream<Vec<Sight>>>>,
        tar3d: Arc<Mutex<Stream<Vec<CldBud>>>>,
        output_stream: Arc<Mutex<Stream<Vec<Target>>>>,
    ) -> Self {
        Self {
            sight,
            tar3d,
            target: output_stream,
            next_id: 1,
        }
    }

    pub fn class_sync(&mut self) -> Result<(), TrackerError> {
        // 使用Stream的read方法直接获取数据并自动推进索引
        let tar3d_data = {
            let mut tar3d_guard = self.tar3d.lock()?;
            match tar3d_guard.read() {
                Some(data) => data,
                None => Vec::new(),
            }
        };
        
        // 使用Stream的read方法直接获取数据并自动推进索引
        let sight_data = {
            let mut sight_guard = self.sight.lock()?;
            match sight_guard.read() {
                Some(data) => data,
                None => Vec::new(),
            }
        };

        // 创建匹配结果容器
        let mut matched_targets = Vec::new();
        
        // 对每个视线与每个3D检测框进行匹配
        for sight in sight_data.iter() {
            for cld_bud in tar3d_data.iter() {
                // 使用slab算法检测视线与3D包围盒是否相交
                if sight.slab(&cld_bud.the_box) {
                    // 创建匹配的目标对象
                    let target = Target {
                        the_box: cld_bud.the_box,
                        class_type: cld_bud.class_name.clone(),
                        id: self.next_id,
                    };
                    
                    self.next_id += 1;
                    
                    // 将匹配的目标添加到结果中
                    matched_targets.push(target);
                }
            }
        }

        // 将匹配结果写入输出流
        {
            let mut target_guard = self.target.lock()?;
            target_guard.write(matched_targets)?;
        }

        Ok(())
    }
    
    /// 计算两个3D边界框之间的距离
    fn calculate_distance(box1: &crate::utils::boxes::Box3D, box2: &crate::utils::boxes::Box3D) -> f32 {
        // 计算两个包围盒中心点
        let center1 = Vector3::new(
            (box1.x_min + box1.x_max) / 2.0,
            (box1.y_min + box1.y_max) / 2.0,
            (box1.z_min + box1.z_max) / 2.0,
        );
        
        let center2 = Vector3::new(
            (box2.x_min + box2.x_max) / 2.0,
            (box2.y_min + box2.y_max) / 2.0,
            (box2.z_min + box2.z_max) / 2.0,
        );
        
        // 返回中心点之间的欧几里得距离
        (center1 - center2).norm()
    }
    
    /// 更新跟踪目标
    pub fn track(&mut self) -> Result<(), TrackerError> {
        // 读取最新的3D目标检测结果
        let current_detections = {
            let mut tar3d_guard = self.tar3d.lock()?;
            match tar3d_guard.read() {
                Some(data) => data,
                None => Vec::new(),
            }
        };
        
        // 读取之前的跟踪目标
        let previous_targets = {
            let mut target_guard = self.target.lock()?;
            match target_guard.read() {
                Some(data) => data,
                None => Vec::new(),
            }
        };
        
        // 创建新的跟踪目标列表
        let mut tracked_targets = Vec::new();
        let mut used_detections = vec![false; current_detections.len()];
        
        // 对于每一个之前的跟踪目标，尝试在当前检测中找到匹配
        for mut target in previous_targets {
            let mut best_match_index = None;
            let mut best_distance = f32::MAX;
            
            // 在当前检测中寻找最近的匹配
            for (i, detection) in current_detections.iter().enumerate() {
                if used_detections[i] {
                    continue;
                }
                
                let distance = Self::calculate_distance(&target.the_box, &detection.the_box);
                if distance < best_distance && distance < 1.0 { // 阈值设为1米
                    best_distance = distance;
                    best_match_index = Some(i);
                }
            }
            
            // 如果找到了匹配，则更新目标的位置和类型
            if let Some(index) = best_match_index {
                let detection = &current_detections[index];
                target.the_box = detection.the_box;
                target.class_type = detection.class_name.clone();
                used_detections[index] = true;
                tracked_targets.push(target);
            }
            // 如果没找到匹配，暂时移除该目标（在实际应用中可能需要更复杂的消失处理）
        }
        
        // 为未匹配的检测创建新目标
        for (i, detection) in current_detections.iter().enumerate() {
            if !used_detections[i] {
                let new_target = Target {
                    the_box: detection.the_box,
                    class_type: detection.class_name.clone(),
                    id: self.next_id,
                };
                self.next_id += 1;
                tracked_targets.push(new_target);
            }
        }
        
        // 将更新后的跟踪结果写入输出流
        {
            let mut target_guard = self.target.lock()?;
            target_guard.write(tracked_targets)?;
        }
        
        Ok(())
    }
}