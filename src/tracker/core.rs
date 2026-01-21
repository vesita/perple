use std::collections::HashMap;

use log::info;
use ndarray::arr1;

use crate::{
    cloud::CldBud,
    swapl::global_swapl,
    tracker::{kalman::KalmanFilter, output::Target},
    utils::{
        sight::Sight,
        stream::{Eap, Stream, StreamError},
    },
};

/// Tracker模块的错误类型
#[derive(Debug)]
pub enum TrackerError {
    StreamError(StreamError),
    PoisonError(String),
    KalmanError(crate::tracker::kalman::KalmanError),
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

impl From<crate::tracker::kalman::KalmanError> for TrackerError {
    fn from(error: crate::tracker::kalman::KalmanError) -> Self {
        TrackerError::KalmanError(error)
    }
}

impl std::fmt::Display for TrackerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TrackerError::StreamError(e) => write!(f, "流错误: {}", e),
            TrackerError::PoisonError(e) => write!(f, "线程锁中毒: {}", e),
            TrackerError::KalmanError(e) => write!(f, "卡尔曼滤波错误: {}", e),
        }
    }
}

impl std::error::Error for TrackerError {}

/// 跟踪目标信息
struct TrackedObject {
    _id: usize,
    _kalman_filter: KalmanFilter,
    _class_type: String,
    _last_seen: std::time::SystemTime,
}

impl TrackedObject {
    pub fn new(id: usize, initial_position: (f64, f64, f64), class_type: String) -> Result<Self, crate::tracker::kalman::KalmanError> {
        let mut kf = KalmanFilter::new(6, 3)?; // 6维状态向量(x,y,z,vx,vy,vz)，3维测量向量(x,y,z)
        let initial_state = arr1(&[
            initial_position.0,
            initial_position.1,
            initial_position.2,
            0.0, // 初始vx
            0.0, // 初始vy
            0.0, // 初始vz
        ]);
        let initial_covariance = ndarray::Array2::eye(6) * 10.0; // 初始高不确定性
        kf.init(initial_state, initial_covariance, 0.1)?;
        
        Ok(Self {
            _id: id,
            _kalman_filter: kf,
            _class_type: class_type,
            _last_seen: std::time::SystemTime::now(),
        })
    }
}

pub struct Tracker {
    sight: Eap<Stream<Vec<Sight>>>,
    tar3d: Eap<Stream<Vec<CldBud>>>,
    target: Eap<Stream<Vec<Target>>>,
    next_id: usize,
    _tracked_objects: HashMap<usize, TrackedObject>, // 存储已跟踪的对象
    _max_disappeared: u32, // 对象在被认为消失之前可以丢失的最大帧数
}

impl Tracker {
    pub fn new() -> Self {
        let swapl = global_swapl();
        Self {
            sight: swapl.sights.clone(),
            tar3d: swapl.cld_objs.clone(),
            target: swapl.targets.clone(),
            next_id: 1,
            _tracked_objects: HashMap::new(),
            _max_disappeared: 5,
        }
    }

    pub async fn class_sync(&mut self) -> Result<(), TrackerError> {
        // 使用Stream的read方法直接获取数据并自动推进索引
        let tar3d_data = {
            let mut tar3d_guard = self.tar3d.lock().await;
            match tar3d_guard.read() {
                Some(data) => data,
                None => Vec::new(),
            }
        };

        // 使用Stream的read方法直接获取数据并自动推进索引
        let sight_data = {
            let mut sight_guard = self.sight.lock().await;
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
                        the_box: cld_bud.the_box.clone(),
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
            let mut target_guard = self.target.lock().await;
            target_guard.write(matched_targets)?;
        }

        Ok(())
    }

    /// 计算两个3D边界框之间的距离
    fn calculate_distance(
        box1: &crate::utils::boxes::Box3D,
        box2: &crate::utils::boxes::Box3D,
    ) -> f32 {
        // 获取两个包围盒的中心点
        let center1 = box1.center();
        let center2 = box2.center();

        // 计算中心点之间的欧几里得距离
        let dx = center1.x - center2.x;
        let dy = center1.y - center2.y;
        let dz = center1.z - center2.z;

        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// 更新跟踪目标，结合sight关联功能
    pub async fn run(&mut self) -> Result<(), TrackerError> {
        info!("跟踪模块启动");
        // 读取最新的3D目标检测结果
        let current_detections = {
            let mut tar3d_guard = self.tar3d.lock().await;
            match tar3d_guard.read() {
                Some(data) => data,
                None => Vec::new(),
            }
        };

        // 读取视线数据用于后续分类修正
        let sight_data = {
            let mut sight_guard = self.sight.lock().await;
            match sight_guard.read() {
                Some(data) => data,
                None => Vec::new(),
            }
        };

        // 如果没有检测结果，则不进行处理
        if current_detections.is_empty() {
            return Ok(());
        }

        // 读取之前的跟踪目标
        let previous_targets = {
            let mut target_guard = self.target.lock().await;
            match target_guard.read() {
                Some(data) => data,
                None => Vec::new(),
            }
        };

        // 第一步：基于历史数据进行目标跟踪
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
                if distance < best_distance && distance < 1.0 {
                    // 阈值设为1米
                    best_distance = distance;
                    best_match_index = Some(i);
                }
            }

            // 如果找到了匹配，则更新目标的位置和类型
            if let Some(index) = best_match_index {
                let detection = &current_detections[index];
                target.the_box = detection.the_box.clone();
                // 保留历史分类信息，稍后用视觉信息修正
                used_detections[index] = true;
                tracked_targets.push(target);
            }
            // 如果没找到匹配，暂时移除该目标（在实际应用中可能需要更复杂的消失处理）
        }

        // 为未匹配的检测创建新目标
        for (i, detection) in current_detections.iter().enumerate() {
            if !used_detections[i] {
                let new_target = Target {
                    the_box: detection.the_box.clone(),
                    class_type: detection.class_name.clone(),
                    id: self.next_id,
                };
                self.next_id += 1;
                tracked_targets.push(new_target);
            }
        }

        // 第二步：使用视觉信息对分类结果进行修正
        if !sight_data.is_empty() {
            // 对每个跟踪目标，使用视线数据进行分类修正
            for target in tracked_targets.iter_mut() {
                let mut matched = false;

                // 检查目标是否与任何视线相交
                for sight in &sight_data {
                    if sight.slab(&target.the_box) {
                        // 如果相交，将其分类修正为"person"
                        target.class_type = "person".to_string();
                        matched = true;
                        break;
                    }
                }

                // 如果没有匹配到视线，且原分类为空，则标记为"obstacle"
                if !matched && target.class_type.is_empty() {
                    target.class_type = "obstacle".to_string();
                }
            }
        }

        // 将更新后的跟踪结果写入输出流
        {
            let mut target_guard = self.target.lock().await;
            target_guard.write(tracked_targets)?;
        }

        Ok(())
    }
}
