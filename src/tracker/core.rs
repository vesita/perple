use std::collections::HashMap;
use std::time::SystemTime;

use nalgebra::Vector3;

use crate::{
    cloud::CldBud,
    swapl::global_swapl,
    tracker::{
        kalman::{KalmanConfig, KalmanFilterWrapper},
        output::Target,
    },
    utils::{
        boxes::Box3D,
        sight::Sight,
        stream::{Eap, Stream, StreamError},
    },
};

/// Tracker 模块的错误类型
#[derive(Debug)]
pub enum TrackerError {
    StreamError(StreamError),
    PoisonError(String),
    KalmanError(adskalman::Error),
    AssociationError(String),
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
            TrackerError::KalmanError(e) => write!(f, "卡尔曼滤波错误：{}", e),
            TrackerError::AssociationError(e) => write!(f, "数据关联错误：{}", e),
        }
    }
}

impl std::error::Error for TrackerError {}

/// 跟踪目标信息（包含卡尔曼滤波器）
struct TrackedObject {
    /// 目标唯一 ID（保留用于未来扩展）
    #[allow(dead_code)]
    id: usize,
    /// 目标类型
    class_type: String,
    /// 最后可见时间
    last_seen: SystemTime,
    /// 连续丢失的帧数
    disappeared_count: u32,
    /// 置信度
    confidence: f32,
    /// 卡尔曼滤波器
    kalman_filter: KalmanFilterWrapper,
}

impl TrackedObject {
    /// 创建新的跟踪对象实例
    /// 
    /// # 参数
    /// * `id` - 目标唯一标识符
    /// * `initial_box` - 初始 3D 边界框
    /// * `class_type` - 目标类型
    /// * `confidence` - 置信度
    /// 
    /// # 返回值
    /// 返回初始化后的 TrackedObject 实例，包含配置好的卡尔曼滤波器
    fn new(
        id: usize,
        initial_box: &Box3D,
        class_type: String,
        confidence: f32,
    ) -> Result<Self, adskalman::Error> {
        let mut kalman_filter = KalmanFilterWrapper::new(KalmanConfig::default())?;
        
        // 从边界盒中心提取初始位置
        let center = initial_box.center();
        let position = Vector3::new(center.x as f64, center.y as f64, center.z as f64);
        
        // 初始化卡尔曼滤波器状态
        kalman_filter.init_with_state(position, None);
        
        Ok(Self {
            id,
            class_type,
            last_seen: SystemTime::now(),
            disappeared_count: 0,
            confidence,
            kalman_filter,
        })
    }
    
    /// 更新目标状态
    /// 
    /// # 参数
    /// * `new_box` - 新的 3D 边界框
    /// * `new_class_type` - 新的目标类型
    /// * `new_confidence` - 新的置信度
    fn update(
        &mut self,
        new_box: &Box3D,
        new_class_type: String,
        new_confidence: f32,
    ) -> Result<(), adskalman::Error> {
        let center = new_box.center();
        let measurement = Vector3::new(center.x as f64, center.y as f64, center.z as f64);
        
        // 使用卡尔曼滤波更新状态
        self.kalman_filter.update(measurement)?;
        
        // 更新其他属性
        self.class_type = new_class_type;
        self.confidence = new_confidence;
        self.last_seen = SystemTime::now();
        self.disappeared_count = 0;
        
        Ok(())
    }
    
    /// 预测目标的下一位置
    fn predict(&mut self) -> Result<(), adskalman::Error> {
        self.kalman_filter.predict()?;
        self.disappeared_count += 1;
        Ok(())
    }
    
    /// 获取预测的位置
    fn get_predicted_position(&self) -> (f32, f32, f32) {
        let pos = self.kalman_filter.get_position();
        (pos.x as f32, pos.y as f32, pos.z as f32)
    }
    
    /// 检查目标是否应该被移除
    fn is_permanently_lost(&self, max_disappeared: u32) -> bool {
        self.disappeared_count >= max_disappeared
    }
}

/// 主跟踪器类
/// 
/// 负责多目标跟踪，包括：
/// - 数据关联（将检测与现有轨迹匹配）
/// - 状态估计（使用卡尔曼滤波）
/// - 轨迹管理（创建、更新、删除轨迹）
/// 
/// # 示例
/// ```rust,no_run
/// use crate::tracker::core::Tracker;
/// 
/// let mut tracker = Tracker::new();
/// // 设置自定义参数
/// // tracker.set_association_threshold(2.0);
/// // tracker.set_max_disappeared(10);
/// 
/// // 在主循环中调用
/// loop {
///     if let Err(e) = tracker.run() {
///         eprintln!("跟踪器运行错误：{}", e);
///     }
/// }
/// ```
pub struct Tracker {
    /// 视线数据流
    sight: Eap<Stream<Vec<Sight>>>,
    /// 3D 检测结果流
    tar3d: Eap<Stream<Vec<CldBud>>>,
    /// 跟踪输出流
    target: Eap<Stream<Vec<Target>>>,
    /// 下一个可用的目标 ID
    next_id: usize,
    /// 已跟踪的目标集合
    tracked_objects: HashMap<usize, TrackedObject>,
    /// 目标在被认为永久消失前可以丢失的最大帧数
    max_disappeared: u32,
    /// 匹配距离阈值（米）
    association_threshold: f32,
    /// 最小检测置信度
    min_confidence: f32,
}

impl Tracker {
    /// 创建新的跟踪器实例
    pub fn new() -> Self {
        let swapl = global_swapl();
        Self {
            sight: swapl.sights.clone(),
            tar3d: swapl.cld_objs.clone(),
            target: swapl.targets.clone(),
            next_id: 1,
            tracked_objects: HashMap::new(),
            max_disappeared: 5,           // 5 帧未出现则认为消失
            association_threshold: 1.5,   // 1.5 米匹配阈值
            min_confidence: 0.3,          // 最小置信度 0.3
        }
    }

    /// 设置匹配距离阈值（米）
    pub fn set_association_threshold(&mut self, threshold: f32) {
        self.association_threshold = threshold;
    }

    /// 设置最大消失帧数
    pub fn set_max_disappeared(&mut self, max: u32) {
        self.max_disappeared = max;
    }

    /// 设置最小检测置信度
    pub fn set_min_confidence(&mut self, confidence: f32) {
        self.min_confidence = confidence;
    }

    /// 计算两个 3D 边界框之间的距离（保留用于未来扩展）
    #[allow(dead_code)]
    fn calculate_distance(box1: &Box3D, box2: &Box3D) -> f32 {
        let center1 = box1.center();
        let center2 = box2.center();

        let dx = center1.x - center2.x;
        let dy = center1.y - center2.y;
        let dz = center1.z - center2.z;

        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// 匈牙利算法进行数据关联（简化版本）
    /// 
    /// # 参数
    /// * `predictions` - 预测位置列表
    /// * `detections` - 检测位置列表
    /// 
    /// # 返回值
    /// 返回匹配对 (prediction_index, detection_index) 和未匹配的 detections
    fn associate(
        predictions: &[(f32, f32, f32)],
        detections: &[CldBud],
        threshold: f32,
    ) -> (Vec<(usize, usize)>, Vec<usize>) {
        let mut matches = Vec::new();
        let mut used_detections = vec![false; detections.len()];
        let mut used_predictions = vec![false; predictions.len()];

        // 贪心匹配（可以用匈牙利算法优化）
        for (pred_idx, pred_pos) in predictions.iter().enumerate() {
            let mut best_match_idx = None;
            let mut best_distance = f32::MAX;

            for (det_idx, detection) in detections.iter().enumerate() {
                if used_detections[det_idx] {
                    continue;
                }

                let det_center = detection.the_box.center();
                let distance = ((pred_pos.0 - det_center.x).powi(2)
                    + (pred_pos.1 - det_center.y).powi(2)
                    + (pred_pos.2 - det_center.z).powi(2))
                .sqrt();

                if distance < best_distance && distance < threshold {
                    best_distance = distance;
                    best_match_idx = Some(det_idx);
                }
            }

            if let Some(match_idx) = best_match_idx {
                matches.push((pred_idx, match_idx));
                used_predictions[pred_idx] = true;
                used_detections[match_idx] = true;
            }
        }

        // 收集未匹配的 detections
        let unmatched_detections: Vec<usize> = used_detections
            .iter()
            .enumerate()
            .filter(|&(_, &used)| !used)
            .map(|(idx, _)| idx)
            .collect();

        (matches, unmatched_detections)
    }

    /// 使用视觉信息修正分类
    /// 
    /// # 参数
    /// * `target` - 要修正的目标
    /// * `sight_data` - 视线数据
    fn refine_classification_with_sight(
        target: &mut Target,
        sight_data: &[Sight],
    ) {
        for sight in sight_data {
            if sight.slab(&target.the_box) {
                // 如果与视线相交，分类为"person"
                target.class_type = "person".to_string();
                return;
            }
        }

        // 如果没有匹配到视线且原分类为空，标记为"obstacle"
        if target.class_type.is_empty() {
            target.class_type = "obstacle".to_string();
        }
    }

    /// 主跟踪循环
    /// 
    /// 执行以下步骤：
    /// 1. 读取最新检测结果
    /// 2. 对所有活跃轨迹进行预测
    /// 3. 数据关联（匹配预测与检测）
    /// 4. 更新匹配轨迹的状态
    /// 5. 创建新轨迹
    /// 6. 处理消失的轨迹
    /// 7. 使用视觉信息修正分类
    /// 8. 输出跟踪结果
    pub fn run(&mut self) -> Result<(), TrackerError> {
        // 读取最新的 3D 目标检测结果
        let current_detections = {
            let mut tar3d_guard = self.tar3d.blocking_lock();
            match tar3d_guard.read() {
                Some(data) => data.into_iter()
                    .filter(|d| d.confidence >= self.min_confidence)
                    .collect::<Vec<_>>(),
                None => Vec::new(),
            }
        };

        // 读取视线数据用于分类修正
        let sight_data = {
            let mut sight_guard = self.sight.blocking_lock();
            match sight_guard.read() {
                Some(data) => data,
                None => Vec::new(),
            }
        };

        // 步骤 1: 对所有现有轨迹进行预测
        for tracked_obj in self.tracked_objects.values_mut() {
            tracked_obj.predict()?;
        }

        // 步骤 2: 准备预测位置用于关联
        let predictions: Vec<(f32, f32, f32)> = self.tracked_objects
            .values()
            .map(|obj| obj.get_predicted_position())
            .collect();

        // 步骤 3: 数据关联
        let (matches, unmatched_detections) = Self::associate(
            &predictions,
            &current_detections,
            self.association_threshold,
        );

        // 步骤 4: 更新匹配的轨迹
        let mut tracked_ids: Vec<usize> = Vec::new();
        for (pred_idx, det_idx) in matches {
            let tracked_id: usize = self.tracked_objects.keys().nth(pred_idx).copied().unwrap();
            let detection = &current_detections[det_idx];
            
            if let Some(tracked_obj) = self.tracked_objects.get_mut(&tracked_id) {
                tracked_obj.update(
                    &detection.the_box,
                    detection.class_name.clone(),
                    detection.confidence,
                )?;
                
                tracked_ids.push(tracked_id);
            }
        }

        // 步骤 5: 为未匹配的检测创建新轨迹
        for det_idx in unmatched_detections {
            let detection = &current_detections[det_idx];
            let new_id = self.next_id;
            self.next_id += 1;

            match TrackedObject::new(
                new_id,
                &detection.the_box,
                detection.class_name.clone(),
                detection.confidence,
            ) {
                Ok(new_object) => {
                    self.tracked_objects.insert(new_id, new_object);
                    tracked_ids.push(new_id);
                }
                Err(e) => {
                    eprintln!("创建新跟踪对象失败：{}", e);
                }
            }
        }

        // 步骤 6: 移除永久消失的轨迹
        self.tracked_objects.retain(|id, obj| {
            if obj.is_permanently_lost(self.max_disappeared) {
                eprintln!("移除消失的目标 ID: {}, 丢失帧数：{}", id, obj.disappeared_count);
                false
            } else {
                true
            }
        });

        // 步骤 7: 生成输出并应用视觉分类修正
        let mut output_targets = Vec::new();
        for tracked_id in tracked_ids {
            if let Some(tracked_obj) = self.tracked_objects.get(&tracked_id) {
                // 获取预测位置
                let pos = tracked_obj.kalman_filter.get_position();
                
                // 使用原始检测的尺寸（这里取第一个匹配的检测或使用默认值）
                let default_box = Box3D::empty_box();
                let reference_box = current_detections
                    .iter()
                    .find(|d| {
                        let center = d.the_box.center();
                        ((center.x as f64 - pos.x).powi(2) 
                            + (center.y as f64 - pos.y).powi(2) 
                            + (center.z as f64 - pos.z).powi(2)).sqrt() < 1.0
                    })
                    .map(|d| &d.the_box)
                    .unwrap_or(&default_box);
                
                // 创建新的 Box3D，使用预测的位置和参考尺寸
                let mut predicted_box = Box3D::from_position_and_angles(
                    pos.x as f32,
                    pos.y as f32,
                    pos.z as f32,
                    0.0, 0.0, 0.0,  // 无旋转
                    reference_box.length,
                    reference_box.width,
                    reference_box.height,
                );
                predicted_box.pose = reference_box.pose; // 保留原始朝向
                
                let mut target = Target {
                    the_box: predicted_box,
                    class_type: tracked_obj.class_type.clone(),
                    id: tracked_id,
                };

                // 使用视觉信息修正分类
                Self::refine_classification_with_sight(&mut target, &sight_data);

                output_targets.push(target);
            }
        }

        // 步骤 8: 写入输出流
        {
            let mut target_guard = self.target.blocking_lock();
            target_guard.write(output_targets)?;
        }

        Ok(())
    }

    /// 获取当前跟踪的目标数量
    pub fn get_tracking_count(&self) -> usize {
        self.tracked_objects.len()
    }

    /// 获取所有跟踪的目标 ID
    pub fn get_tracked_ids(&self) -> Vec<usize> {
        self.tracked_objects.keys().copied().collect()
    }

    /// 清除所有跟踪目标
    pub fn clear(&mut self) {
        self.tracked_objects.clear();
    }
}

impl Default for Tracker {
    fn default() -> Self {
        Self::new()
    }
}
