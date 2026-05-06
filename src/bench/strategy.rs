use std::time::Duration;
use super::recorder::BenchRecorder;

/// 单帧数据，由 BenchHarness 在预处理后构建。
pub struct FrameData<'a> {
    /// 原始点云（未排序，地面策略测试用）。
    pub cloud: &'a [[f32; 3]],
    /// 预处理后的点云（默认地面策略排序后，ground 在前）。
    pub preprocessed: &'a [[f32; 3]],
    /// 非地面点子集（preprocessed[n_ground..]）。
    pub non_ground: &'a [[f32; 3]],
    /// 帧序号。
    pub frame_idx: usize,
}

/// 策略测试接口。
///
/// 每个 benchmark 策略实现此 trait，由 BenchHarness 驱动执行。
/// 策略自行管理统计累加和帧可视化逻辑。
pub trait BenchStrategy {
    /// 策略名称（用于日志和输出文件名）。
    fn name(&self) -> &str;

    /// 每帧执行候选策略，返回执行耗时。
    fn run(&mut self, frame: &FrameData) -> Duration;

    /// 将当前帧的检测结果写入 recorder。
    fn write_frame(&mut self, recorder: &mut BenchRecorder, frame: &FrameData);

    /// 输出汇总统计表。
    fn summarize(&self);
}
