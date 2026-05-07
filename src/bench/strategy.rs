use std::time::Duration;
use super::recorder::BenchRecorder;
use crate::cloud::ground::{GroundPickStrategy, create_ground_strategy};

// ── 预处理结果 ──────────────────────────────────────────

/// 单帧预处理结果。
///
/// 每个变体包裹上一层的数据并追加本层产出。
/// 新增下游策略时，扩展枚举即可。
pub enum Preprocessed {
    /// 无预处理，原始点云即输入。
    Passthrough,
    /// 地面提取完成，非地面点可用。
    Ground {
        non_ground: Vec<[f32; 3]>,
    },
}

// ── 预处理器 trait ──────────────────────────────────────

/// 预处理步骤，每帧执行一次，结果共享给所有候选策略。
pub trait Preprocessor {
    fn name(&self) -> &str;
    fn preprocess(&mut self, cloud: &[[f32; 3]]) -> Preprocessed;
}

/// 直通预处理器（零开销），ground_bench 使用。
pub struct PassthroughPreprocessor;

impl Preprocessor for PassthroughPreprocessor {
    fn name(&self) -> &str { "passthrough" }
    fn preprocess(&mut self, _cloud: &[[f32; 3]]) -> Preprocessed {
        Preprocessed::Passthrough
    }
}

/// 地面提取预处理器，cluster_bench 使用。
pub struct GroundPreprocessor {
    strategy: Box<dyn GroundPickStrategy>,
}

impl GroundPreprocessor {
    pub fn new(strategy: Box<dyn GroundPickStrategy>) -> Self {
        Self { strategy }
    }

    pub fn default() -> Self {
        Self::new(create_ground_strategy())
    }
}

impl Preprocessor for GroundPreprocessor {
    fn name(&self) -> &str { "ground" }
    fn preprocess(&mut self, cloud: &[[f32; 3]]) -> Preprocessed {
        let mut buf = cloud.to_vec();
        let (n_ground, _, _) = self.strategy.pick(&mut buf);
        Preprocessed::Ground {
            non_ground: buf[n_ground..].to_vec(),
        }
    }
}

// ── 帧数据 ──────────────────────────────────────────────

/// 单帧数据，由 BenchHarness 在预处理后构建。
pub struct FrameData<'a> {
    /// 原始点云（未排序，预处理不修改此数据）。
    pub cloud: &'a [[f32; 3]],
    /// 预处理结果。
    pub preprocessed: &'a Preprocessed,
    /// 帧序号。
    pub frame_idx: usize,
}

impl<'a> FrameData<'a> {
    /// 获取非地面点。
    ///
    /// - `Preprocessed::Ground` 时返回预处理产出的非地面点
    /// - `Preprocessed::Passthrough` 时返回原始点云
    pub fn non_ground(&self) -> &'a [[f32; 3]] {
        match self.preprocessed {
            Preprocessed::Ground { non_ground } => non_ground,
            Preprocessed::Passthrough => self.cloud,
        }
    }
}

// ── 候选策略 trait ──────────────────────────────────────

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
