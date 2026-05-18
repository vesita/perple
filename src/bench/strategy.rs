use std::time::Duration;
use serde::Serialize;
use super::recorder::BenchRecorder;
use crate::cloud::ground::{GroundPickStrategy, create_ground_strategy};
use crate::cloud::wall::WallPickStrategy;
use crate::cloud::wall::BevLsd;
use crate::cloud::ground::PeakScan;
use crate::cloud::denoise::{DenoiseStrategy, RadiusOutlierRemoval};

// ── 预处理结果 ──────────────────────────────────────────

/// 单帧预处理结果。
///
/// 每个变体包裹上一层的数据并追加本层产出。
/// 新增下游策略时，扩展枚举即可。
pub enum Preprocessed {
    /// 无预处理，原始点云即输入。
    Passthrough,
    /// 地面检测完成，非地面点可用。
    Ground {
        non_ground: Vec<[f32; 3]>,
    },
    /// 地面+降噪完成，墙体策略从此接管自有降噪逻辑。
    GroundDenoised {
        non_ground: Vec<[f32; 3]>,
    },
    /// 地面+墙面检测完成，非地面和非墙面点均可用。
    Wall {
        non_ground: Vec<[f32; 3]>,
        non_wall: Vec<[f32; 3]>,
    },
    /// 地面+墙面+降噪检测完成，三种数据均可用。
    Denoise {
        non_ground: Vec<[f32; 3]>,
        non_wall: Vec<[f32; 3]>,
        denoised: Vec<[f32; 3]>,
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

/// 地面检测预处理器，cluster_bench 使用。
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

/// 地面+降噪预处理器（无墙体检测），wall_bench 使用。
/// 管线：地面 → 降噪，墙体策略自行处理自有降噪逻辑。
pub struct GroundDenoisePreprocessor {
    ground: Box<dyn GroundPickStrategy>,
    denoise: Box<dyn DenoiseStrategy>,
}

impl GroundDenoisePreprocessor {
    pub fn new(
        ground: Box<dyn GroundPickStrategy>,
        denoise: Box<dyn DenoiseStrategy>,
    ) -> Self {
        Self { ground, denoise }
    }

    pub fn default() -> Self {
        Self::new(
            Box::new(crate::cloud::ground::PeakScan::new()),
            Box::new(RadiusOutlierRemoval::new(0.30, 3)),
        )
    }
}

impl Preprocessor for GroundDenoisePreprocessor {
    fn name(&self) -> &str { "ground+denoise" }
    fn preprocess(&mut self, cloud: &[[f32; 3]]) -> Preprocessed {
        let mut buf = cloud.to_vec();
        let (n_ground, _, _) = self.ground.pick(&mut buf);
        let non_ground = buf[n_ground..].to_vec();
        let (denoised, _) = self.denoise.denoise(&non_ground);
        Preprocessed::GroundDenoised { non_ground: denoised }
    }
}

/// 地面+墙面（+预处理降噪）检测预处理器，wall_bench 使用。
/// 管线：地面 → 降噪（预处理，改善墙体 BFS 连通性）→ 墙体
pub struct WallPreprocessor {
    ground: Box<dyn GroundPickStrategy>,
    denoise: Box<dyn DenoiseStrategy>,
    wall: Box<dyn WallPickStrategy>,
}

impl WallPreprocessor {
    pub fn new(
        ground: Box<dyn GroundPickStrategy>,
        denoise: Box<dyn DenoiseStrategy>,
        wall: Box<dyn WallPickStrategy>,
    ) -> Self {
        Self { ground, denoise, wall }
    }

    pub fn default() -> Self {
        Self::new(
            Box::new(PeakScan::new()),
            Box::new(RadiusOutlierRemoval::new(0.30, 3)),
            Box::new(BevLsd::with_params(0.08, 20)),
        )
    }
}

impl Preprocessor for WallPreprocessor {
    fn name(&self) -> &str { "ground+denoise+wall" }
    fn preprocess(&mut self, cloud: &[[f32; 3]]) -> Preprocessed {
        let mut buf = cloud.to_vec();
        let (n_ground, _, _) = self.ground.pick(&mut buf);
        let non_ground = buf[n_ground..].to_vec();

        // 预处理降噪：去除孤立噪点，改善墙体检测的 BFS 连通性
        let (denoised_ng, _) = self.denoise.denoise(&non_ground);

        let mut wall_buf = denoised_ng.clone();
        let (n_wall, _) = self.wall.pick(&mut wall_buf);
        let non_wall = wall_buf[n_wall..].to_vec();

        Preprocessed::Wall { non_ground: denoised_ng, non_wall }
    }
}

/// 地面+墙面检测预处理器（无降噪），cluster_bench 使用。
/// 管线：地面 → 墙体，降噪移至各聚类策略内部处理。
pub struct GroundWallPreprocessor {
    ground: Box<dyn GroundPickStrategy>,
    wall: Box<dyn WallPickStrategy>,
}

impl GroundWallPreprocessor {
    pub fn new(
        ground: Box<dyn GroundPickStrategy>,
        wall: Box<dyn WallPickStrategy>,
    ) -> Self {
        Self { ground, wall }
    }

    pub fn default() -> Self {
        Self::new(
            Box::new(PeakScan::new()),
            Box::new(BevLsd::with_params(0.08, 20)),
        )
    }
}

impl Preprocessor for GroundWallPreprocessor {
    fn name(&self) -> &str { "ground+wall" }
    fn preprocess(&mut self, cloud: &[[f32; 3]]) -> Preprocessed {
        let mut buf = cloud.to_vec();
        let (n_ground, _, _) = self.ground.pick(&mut buf);
        let non_ground = buf[n_ground..].to_vec();

        let mut wall_buf = non_ground.clone();
        let (n_wall, _) = self.wall.pick(&mut wall_buf);
        let non_wall = wall_buf[n_wall..].to_vec();

        Preprocessed::Wall { non_ground, non_wall }
    }
}

/// 地面+墙面+降噪预处理器，旧版 cluster_bench 使用。
/// 管线：地面 → 降噪（预处理）→ 墙体 → 降噪（后处理，聚类前清洁）
pub struct DenoisePreprocessor {
    wall_pp: WallPreprocessor,
    denoise: Box<dyn DenoiseStrategy>,
}

impl DenoisePreprocessor {
    pub fn new(
        wall_pp: WallPreprocessor,
        denoise: Box<dyn DenoiseStrategy>,
    ) -> Self {
        Self { wall_pp, denoise }
    }
}

impl Preprocessor for DenoisePreprocessor {
    fn name(&self) -> &str { "ground+denoise+wall+denoise" }
    fn preprocess(&mut self, cloud: &[[f32; 3]]) -> Preprocessed {
        let (non_ground, non_wall) = match self.wall_pp.preprocess(cloud) {
            Preprocessed::Wall { non_ground, non_wall } => (non_ground, non_wall),
            _ => unreachable!(),
        };

        // 后处理降噪：去除聚类前的残留噪点
        let (denoised, _) = self.denoise.denoise(&non_wall);

        Preprocessed::Denoise { non_ground, non_wall, denoised }
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
    /// - `Preprocessed::Wall` 时返回非地面点
    /// - `Preprocessed::Passthrough` 时返回原始点云
    pub fn non_ground(&self) -> &'a [[f32; 3]] {
        match self.preprocessed {
            Preprocessed::Ground { non_ground }
            | Preprocessed::GroundDenoised { non_ground }
            | Preprocessed::Wall { non_ground, .. }
            | Preprocessed::Denoise { non_ground, .. } => non_ground,
            Preprocessed::Passthrough => self.cloud,
        }
    }

    /// 获取非墙面点（去除地面+墙面后的剩余点）。
    ///
    /// - `Preprocessed::Wall` 时返回非墙面点
    /// - `Preprocessed::Denoise` 时返回非墙面点
    /// - 其他情况 fallback 到 `non_ground()`
    pub fn non_wall(&self) -> &'a [[f32; 3]] {
        match self.preprocessed {
            Preprocessed::Wall { non_wall, .. } | Preprocessed::Denoise { non_wall, .. } => non_wall,
            _ => self.non_ground(),
        }
    }

    /// 获取降噪后的点（去除地面+墙面+稀疏噪点）。
    ///
    /// - `Preprocessed::Denoise` 时返回去噪后的点
    /// - 其他情况 fallback 到 `non_wall()`
    pub fn denoised(&self) -> &'a [[f32; 3]] {
        match self.preprocessed {
            Preprocessed::Denoise { denoised, .. } => denoised,
            _ => self.non_wall(),
        }
    }
}

// ── 策略统计 ──────────────────────────────────────────────

/// 策略运行统计，由 BenchHarness 收集后导出 JSON。
#[derive(Serialize)]
pub struct BenchStats {
    pub name: String,
    pub frame_count: usize,
    pub total_ms: f64,
    pub frame_times: Vec<f64>,
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

    /// 返回运行统计（用于 JSON 导出）。
    fn stats(&self) -> BenchStats;

    /// 策略特有指标，用于 JSON 导出（平均每帧值）。
    /// 返回 key→value 映射，会被合并到 JSON 输出中。
    fn extra_metrics(&self) -> Vec<(String, f64)> {
        Vec::new()
    }
}

/// 将聚类索引列表转换为点集，并统计噪声数量。
///
/// 输入 `(sampled, objects)` 是 ClusteringStrategy 的产出格式：
/// - `objects` 中每个 `Vec<usize>` 是 sampled 中的索引
/// - 未出现在任何簇中的索引视为噪声
/// 返回 `(各簇实际点集, 噪声点数)`。
pub fn to_cluster_result(points: &[[f32; 3]], objects: &[Vec<usize>]) -> (Vec<Vec<[f32; 3]>>, usize) {
    let total: usize = objects.iter().map(|c| c.len()).sum();
    (objects.iter().map(|c| c.iter().map(|&i| points[i]).collect()).collect(), points.len() - total)
}
