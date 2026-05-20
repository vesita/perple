/// 四叉树节点分裂策略 — 决定何时停止细分
pub trait SplitPolicy: Send + Sync {
    /// 节点 (cx, cy) 在 depth 层、对角线 diagonal 时是否应继续分裂
    fn should_split(&self, depth: usize, cx: f32, cy: f32, diagonal: f32) -> bool;

    /// 全局最大深度（硬上限，防止无限递归）
    fn global_max_depth(&self) -> usize;
}

// ─────────────────────────────────────────────────────────────────────────────
// FixedDepthPolicy — 现有行为：固定最大深度
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct FixedDepthPolicy {
    pub max_depth: usize,
}

impl FixedDepthPolicy {
    pub fn new(max_depth: usize) -> Self {
        Self { max_depth }
    }
}

impl Default for FixedDepthPolicy {
    fn default() -> Self {
        Self { max_depth: 10 }
    }
}

impl SplitPolicy for FixedDepthPolicy {
    fn should_split(&self, depth: usize, _cx: f32, _cy: f32, _diagonal: f32) -> bool {
        depth < self.max_depth
    }

    fn global_max_depth(&self) -> usize {
        self.max_depth
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// AdaptiveDepthPolicy — 距离自适应分辨率（对数衰减）
//
// 核心公式：
//   target_res(r) = res₀ * (1 + β * log₂(1 + r / r₀))
//
// 其中 r = √(cx² + cy²) 是节点中心到传感器的距离。
// β 控制分辨率随对数距离的增长速率。
// 相比于 log₂(r/r₀) 形式，log₂(1+r/r₀) 在 r=0 处自然为 0（log₂1=0），
// 无需条件分支，且 β 物理含义更直观：β = (res(r₀) / res₀) - 1。
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct AdaptiveDepthPolicy {
    /// 硬上限深度（防止递归过深）
    pub global_max_depth: usize,
    /// 基准分辨率（米），r ≤ r₀ 时使用此值
    pub res0: f32,
    /// 基准距离（米），r ≤ r₀ 时不衰减
    pub r0: f32,
    /// 增长系数 k（米），控制分辨率随对数距离的增长率
    pub k: f32,
}

impl AdaptiveDepthPolicy {
    /// 在 (cx, cy) 处的目标叶子对角线长度
    ///
    /// 公式：res(r) = res₀ + k * log₂(1 + r / r₀)
    ///
    /// 加法形式：res₀ 独立控制近端分辨率，k 独立控制远端粗化速率，
    /// 两个参数解耦，物理含义更清晰。相比乘法形式 res₀*(1+β*log₂)，
    /// 近端 log₂→0 时不会将深度约束退化到几乎为零。
    pub fn target_resolution(&self, cx: f32, cy: f32) -> f32 {
        let r = (cx * cx + cy * cy).sqrt(); // 欧氏距离
        let log2_arg = 1.0 + r / self.r0;
        self.res0 + self.k * log2_arg.log2()
    }
}

impl SplitPolicy for AdaptiveDepthPolicy {
    fn should_split(&self, depth: usize, cx: f32, cy: f32, diagonal: f32) -> bool {
        if depth >= self.global_max_depth {
            return false;
        }
        diagonal > self.target_resolution(cx, cy)
    }

    fn global_max_depth(&self) -> usize {
        self.global_max_depth
    }
}
