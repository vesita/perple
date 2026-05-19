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
// AdaptiveDepthPolicy — 距离自适应分辨率
//
// 核心公式（log₂ + 抑制系数）：
//   target_res(r) = res₀ * (1 + β * log₂(max(r, r₀) / r₀))
//
// 其中 r = √(cx² + cy²) 是节点中心到传感器的距离。
// 近处（r ≤ r₀）保持 res₀ 的精细分辨率，远处分辨率按 log₂ 缓慢增大。
// β 控制增长速率：β=0 → 恒定分辨率；β=1 → 每次距离翻倍，分辨率粗化一倍。
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct AdaptiveDepthPolicy {
    /// 硬上限深度（防止递归过深）
    pub global_max_depth: usize,
    /// 基准分辨率（米），r ≤ r₀ 时使用此值
    pub res0: f32,
    /// 基准距离（米），r ≤ r₀ 时不衰减
    pub r0: f32,
    /// 抑制系数，β ∈ [0, 2]
    pub beta: f32,
}

impl AdaptiveDepthPolicy {
    /// 在 (cx, cy) 处的目标叶子对角线长度
    pub fn target_resolution(&self, cx: f32, cy: f32) -> f32 {
        let r_sq = cx * cx + cy * cy;
        let r0_sq = self.r0 * self.r0;
        if r_sq <= r0_sq {
            return self.res0;
        }
        let log2_ratio = 0.5 * (r_sq / r0_sq).log2();
        self.res0 * (1.0 + self.beta * log2_ratio)
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
