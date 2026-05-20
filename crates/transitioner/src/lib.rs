/// 过渡器 — 带迟滞的阈值防抖器
///
/// 对泛型输入信号施加入/出两道门槛，配合连续帧确认防止状态抖动：
///
/// | 当前状态 | 条件 | `retain=true` | `retain=false` |
/// |---------|------|--------------|----------------|
/// | OFF | `value ≥ on_threshold` | 计数器递增，达 `cooldown` 后切 ON | 同左 |
/// | OFF | `value < on_threshold` | 计数器复位 | 同左 |
/// | ON  | `value ≤ off_threshold` | 计数器递增，达 `cooldown` 后切 OFF | **立即切 OFF** |
/// | ON  | `value > off_threshold` | 计数器复位 | 同左 |
///
/// `retain=true`（默认）适合需要双向迟滞的场景（如 Moving↔Movable），
/// `retain=false` 适合游程计数场景（如几何判定：一次失败立即复位，持续通过才标记）。
///
/// # 示例
///
/// ```
/// use transitioner::Transitioner;
///
/// // 速度门槛：>0.5 进入 ON，<0.3 回到 OFF，需保持 3 帧
/// let mut t = Transitioner::new(0.5, 0.3, 3);
///
/// assert!(!t.feed(&0.2));   // OFF: 0.2 < 0.5 → 复位
/// assert!(!t.feed(&0.6));   // OFF: 0.6 ≥ 0.5 → counter=1
/// assert!(!t.feed(&0.7));   // OFF: counter=2
/// assert!( t.feed(&0.9));   // ON!  counter=3 ≥ cooldown=3
///
/// // 现在 ON，迟滞带中保持
/// assert!( t.feed(&0.4));   // ON: 0.4 > 0.3 → 复位 counter
/// assert!( t.feed(&0.3));   // ON: 0.3 ≤ 0.3 → counter=1
/// assert!( t.feed(&0.3));   // ON: counter=2
/// assert!(!t.feed(&0.3));   // OFF! counter=3 ≥ cooldown → 切 OFF
/// assert!(!t.state());
/// ```
pub struct Transitioner<T> {
    /// 当前输出状态
    state: bool,
    /// 关→开 门槛
    on_threshold: T,
    /// 开→关 门槛
    off_threshold: T,
    /// 状态切换前需连续满足条件的帧数
    cooldown: u32,
    /// 当前连续计数
    counter: u32,
    /// true=双向迟滞（ON 后需 cooldown 帧回落）；
    /// false=信号一掉立即复位（游程计数模式）
    retain: bool,
}

impl<T: PartialOrd> Transitioner<T> {
    /// 创建新过渡器，初始状态为 OFF，默认 `retain=true`
    pub fn new(on_threshold: T, off_threshold: T, cooldown: u32) -> Self {
        Transitioner { state: false, on_threshold, off_threshold, cooldown, counter: 0, retain: true }
    }

    /// 创建新过渡器，初始状态为 ON，默认 `retain=true`
    pub fn new_on(on_threshold: T, off_threshold: T, cooldown: u32) -> Self {
        Transitioner { state: true, on_threshold, off_threshold, cooldown, counter: 0, retain: true }
    }

    /// 设置状态回落模式：
    /// - `true`：双向迟滞，信号翻转后需 `cooldown` 帧才回落
    /// - `false`：信号一掉立即复位（适合游程计数）
    pub fn with_retain(mut self, retain: bool) -> Self {
        self.retain = retain;
        self
    }

    /// 输入新值，返回当前状态（true=ON, false=OFF）
    pub fn feed(&mut self, value: &T) -> bool {
        if self.state {
            if *value <= self.off_threshold {
                if self.retain {
                    self.counter += 1;
                    if self.counter >= self.cooldown {
                        self.state = false;
                        self.counter = 0;
                    }
                } else {
                    // retain=false：信号一掉立即复位
                    self.state = false;
                    self.counter = 0;
                }
            } else {
                self.counter = 0;
            }
        } else if *value >= self.on_threshold {
            self.counter += 1;
            if self.counter >= self.cooldown {
                self.state = true;
                self.counter = 0;
            }
        } else {
            self.counter = 0;
        }
        self.state
    }

    /// 当前状态
    pub fn state(&self) -> bool {
        self.state
    }

    /// 复位到 OFF
    pub fn reset(&mut self) {
        self.counter = 0;
        self.state = false;
    }

    /// 当前进度（0.0 ~ 1.0），cooldown=0 时始终返回 1.0
    pub fn progress(&self) -> f32 {
        if self.cooldown == 0 { 1.0 } else { self.counter as f32 / self.cooldown as f32 }
    }
}

impl Transitioner<bool> {
    /// 快捷构造：输入即条件，`true` 为 ON 信号，`false` 为 OFF 信号
    pub fn new_bool(cooldown: u32) -> Self {
        Transitioner::new(true, false, cooldown)
    }

    /// 快捷构造，初始为 ON
    pub fn new_bool_on(cooldown: u32) -> Self {
        Transitioner::new_on(true, false, cooldown)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_hysteresis() {
        let mut t = Transitioner::new(10.0, 5.0, 3);
        assert!(!t.state());

        // 未达门槛
        assert!(!t.feed(&4.0));
        assert!(!t.feed(&7.0));
        // 达到门槛，开始累积
        assert!(!t.feed(&10.0)); // counter=1
        assert!(!t.feed(&12.0)); // counter=2
        assert!( t.feed(&15.0)); // counter=3 → ON
        assert!( t.state());

        // 在迟滞带中保持 ON
        assert!( t.feed(&7.0));
        assert!( t.state());

        // 降到 off 门槛以下
        assert!( t.feed(&5.0)); // counter=1
        assert!( t.feed(&4.0)); // counter=2
        assert!(!t.feed(&3.0)); // counter=3 → OFF
        assert!(!t.state());
    }

    #[test]
    fn test_bool_transitioner() {
        let mut t = Transitioner::new_bool(3);
        assert!(!t.feed(&false));
        assert!(!t.feed(&true));  // counter=1
        assert!(!t.feed(&true));  // counter=2
        assert!( t.feed(&true));  // counter=3 → ON
        assert!( t.feed(&true));  // stays ON
        assert!( t.feed(&false)); // counter=1 (going OFF)
        assert!( t.feed(&false)); // counter=2
        assert!(!t.feed(&false)); // counter=3 → OFF
    }

    #[test]
    fn test_reset() {
        let mut t = Transitioner::new(1.0, 0.0, 5);
        t.feed(&2.0);
        t.feed(&2.0);
        assert_eq!(t.counter, 2);
        t.reset();
        assert_eq!(t.counter, 0);
        assert!(!t.state());
    }

    #[test]
    fn test_zero_cooldown() {
        let mut t = Transitioner::new(1.0, 0.0, 0);
        assert!(t.feed(&2.0));  // 0 cooldown → instant ON
        assert!(t.state());
        assert!(t.feed(&0.5));  // still ON (0.5 > 0.0, off threshold not hit)
        assert!(!t.feed(&0.0)); // OFF (0.0 ≤ 0.0, instant)
    }
}
