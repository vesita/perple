use std::collections::HashMap;

/// 运行模式。
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BenchMode {
    /// 快速测试（1 帧），不跳过任何策略。
    Quick,
    /// 全量测试（多帧），跳过 slowest_ms > 100ms 的策略。
    Full,
    /// 单策略单参数（传统模式），不写 TOML。
    Single,
}

/// 从 CLI args 提取 `--key=value` 对，无前缀的返回 `args` 序列。
pub struct CliArgs {
    pub argv: Vec<String>,
    pub map: HashMap<String, String>,
}

impl CliArgs {
    pub fn parse(args: &[String]) -> Self {
        let mut argv = Vec::new();
        let mut map = HashMap::new();
        for a in args {
            if let Some(eq) = a.find('=') {
                let key = a[..eq].trim_start_matches('-');
                let val = a[eq + 1..].to_string();
                map.insert(key.to_string(), val);
            } else if a.starts_with('-') {
                map.insert(a.trim_start_matches('-').to_string(), "true".to_string());
            } else {
                argv.push(a.clone());
            }
        }
        Self { argv, map }
    }

    pub fn get<T: std::str::FromStr>(&self, key: &str, default: T) -> T {
        self.map.get(key).and_then(|v| v.parse().ok()).unwrap_or(default)
    }

    pub fn has(&self, key: &str) -> bool {
        self.map.contains_key(key)
    }

    pub fn strategy(&self) -> Option<String> {
        self.map.get("strategy").cloned()
    }

    /// 返回运行模式。
    pub fn mode(&self) -> BenchMode {
        match self.map.get("mode").map(|s| s.as_str()) {
            Some("quick") => BenchMode::Quick,
            Some("full") => BenchMode::Full,
            _ => BenchMode::Single,
        }
    }
}
