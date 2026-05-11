pub mod cli;
pub mod config;
pub mod recorder;
pub mod strategy;
pub mod harness;

pub use cli::{CliArgs, BenchMode};
pub use config::{StrategyFamily, StatsInfo, load_task_strategies, update_strategy_stats, param_dirname, compute_median, get_f32, get_i64};
pub use recorder::{BenchRecorder, CLUSTER_PALETTE, mats};
pub use strategy::{BenchStrategy, BenchStats, FrameData, Preprocessed, Preprocessor, to_cluster_result};
pub use strategy::{PassthroughPreprocessor, GroundPreprocessor, WallPreprocessor, DenoisePreprocessor};
pub use harness::{BenchHarness, run_toml_bench, StrategyBuilder};
