# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

```bash
# Build (default includes graph feature)
cargo build

# Release build
cargo build --release

# Run with UI
cargo run

# Run headless (no Bevy/egui)
cargo run --no-default-features

# Run with client feature
cargo run --features client

# Run tests in a specific crate
cargo test -p smooth-bevy-cameras

# Run a single test
cargo test -p smooth-bevy-cameras -- test_initial_states

# Run client test examples (start server first)
cargo run --example redra_test --package redra_client
cargo run --example label_test --package redra_client

# Bench examples
cargo run --example ground_bench -- --mode=quick    # 地面提取快速测试
cargo run --example wall_bench -- --mode=quick       # 墙体提取快速测试
cargo run --example cluster_bench -- --mode=quick    # 后聚类快速测试
cargo run --example cluster_bench -- --strategy=xy_dbscan --denoise-radius=0.20 --denoise-min-pts=3  # 聚类（降噪默认开启）
cargo run --example denoise_bench -- --mode=quick    # 降噪快速测试
cargo run --example pipeline_evolution_bench         # 管线演化对比（论文用）
cargo run --example pipeline_evolution_bench -- --frames 50  # 50 帧管线演化
cargo run --example wall_pipeline_bench -- --mode=quick  # 墙体管线对比（不同墙体策略对聚类影响）
cargo run --example wall_pipeline_bench -- --mode=full   # 全量墙体管线对比

# Python analysis pipeline
.venv/Scripts/python.exe scripts/bench_pipeline.py --tasks ground,cluster,wall,denoise  # 完整流水线
.venv/Scripts/python.exe scripts/bench_pipeline.py --analysis-only               # 仅从已有数据重绘图
.venv/Scripts/python.exe scripts/run_wall_pipeline.py            # 墙体管线对比快速测试
.venv/Scripts/python.exe scripts/run_wall_pipeline.py --mode=full  # 墙体管线对比全量测试

# Generate protobuf code (requires protoc)
python script/compile_proto.py

# Build data packs
python script/build_packs.py
```

## Architecture

### Layer Stack (top-down)

```
control/    — Orchestration: plugin composition, cross-module wiring
data/       — Pure data: frame management, protocol conversion, persistence
assets/     — Resources: materials (bevy_materialize TOML), fonts
render/     — Bevy rendering: scene init, frame rendering, camera, picking
ui/         — egui UI: VS Code-style sidebar, playback controls, file manager, wheel menu
```

### Entry Point

`main.rs` → `control::ControlPlugin` (a Bevy `Plugin`) composes all sub-plugins in dependency order. The app uses `DefaultPlugins` + `MeshPickingPlugin` + `LookTransformPlugin`.

### Workspace Crates

| Crate | Role |
|-------|------|
| `expto` | Core protocol: protobuf types, TCMP encoding/decoding, config loading |
| `redra_net` | Async TCP networking via Tokio, RDChannel for frame data |
| `redra_client` | Test client: sends example frame data over network |
| `redra_geo` | Geometry utilities: axis convention conversion, transform helpers |
| `redra_calib` | Point cloud registration/calibration module |
| `smooth-bevy-cameras` | FPS camera controller (forked upstream, has its own state machine) |
| `bevy_wheel_menu` | Radial wheel menu (forked Bevy plugin) |
| `utils` | Shared utility functions |

### Features

- `graph` (default) — Enables Bevy rendering + egui UI + file dialogs
- `client` — Enables `redra_client` test utilities

### Key Data Types

- `FrameManager` (Resource) — Owns `Vec<KeyFrame>`, manages current frame index, ingests `Unit` stream
- `KeyFrame` — Contains `packs: Vec<Inpto>` + `ids: HashMap<u64, usize>` entity lookup
- `Inpto` — Intermediate representation: `Transform`, `ExMesh`, material path, optional `Tag`
- `PlaybackState` (Resource) — Play/pause, FPS, frame navigation state

### Rendering Pipeline

`render::frame_renderer` reads `FrameManager` each frame in `Update`, spawns/despawns Bevy entities to match current keyframe. Camera is a separate `FpsCameraController` component.

### Five-Stage Processing Pipeline (Ground → Denoise → Wall → Denoise → Cluster)

The point cloud processing pipeline: ground extraction → pre-denois e → wall extraction → post-denoise → clustering.

```
Raw Cloud (~20k pts)
  → Ground Extraction (GroundPickStrategy: histogram/peak_scan/ransac)
    → Pre-Denoise (DenoiseStrategy: RadiusOutlierRemoval r=0.30 m=3, 改善墙体 BFS 连通性)
      → Wall Extraction (WallPickStrategy: XYRansacWall/TopDown/Quadtree)
        → Post-Denoise (DenoiseStrategy: RadiusOutlierRemoval r=0.20 m=3, 聚类前清洁)
          → Post-Clustering (ClusteringStrategy: xy_dbscan/lvdot/range_image/xy_grid_dbscan)
            → Detection Results (障碍物簇)
```

Key insight: downsampling gives 244x speedup, ground removal adds 1.4x, wall removal reduces noise.

- `src/bench/strategy.rs` — Preprocessor trait + WallPreprocessor (地面→降噪→墙体) + DenoisePreprocessor (封装 WallPreprocessor + 后降噪)
- `src/cloud/denoise.rs` — DenoiseStrategy trait + RadiusOutlierRemoval/SOR
- `examples/cluster_bench.rs` — 聚类策略 bench（降噪默认开启，`--denoise` 标志已移除）
- `examples/wall_pipeline_bench.rs` — 墙体管线对比：固定后聚类（xy_grid_dbscan e0.15_m3），遍历墙体策略
- `examples/denoise_bench.rs` — 降噪策略 bench
- `examples/pipeline_evolution_bench.rs` — 管线演化对比（Era1→Era2→Era3）
- `scripts/bench_pipeline.py` — Python 分析图生成
- `scripts/run_wall_pipeline.py` — 墙体管线对比自动执行脚本
