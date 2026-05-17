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
cargo run --example cluster_bench -- --strategy=prune_qt --denoise-radius=0.20 --denoise-min-pts=3  # 聚类（降噪默认开启）
cargo run --example denoise_bench -- --mode=quick    # 降噪快速测试

# Cluster strategy comparison (eval_ablation)
cargo run --release --example eval_ablation -- --cluster-toml 'strategy="prune_qt"' --center-dist 0.5 --frames 408  # prune_qt 最优策略评估
cargo run --release --example eval_ablation -- --cluster-toml 'strategy="dbscan_qt"' --center-dist 0.5 --frames 408  # dbscan_qt 对比
cargo run --release --example eval_ablation -- --cluster-toml 'strategy="cc"' --center-dist 0.5 --frames 408  # 连通域聚类对比
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

### Three-Stage Processing Pipeline (Ground → Wall → Cluster)

The point cloud processing pipeline: ground extraction → wall extraction → clustering (with internal denoise).

```
Raw Cloud (~20k pts)
  → Ground Extraction (GroundPickStrategy: peak_scan/histogram/ransac)
    → Wall Extraction (WallPickStrategy: bev_lsd / bev_edlines, image-based edge detection)
      → Post-Clustering (ClusteringStrategy: prune_qt/dbscan_qt/lvdot/xy_dbscan/cc/ransac/seq, denoise internalized)
        → YOLO fusion + Tracking → Detection Results
```

Key insight: image-based wall detection (BevEdLines > BevLsd > BevHough) outperforms all geometric methods. BevEdLines achieves Person F1 0.745 (P=82.3%, R=68.1%, 3-run average) vs BevLsd 0.686 (P=84.5%, R=57.8%) — edlines' fragmented edge chains better preserve near-wall pedestrian points. Tracking: MOTA 55.3%, IDF1 76.0% (center-dist 0.5m, 3-run average).

### Key Fixes

- **Density weighting `cluster.rs`**: Formula changed from `1/r^α` → `r^α` (sign inversion bug). Original code amplified centroid bias toward sensor instead of compensating. Fix improved F1 by +8.3%.
- **Tracker container**: `HashMap` → `BTreeMap` for `tracked_objects` to eliminate iteration-order non-determinism.
- **YOLO label smoothing**: `yolo_smooth.rs` + integrated in `main.rs` and `eval_labeled.rs` — frame-to-frame momentum filter on YOLO "person" labels, reducing label flicker. No significant impact on P/R/F1 metrics (verified by 3×3 runs).
- **Config override system `config.rs`**: Added `update_from_toml()` with `PartialConfig` structs + `init_config()` with `OnceLock`, replacing compile-time-only config. **Bug**: `Option<T>` fields (`min_points_per_cluster`, `max_points_per_node`, `max_tree_depth`) need `Some()` wrapping — the `update_cluster_field!` macro's `value.clone()` produces `T`, not `Option<T>`.
- **Multi-frame accumulation** (tested, abandoned): Merging N frames of non-ground points before clustering increased Person Recall +5.8pp but caused FP explosion (135→538) as accumulated clusters produced oversized boxes that passed geometry fallback.
- **Default wall strategy `bev_edlines` (binary direction)**: Refactored from angle-based (atan2/cos/sin) to binary direction (`|gx| >= |gy|` → EDGE_VERTICAL/HORIZONTAL), matching C++ reference. Removes all trigonometry, uses `|gx|+|gy|` magnitude (faster: ~14ms vs 17ms for bev_lsd). Current default config: `wall_strategy="bev_edlines"`, `wall_distance=0.08`. Replaced `dbscan_qt` after comprehensive 9-strategy benchmark (408 frames). Binary-direction EDLines + prune_qt achieve Person F1 0.745 (P=82.3%, R=68.1%, 3-run average), improving recall by +11.4pp and reducing FP by 59% vs old dbscan_qt. The factory was also updated to read `merge_patience` and `min_points_per_cluster` from config instead of hardcoded values.

- `src/cloud/wall.rs` — WallPickStrategy trait + XYGrid shared infra + wall module root: `bev_lsd` (active), `bev_edlines` (active), and `bev_hough` (reserved)
- `src/cloud/wall/bev_lsd.rs` — BEV image + LSD 风格区域生长墙体检测
- `src/cloud/wall/bev_edlines.rs` — BEV image + EDLines 锚点检测链式追踪墙体检测
- `src/cloud/wall/bev_hough.rs` — Hough 变换备选
- `src/cloud/classify/core.rs` — Three-stage pipeline: ground → wall → cluster (no denoise stages)
- `src/yolo_smooth.rs` — YOLO 帧间标签平滑模块（Camera→Fuse 间介入）
- `src/bench/strategy.rs` — Preprocessor trait + GroundWallPreprocessor (地面→墙体, 无降噪) + GroundPreprocessor
