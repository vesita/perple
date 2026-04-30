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
