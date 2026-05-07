use redra_client::{RdraWriter, spawn_point, spawn_cube};
use crate::utils::boxes::Box3D;

/// 策略测试的数据输出模块。
///
/// 封装 RdraWriter，提供点云和检测框的写入辅助方法。
/// 每个策略持有独立的 BenchRecorder，所有帧写入同一个 .rdra 文件。
pub struct BenchRecorder {
    writer: RdraWriter,
    base_id: u64,
    point_counter: u64,
    write_raw: bool,
}

impl BenchRecorder {
    pub fn new() -> Self {
        BenchRecorder {
            writer: RdraWriter::new(),
            base_id: 0,
            point_counter: 0,
            write_raw: false,
        }
    }

    /// 设置是否写入原始点云背景。
    ///
    /// 开启后调用 `write_raw_cloud` 会写入原始点云作为背景参照层。
    /// 关闭时 `write_raw_cloud` 为 no-op，避免与分类点云重复。
    pub fn set_write_raw(&mut self, enable: bool) {
        self.write_raw = enable;
    }

    /// 开始新帧，清理上一帧的实体。
    pub fn begin_frame(&mut self, frame_idx: usize) {
        self.writer.destroy_all();
        self.base_id = frame_idx as u64 * 1_000_000;
        self.point_counter = 0;
    }

    /// 写入原始点云背景（受 `write_raw` 开关控制）。
    ///
    /// 开启时写入原始点云作为背景参照层，关闭时为 no-op。
    /// 用于避免与后续分类点云（地面/非地面/墙面等）重复写入。
    pub fn write_raw_cloud(&mut self, points: &[[f32; 3]], material: &str, max_points: usize) {
        if !self.write_raw { return; }
        self.write_point_cloud(points, material, max_points);
    }

    /// 写入点云，自动下采样到 max_points。
    pub fn write_point_cloud(&mut self, points: &[[f32; 3]], material: &str, max_points: usize) {
        let step = (points.len() / max_points.max(1)).max(1);
        for (i, p) in points.iter().enumerate() {
            if i % step == 0 {
                let id = self.base_id + self.point_counter * 4;
                self.writer.spawn(spawn_point(*p, material).id(id));
                self.point_counter += 1;
            }
        }
    }

    /// 写入检测框列表，每个框带 tag。
    pub fn write_boxes(&mut self, boxes: &[(Box3D, String)], material: &str) {
        for (i, (box3d, tag)) in boxes.iter().enumerate() {
            let verts: Vec<(f32, f32, f32)> = box3d.vertices().iter()
                .map(|v| (v.x, v.y, v.z))
                .collect();
            self.writer.spawn(
                spawn_cube(verts, material)
                    .id(self.base_id + 500_000 + i as u64)
                    .tag(tag.clone())
            );
        }
    }

    /// 结束当前帧。
    pub fn end_frame(&mut self) {
        self.writer.end_frame();
    }

    /// 保存到 .rdra 文件。
    pub fn save(&self, path: &str) -> Result<(), String> {
        self.writer.save(path)
    }

    /// 清空所有帧数据。
    pub fn clear(&mut self) {
        self.writer.clear();
    }
}

impl Default for BenchRecorder {
    fn default() -> Self { Self::new() }
}
