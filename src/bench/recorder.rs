use redra_client::{RdraWriter, spawn_point, spawn_cube};
use crate::utils::boxes::Box3D;

/// 策略测试的数据输出模块。
///
/// 封装 RdraWriter，提供点云和检测框的写入辅助方法。
/// 每个策略持有独立的 BenchRecorder，所有帧写入同一个 .rdra 文件。
pub struct BenchRecorder {
    writer: RdraWriter,
    base_id: u64,
}

impl BenchRecorder {
    pub fn new() -> Self {
        BenchRecorder {
            writer: RdraWriter::new(),
            base_id: 0,
        }
    }

    /// 开始新帧，清理上一帧的实体。
    pub fn begin_frame(&mut self, frame_idx: usize) {
        self.writer.destroy_all();
        self.base_id = frame_idx as u64 * 1_000_000;
    }

    /// 写入点云（白色），自动下采样到 max_points。
    pub fn write_point_cloud(&mut self, points: &[[f32; 3]], material: &str, max_points: usize) {
        let step = (points.len() / max_points.max(1)).max(1);
        for (i, p) in points.iter().enumerate() {
            if i % step == 0 {
                self.writer.spawn(spawn_point(*p, material).id(self.base_id + i as u64 * 4));
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
