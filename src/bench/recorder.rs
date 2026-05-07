use crate::utils::rdra::FrameWriter;
use crate::utils::boxes::Box3D;

/// 策略测试的数据输出模块。
///
/// 封装 FrameWriter，提供策略测试专用的写入方法。
/// 每个策略持有独立的 BenchRecorder，所有帧写入同一个 .rdra 文件。
pub struct BenchRecorder {
    inner: FrameWriter,
    write_raw: bool,
}

impl BenchRecorder {
    pub fn new() -> Self {
        BenchRecorder { inner: FrameWriter::new(), write_raw: false }
    }

    /// 设置是否写入原始点云背景。
    pub fn set_write_raw(&mut self, enable: bool) {
        self.write_raw = enable;
    }

    /// 开始新帧。
    pub fn begin_frame(&mut self, frame_idx: usize) {
        self.inner.begin_frame(frame_idx);
    }

    /// 写入原始点云背景（受 write_raw 开关控制）。
    pub fn write_raw_cloud(&mut self, points: &[[f32; 3]], material: &str, max_points: usize) {
        if !self.write_raw { return; }
        self.inner.write_cloud(points, material, max_points);
    }

    /// 写入分类点云，自动下采样到 max_points。
    pub fn write_point_cloud(&mut self, points: &[[f32; 3]], material: &str, max_points: usize) {
        self.inner.write_cloud(points, material, max_points);
    }

    /// 写入检测框列表，每个框带 tag。
    pub fn write_boxes(&mut self, boxes: &[(Box3D, String)], material: &str) {
        self.inner.write_boxes(boxes, material);
    }

    /// 结束当前帧。
    pub fn end_frame(&mut self) {
        self.inner.end_frame();
    }

    /// 保存到 .rdra 文件。
    pub fn save(&self, path: &str) -> Result<(), String> {
        self.inner.save(path)
    }

    /// 清空所有帧数据。
    pub fn clear(&mut self) {
        self.inner.clear();
    }
}

impl Default for BenchRecorder {
    fn default() -> Self { Self::new() }
}
