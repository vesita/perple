use redra_client::{RdraWriter, ShapeBuilder, spawn_point, spawn_cube, spawn_line};
use crate::utils::boxes::Box3D;

/// 通用 .rdra 帧写入器。
///
/// 封装 RdraWriter，提供点云、包围盒、线段的写入方法。
/// 支持原始点云背景开关：开启时 `write_raw_cloud` 写入原始点云，
/// 关闭时为 no-op。`write_cloud` 始终写入（用于分类后的点云）。
///
/// ID 分配：每帧独立，点云 0-99999，包围盒 800000+，线段 900000+。
pub struct FrameWriter {
    writer: RdraWriter,
    base_id: u64,
    point_counter: u64,
    box_counter: u64,
    line_counter: u64,
    raw_material: Option<String>,
}

impl FrameWriter {
    pub fn new() -> Self {
        Self {
            writer: RdraWriter::new(),
            base_id: 0,
            point_counter: 0,
            box_counter: 0,
            line_counter: 0,
            raw_material: None,
        }
    }

    /// 设置原始点云材质。设置后 `write_raw_cloud` 生效，否则为 no-op。
    pub fn set_raw_material(&mut self, material: impl Into<String>) {
        self.raw_material = Some(material.into());
    }

    /// 开始新帧。
    pub fn begin_frame(&mut self, frame_idx: usize) {
        self.writer.destroy_all();
        self.base_id = frame_idx as u64 * 1_000_000;
        self.point_counter = 0;
        self.box_counter = 0;
        self.line_counter = 0;
    }

    /// 写入原始点云（受 `set_raw_material` 控制）。
    ///
    /// 未设置 raw_material 时为 no-op。用于避免与分类点云重复。
    pub fn write_raw_cloud(&mut self, points: &[[f32; 3]], max_points: usize) {
        let material = match &self.raw_material {
            Some(m) => m.clone(),
            None => return,
        };
        self.write_cloud_inner(points, &material, max_points);
    }

    /// 写入分类点云（始终写入）。
    pub fn write_cloud(&mut self, points: &[[f32; 3]], material: &str, max_points: usize) {
        self.write_cloud_inner(points, material, max_points);
    }

    fn write_cloud_inner(&mut self, points: &[[f32; 3]], material: &str, max_points: usize) {
        let step = (points.len() / max_points.max(1)).max(1);
        for (i, p) in points.iter().enumerate() {
            if i % step == 0 {
                let id = self.base_id + self.point_counter;
                self.writer.spawn(spawn_point(*p, material).id(id));
                self.point_counter += 1;
            }
        }
    }

    /// 写入多个点云组（每组独立材质）。
    ///
    /// 用于将不同类别的点云一次性写入，每组自动下采样到 max_points。
    pub fn write_cloud_groups(&mut self, groups: &[(&[[f32; 3]], &str)], max_points: usize) {
        for &(points, material) in groups {
            self.write_cloud(points, material, max_points);
        }
    }

    /// 写入包围盒列表（半透明材质 + tag）。
    pub fn write_boxes(&mut self, boxes: &[(Box3D, String)], material: &str) {
        for (box3d, tag) in boxes {
            let verts: Vec<(f32, f32, f32)> = box3d.vertices().iter()
                .map(|v| (v.x, v.y, v.z))
                .collect();
            let id = self.base_id + 800_000 + self.box_counter;
            self.writer.spawn(
                spawn_cube(verts, material).id(id).tag(tag.clone())
            );
            self.box_counter += 1;
        }
    }

    /// 写入单个包围盒。
    pub fn write_box(&mut self, box3d: &Box3D, material: &str, tag: &str) {
        self.write_boxes(&[(box3d.clone(), tag.to_string())], material);
    }

    /// 写入线段。
    pub fn write_line(&mut self, from: [f32; 3], to: [f32; 3], material: &str) {
        let id = self.base_id + 900_000 + self.line_counter;
        self.writer.spawn(spawn_line(from, to, material).id(id));
        self.line_counter += 1;
    }

    /// 写入多条线段。
    pub fn write_lines(&mut self, lines: &[([f32; 3], [f32; 3])], material: &str) {
        for &(from, to) in lines {
            self.write_line(from, to, material);
        }
    }

    /// 写入自定义 ShapeBuilder。
    pub fn spawn(&mut self, builder: ShapeBuilder) -> u64 {
        self.writer.spawn(builder)
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

impl Default for FrameWriter {
    fn default() -> Self { Self::new() }
}
