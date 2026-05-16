use std::path::Path;

use crate::utils::rdra::FrameWriter;
use crate::utils::boxes::Box3D;
use redra_client::ShapeBuilder;

/// 12 色调色板（redra data/ 色板）
pub const CLUSTER_PALETTE: &[&str] = &[
    "cluster_01", "cluster_02", "cluster_03", "cluster_04",
    "cluster_05", "cluster_06", "cluster_07", "cluster_08",
    "cluster_09", "cluster_10", "cluster_11", "cluster_12",
];

/// 语义材质名（使用 redra 颜色名材质体系）
pub mod mats {
    /// 背景点云（暖白点）
    pub const BG: &str = "point_cloud";
    /// 地面点（语义绿）
    pub const GROUND: &str = "ground";
    /// 墙面点（高亮红，强对比）
    pub const WALL: &str = "bright_red";
    /// 噪声点（低饱和灰）
    pub const NOISE: &str = "dark_gray";
    /// 丢弃/辅助点（深蓝）
    pub const DISCARD: &str = "dark_blue";
    /// 默认障碍物包围盒（半透明灰）
    pub const BOX: &str = "disabled";
    /// 远距障碍物包围盒（青色）
    pub const FAR_BOX: &str = "cluster_07";
    /// 地面包围盒（半透明绿，不遮挡点云）
    pub const GROUND_BOX: &str = "green_transparent";
    /// 墙面包围盒（半透红）
    pub const WALL_BOX: &str = "red_transparent";
    /// 聚类包围盒（半透明橙，区别于默认 BOX）
    pub const CLUSTER_BOX: &str = "orange_transparent";
    /// 告警/待关注（亮橙）
    pub const ALERT: &str = "bright_orange";
    /// 选中/标记（亮黄）
    pub const SELECTED: &str = "bright_yellow";
}

/// 策略测试的数据输出模块。
///
/// 封装 FrameWriter，提供策略测试专用的写入方法。
/// 每个策略持有独立的 BenchRecorder，所有帧直接写入 SQLite 数据库。
///
/// 数据在每帧 `end_frame()` 时立即持久化，不累积在内存中。
/// 无需片段模式——无需临时文件，无需合并步骤。
pub struct BenchRecorder {
    inner: FrameWriter,
    write_raw: bool,
}

impl BenchRecorder {
    /// 创建写入器并打开/创建 SQLite 数据库文件。
    ///
    /// `path` 是输出数据库文件的路径。所有帧数据直接持久化到该文件。
    pub fn new(path: impl AsRef<Path>) -> Result<Self, String> {
        Ok(BenchRecorder {
            inner: FrameWriter::new(path)?,
            write_raw: false,
        })
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

    /// 写入多组分类点云，每组用独立材质染色。
    pub fn write_cloud_groups(&mut self, groups: &[(&[[f32; 3]], &str)], max_points: usize) {
        self.inner.write_cloud_groups(groups, max_points);
    }

    /// 写入检测框列表，每个框带 tag。
    pub fn write_boxes(&mut self, boxes: &[(Box3D, String)], material: &str) {
        self.inner.write_boxes(boxes, material);
    }

    /// 写入聚类结果：每个聚类用 12 色调色板循环染色，对应位置写包围盒。
    pub fn write_clusters(&mut self, clusters: &[Vec<[f32; 3]>], boxes: &[Box3D]) {
        for (i, cluster) in clusters.iter().enumerate() {
            if cluster.is_empty() { continue; }
            let color = CLUSTER_PALETTE[i % CLUSTER_PALETTE.len()];
            self.write_point_cloud(cluster, color, cluster.len());
        }
        if !boxes.is_empty() {
            let tagged: Vec<(Box3D, String)> = boxes.iter().enumerate()
                .map(|(i, b)| (b.clone(), format!("c{}", i)))
                .collect();
            self.write_boxes(&tagged, mats::CLUSTER_BOX);
        }
    }

    /// 写入一组带标签的聚类点云，跳过空簇。
    pub fn write_labeled_clusters(&mut self, clusters: &[(&str, &[[f32; 3]], &Box3D)]) {
        for (i, (label, pts, box3d)) in clusters.iter().enumerate() {
            if pts.is_empty() { continue; }
            let color = CLUSTER_PALETTE[i % CLUSTER_PALETTE.len()];
            self.write_point_cloud(pts, color, pts.len());
            self.write_boxes(&[((*box3d).clone(), label.to_string())], mats::CLUSTER_BOX);
        }
    }

    /// 写入地面检测结果：地面点（语义绿）+ 非地面聚类（色板）+ 各簇包围盒。
    pub fn write_ground_result(&mut self, ground: &[[f32; 3]], clusters: &[Vec<[f32; 3]>], boxes: &[Box3D]) {
        if !ground.is_empty() {
            self.write_point_cloud(ground, mats::GROUND, ground.len());
        }
        self.write_clusters(clusters, boxes);
    }

    /// 写入墙体检测结果：墙面点（红）+ 剩余聚类 + 包围盒 + 远距簇（特殊色）。
    pub fn write_wall_result(
        &mut self,
        wall: &[[f32; 3]],
        near_clusters: &[Vec<[f32; 3]>],
        near_boxes: &[Box3D],
        far_clusters: &[Vec<[f32; 3]>],
        far_distances: &[f32],
    ) {
        if !wall.is_empty() {
            self.write_point_cloud(wall, mats::WALL, wall.len());
        }
        for (i, cluster) in near_clusters.iter().enumerate() {
            if cluster.is_empty() { continue; }
            let color = CLUSTER_PALETTE[i % CLUSTER_PALETTE.len()];
            self.write_point_cloud(cluster, color, cluster.len());
        }
        if !near_boxes.is_empty() && near_boxes.len() == near_clusters.len() {
            let tagged: Vec<(Box3D, String)> = near_boxes.iter().enumerate()
                .map(|(i, b)| (b.clone(), format!("n{}", i)))
                .collect();
            self.write_boxes(&tagged, mats::WALL_BOX);
        }
        for (_i, cluster) in far_clusters.iter().enumerate() {
            if cluster.is_empty() { continue; }
            self.write_point_cloud(cluster, mats::FAR_BOX, cluster.len());
        }
        if !far_distances.is_empty() && far_distances.len() == far_clusters.len() {
            let tagged: Vec<(Box3D, String)> = far_boxes(far_clusters, far_distances);
            self.write_boxes(&tagged, mats::FAR_BOX);
        }
    }

    /// 直接添加自定义实体。
    pub fn spawn(&mut self, builder: ShapeBuilder) -> u64 {
        self.inner.spawn(builder)
    }

    /// 修改已有实体的材质。
    pub fn set_material(&mut self, id: u64, material: impl Into<String>) {
        self.inner.set_material(id, material);
    }

    /// 结束当前帧，将所有实体写入 SQLite。
    pub fn end_frame(&mut self) {
        self.inner.end_frame();
    }

    /// VACUUM 压缩数据库文件。
    pub fn save(&self) -> Result<(), String> {
        self.inner.save()
    }

    /// 将数据库复制到目标路径（VACUUM + 复制）。
    pub fn save_as(&self, dest: impl AsRef<Path>) -> Result<(), String> {
        self.inner.save_as(dest)
    }

    /// 清空所有帧数据。
    pub fn clear(&mut self) {
        self.inner.clear();
    }
}

fn far_boxes(clusters: &[Vec<[f32; 3]>], distances: &[f32]) -> Vec<(Box3D, String)> {
    clusters.iter().enumerate().filter_map(|(i, cluster)| {
        if cluster.is_empty() { return None; }
        let b = Box3D::from_cloud_aabb(cluster, 0.05);
        let d = distances.get(i).copied().unwrap_or(0.0);
        Some((b, format!("far{}_{:.0}m", i, d)))
    }).collect()
}
