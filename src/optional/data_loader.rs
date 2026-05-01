use image::DynamicImage;
use log::{error, info};
use pcd_rs::DynReader;
use std::collections::HashMap;
use std::{fs, io, sync::Arc, thread, time::Duration};

use crate::{
    color::load_image,
    swapl::global_swapl,
    utils::stream::{Eap, Stream},
};

pub fn load_cloud<R>(reader: &mut DynReader<R>) -> Vec<[f32; 3]>
where
    R: std::io::BufRead,
{
    let mut result = Vec::new();
    while let Some(record_result) = reader.next() {
        if let Ok(point) = record_result {
            if let Some(coords) = point.to_xyz() {
                result.push(coords);
            }
        }
    }
    result
}

/// 数据加载器
///
/// DataLoader 负责从文件系统加载 2D 和 3D 检测数据，并将它们写入相应的数据流中。
pub struct DataLoader {
    /// 2D 检测结果数据流
    clr_stream: Eap<Stream<DynamicImage>>,
    /// 3D 检测结果数据流
    cld_stream: Eap<Stream<Vec<[f32; 3]>>>,
    /// 数据文件路径
    target_path: String,
    /// 图像路径（独立路径模式）
    image_path: Option<String>,
    /// 点云路径（独立路径模式）
    pcd_path: Option<String>,

    files: Vec<Vec<String>>,

    /// 最多加载多少帧（文件对数），None 表示不限
    frame_limit: Option<usize>,

    /// 当前加载到的帧索引（用于按需加载）
    current_index: usize,

    /// 内存缓冲：预加载全部图像，load_next 从此读取（无磁盘 I/O）
    images: Vec<DynamicImage>,
    /// 内存缓冲：预加载全部点云
    clouds: Vec<Vec<[f32; 3]>>,
}

impl DataLoader {
    /// 创建一个新的数据加载器
    ///
    /// 通过 Swapl 数据中枢获取所需的数据流并克隆为独立引用
    pub fn new(target_path: String) -> Self {
        let swapl = global_swapl();
        let clr_stream = Arc::clone(&swapl.colors);
        let cld_stream = Arc::clone(&swapl.clouds);
        let files = vec![];

        Self {
            clr_stream,
            cld_stream,
            target_path,
            image_path: None,
            pcd_path: None,
            files,
            frame_limit: None,
            current_index: 0,
            images: Vec::new(),
            clouds: Vec::new(),
        }
    }

    /// 创建使用独立路径的数据加载器
    ///
    /// 适用于图像和点云数据存放在不同目录的情况
    pub fn new_independent(image_path: String, pcd_path: String) -> Self {
        let swapl = global_swapl();
        let clr_stream = Arc::clone(&swapl.colors);
        let cld_stream = Arc::clone(&swapl.clouds);
        let files = vec![];

        Self {
            clr_stream,
            cld_stream,
            target_path: String::new(),
            image_path: Some(image_path),
            pcd_path: Some(pcd_path),
            files,
            frame_limit: None,
            current_index: 0,
            images: Vec::new(),
            clouds: Vec::new(),
        }
    }

    /// 设置最多加载多少帧（文件对数），超过则截断
    pub fn set_frame_limit(&mut self, limit: usize) -> &mut Self {
        self.frame_limit = Some(limit);
        self
    }

    /// 按比例设置帧数（0.1 = 加载 10% 的文件）
    pub fn set_frame_ratio(&mut self, ratio: f32) -> &mut Self {
        if self.files.is_empty() {
            // 尚未列出文件时先取整
            self.frame_limit = Some(ratio as usize);
            return self;
        }
        let n = (self.files.len() as f32 * ratio).round() as usize;
        self.frame_limit = Some(n.max(1));
        self
    }

    /// 均匀下采样点云到最多 `max_count` 个点
    pub fn downsample(&self, points: &[[f32; 3]], max_count: usize) -> Vec<[f32; 3]> {
        if points.len() <= max_count {
            return points.to_vec();
        }
        let step = (points.len() / max_count).max(1);
        points.iter()
            .enumerate()
            .filter(|(i, _)| i % step == 0)
            .map(|(_, p)| *p)
            .collect()
    }

    /// 列出目标路径中的所有文件
    ///
    /// 返回在 camera 和 lidar 目录中都存在的文件对列表，
    /// 每个元素包含 [文件名，文件名]，为了未来扩展兼容
    pub fn list_files(&self) -> io::Result<Vec<Vec<String>>> {
        // 检查是否使用独立路径模式
        if let (Some(image_path), Some(pcd_path)) = (&self.image_path, &self.pcd_path) {
            return self.list_files_independent(image_path, pcd_path);
        }

        // 否则使用旧格式
        let lidar_path = format!("{}/lidar", self.target_path);
        let camera_path = format!("{}/camera", self.target_path);

        // 读取 camera 目录中的所有文件，构建文件干名到完整文件名的映射
        let clr_files: HashMap<String, String> = fs::read_dir(camera_path)?
            .filter_map(|entry| {
                entry.ok().and_then(|e| {
                    e.path()
                        .file_name()
                        .map(|name| name.to_string_lossy().into_owned())
                        .and_then(|name| {
                            let full_name = name.clone();
                            std::path::Path::new(&name)
                                .file_stem()
                                .map(|s| s.to_string_lossy().into_owned())
                                .map(|stem| (stem, full_name))
                        })
                })
            })
            .collect();

        // 读取 lidar 目录中的所有文件，构建文件干名到完整文件名的映射
        let cld_files: HashMap<String, String> = fs::read_dir(lidar_path)?
            .filter_map(|entry| {
                entry.ok().and_then(|e| {
                    e.path()
                        .file_name()
                        .map(|name| name.to_string_lossy().into_owned())
                        .and_then(|name| {
                            let full_name = name.clone();
                            std::path::Path::new(&name)
                                .file_stem()
                                .map(|s| s.to_string_lossy().into_owned())
                                .map(|stem| (stem, full_name))
                        })
                })
            })
            .collect();

        // 使用函数式方法找出两个目录中基本名称相同的文件对
        let target: Vec<Vec<String>> = clr_files
            .into_iter()
            .filter_map(|(stem, camera_file)| {
                cld_files
                    .get(&stem)
                    .map(|lidar_file| vec![camera_file, lidar_file.clone()])
            })
            .collect();

        Ok(target)
    }

    /// 列出独立路径中的文件（独立路径模式）
    fn list_files_independent(&self, image_path: &str, pcd_path: &str) -> io::Result<Vec<Vec<String>>> {
        // 读取图像目录中的所有文件
        let image_files: Vec<String> = fs::read_dir(image_path)?
            .filter_map(|entry| {
                entry.ok().and_then(|e| {
                    e.path()
                        .file_name()
                        .map(|name| name.to_string_lossy().into_owned())
                })
            })
            .collect();

        // 读取点云目录中的所有文件
        let pcd_files: Vec<String> = fs::read_dir(pcd_path)?
            .filter_map(|entry| {
                entry.ok().and_then(|e| {
                    e.path()
                        .file_name()
                        .map(|name| name.to_string_lossy().into_owned())
                })
            })
            .collect();

        // 尝试按文件名匹配（去除扩展名后比较）
        let mut matched_pairs = Vec::new();
        let mut used_pcd = vec![false; pcd_files.len()];

        for image_file in &image_files {
            // 获取图像文件的基本名（不含扩展名）
            if let Some(image_stem) = std::path::Path::new(image_file)
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
            {
                // 查找匹配的点云文件
                for (i, pcd_file) in pcd_files.iter().enumerate() {
                    if !used_pcd[i] {
                        if let Some(pcd_stem) = std::path::Path::new(pcd_file)
                            .file_stem()
                            .map(|s| s.to_string_lossy().into_owned())
                        {
                            if image_stem == pcd_stem {
                                matched_pairs.push(vec![image_file.clone(), pcd_file.clone()]);
                                used_pcd[i] = true;
                                break;
                            }
                        }
                    }
                }
            }
        }

        // 如果没有匹配的文件对，则按顺序配对
        if matched_pairs.is_empty() {
            let min_len = image_files.len().min(pcd_files.len());
            for i in 0..min_len {
                matched_pairs.push(vec![image_files[i].clone(), pcd_files[i].clone()]);
            }
        }

        Ok(matched_pairs)
    }

    /// 按需加载下一帧数据（从内存缓冲写入流，无磁盘 I/O）
    ///
    /// 返回 true 表示还有数据，false 表示已读完。
    /// 首次调用会自动触发预加载（从磁盘读取全部文件到内存）。
    pub async fn load_next(&mut self) -> io::Result<bool> {
        // 首次调用：列文件并预加载到内存
        if self.images.is_empty() {
            if self.files.is_empty() {
                self.files = self.list_files()?;
                self.apply_frame_limit();
            }
            let n = self.files.len();
            self.images.reserve(n);
            self.clouds.reserve(n);
            info!("预加载 {} 帧数据到内存...", n);
            for file_pair in &self.files {
                let (camera_file, lidar_file) = self.build_paths(file_pair);
                let img = load_image(&camera_file).unwrap_or_else(|_| {
                    // 加载失败时创建一个空白图像
                    use image::RgbaImage;
                    DynamicImage::ImageRgba8(RgbaImage::new(640, 480))
                });
                let lifra = DynReader::open(&lidar_file)
                    .map(|mut r| load_cloud(&mut r))
                    .unwrap_or_default();
                self.images.push(img);
                self.clouds.push(lifra);
            }
            info!("预加载完成，共 {} 帧", n);
        }

        if self.current_index >= self.images.len() {
            return Ok(false);
        }

        let idx = self.current_index;
        self.current_index += 1;

        // 从内存写入流（无 I/O 等待）
        let mut clr_stream = self.clr_stream.lock().await;
        let _ = clr_stream.write(self.images[idx].clone());
        drop(clr_stream);

        let mut cld_stream = self.cld_stream.lock().await;
        let _ = cld_stream.write(self.clouds[idx].clone());

        Ok(true)
    }

    /// 构建文件路径（内部辅助）
    fn build_paths(&self, file_pair: &[String]) -> (String, String) {
        if self.image_path.is_some() && self.pcd_path.is_some() {
            (
                format!("{}/{}", self.image_path.as_ref().unwrap(), &file_pair[0]),
                format!("{}/{}", self.pcd_path.as_ref().unwrap(), &file_pair[1]),
            )
        } else {
            (
                format!("{}/camera/{}", self.target_path, &file_pair[0]),
                format!("{}/lidar/{}", self.target_path, &file_pair[1]),
            )
        }
    }

    /// 预加载全部数据到内存缓冲，不写入流。
    /// 预加载后可用 `load_next()` 从内存逐帧写入流。
    pub async fn load(&mut self) -> io::Result<()> {
        if !self.images.is_empty() {
            return Ok(()); // 已预加载
        }
        if self.files.is_empty() {
            self.files = self.list_files()?;
            self.apply_frame_limit();
        }
        let n = self.files.len();
        self.images.reserve(n);
        self.clouds.reserve(n);
        info!("预加载 {} 帧数据到内存...", n);
        for file_pair in &self.files {
            let (camera_file, lidar_file) = self.build_paths(file_pair);
            let img = load_image(&camera_file).unwrap_or_else(|_| {
                use image::RgbaImage;
                DynamicImage::ImageRgba8(RgbaImage::new(640, 480))
            });
            let lifra = DynReader::open(&lidar_file)
                .map(|mut r| load_cloud(&mut r))
                .unwrap_or_default();
            self.images.push(img);
            self.clouds.push(lifra);
        }
        info!("预加载完成，共 {} 帧", n);
        Ok(())
    }

    /// 循环加载数据文件到流中（20 帧/秒）
    /// 缓冲区满时静默跳过当前帧，等待消费者读取
    pub async fn load_loop(&mut self) -> io::Result<()> {
        if self.files.is_empty() {
            self.files = self.list_files()?;
            self.apply_frame_limit();
        }

        loop {
            for file_pair in &self.files {
                // 构建完整的文件路径
                let (camera_file, lidar_file) = if self.image_path.is_some() && self.pcd_path.is_some() {
                    (
                        format!("{}/{}", self.image_path.as_ref().unwrap(), &file_pair[0]),
                        format!("{}/{}", self.pcd_path.as_ref().unwrap(), &file_pair[1]),
                    )
                } else {
                    (
                        format!("{}/camera/{}", self.target_path, &file_pair[0]),
                        format!("{}/lidar/{}", self.target_path, &file_pair[1]),
                    )
                };

                // 先检查流是否还能写入
                {
                    let mut clr_stream = self.clr_stream.lock().await;
                    if clr_stream.get_write_mut().is_err() {
                        thread::sleep(Duration::from_millis(5));
                        continue;
                    }
                }

                // 先执行耗时的 I/O 操作，不持有锁
                let image_result = load_image(&camera_file);
                let cloud_result =
                    DynReader::open(&lidar_file).map(|mut reader| load_cloud(&mut reader));

                // 然后获取锁并快速写入数据
                {
                    let mut clr_stream = self.clr_stream.lock().await;
                    match image_result {
                        Ok(image) => { let _ = clr_stream.write(image); }
                        Err(e) => {
                            error!("加载图像 {} 时出错：{}", camera_file, e);
                            let _ = clr_stream.write_direct(|slot| *slot = None);
                        }
                    }
                }

                {
                    let mut cld_stream = self.cld_stream.lock().await;
                    match cloud_result {
                        Ok(lifra) => { let _ = cld_stream.write(lifra); }
                        Err(e) => {
                            error!("打开 PCD 文件 {} 时出错：{}", lidar_file, e);
                            let _ = cld_stream.write_direct(|slot| *slot = None);
                        }
                    }
                }

                // 延迟以达到20帧/秒的速度 (50ms per frame)
                thread::sleep(Duration::from_millis(50));
            }
        }
    }

    /// 根据 frame_limit 截断文件列表（在 load / load_loop 中自动调用）
    fn apply_frame_limit(&mut self) {
        if let Some(limit) = self.frame_limit {
            if limit < self.files.len() {
                info!("帧数限制: {} (共 {} 文件，取前 {})", limit, self.files.len(), limit);
                self.files.truncate(limit);
            }
        }
    }
}
