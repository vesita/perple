use image::DynamicImage;
use log::{error, info};
use log::{error, info};
use pcd_rs::DynReader;
use std::collections::HashMap;
use std::num::NonZero;
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

fn build_file_paths(
    target_path: &str,
    image_path: Option<&str>,
    pcd_path: Option<&str>,
    file_pair: &[String],
) -> (String, String) {
    if let (Some(ip), Some(pp)) = (image_path, pcd_path) {
        (
            format!("{}/{}", ip, &file_pair[0]),
            format!("{}/{}", pp, &file_pair[1]),
        )
    } else {
        (
            format!("{}/camera/{}", target_path, &file_pair[0]),
            format!("{}/lidar/{}", target_path, &file_pair[1]),
        )
    }
}

/// 并行预加载全部帧数据（图像 + 点云）
///
/// 将文件对按线程数分块，每块在一个线程内顺序加载，
/// 整体并行执行。返回 `(images, clouds)` 保持原始顺序。
fn preload_parallel(
    files: &[Vec<String>],
    target_path: &str,
    image_path: Option<&str>,
    pcd_path: Option<&str>,
) -> (Vec<DynamicImage>, Vec<Vec<[f32; 3]>>)
{
    let n = files.len();
    if n == 0 {
        return (Vec::new(), Vec::new());
    }

    let num_threads = thread::available_parallelism()
        .map(NonZero::get)
        .unwrap_or(4)
        .min(8)
        .min(n);

    info!("并行预加载 {} 帧，使用 {} 线程", n, num_threads);

    let chunk_size = (n + num_threads - 1) / num_threads;

    thread::scope(|s| {
        let mut handles = Vec::with_capacity(num_threads);

        for chunk in files.chunks(chunk_size) {
            handles.push(s.spawn(move || -> (Vec<DynamicImage>, Vec<Vec<[f32; 3]>>) {
                let mut images = Vec::with_capacity(chunk.len());
                let mut clouds = Vec::with_capacity(chunk.len());
                for file_pair in chunk {
                    let (camera_file, lidar_file) = build_file_paths(target_path, image_path, pcd_path, file_pair);
                    let img = load_image(&camera_file).unwrap_or_else(|_| {
                        use image::RgbaImage;
                        DynamicImage::ImageRgba8(RgbaImage::new(640, 480))
                    });
                    let cloud = DynReader::open(&lidar_file)
                        .map(|mut r| load_cloud(&mut r))
                        .unwrap_or_default();
                    images.push(img);
                    clouds.push(cloud);
                }
                (images, clouds)
            }));
        }

        let mut all_images = Vec::with_capacity(n);
        let mut all_clouds = Vec::with_capacity(n);
        for handle in handles {
            let (images, clouds) = handle.join().expect("预加载线程 panic");
            all_images.extend(images);
            all_clouds.extend(clouds);
        }
        (all_images, all_clouds)
    })
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

    /// 返回已加载或已列出的帧数
    pub fn frame_count(&self) -> usize {
        if !self.images.is_empty() {
            self.images.len()
        } else {
            self.files.len()
        }
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

        if !std::path::Path::new(&camera_path).is_dir() {
            log::warn!("camera 目录不存在: {}，跳过数据加载", camera_path);
            return Ok(Vec::new());
        }
        if !std::path::Path::new(&lidar_path).is_dir() {
            log::warn!("lidar 目录不存在: {}，跳过数据加载", lidar_path);
            return Ok(Vec::new());
        }

        // 读取 camera 目录中的所有文件，构建文件干名到完整文件名的映射
        let clr_files: HashMap<String, String> = fs::read_dir(&camera_path)?
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
        let cld_files: HashMap<String, String> = fs::read_dir(&lidar_path)?
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

        let n_camera = clr_files.len();
        let n_lidar = cld_files.len();

        // 使用函数式方法找出两个目录中基本名称相同的文件对
        let mut target: Vec<Vec<String>> = clr_files
            .into_iter()
            .filter_map(|(stem, camera_file)| {
                cld_files
                    .get(&stem)
                    .map(|lidar_file| vec![camera_file, lidar_file.clone()])
            })
            .collect();

        // 按文件名排序，确保帧顺序一致
        target.sort_by(|a, b| a[0].cmp(&b[0]));

        let n_matched = target.len();
        if n_matched < n_camera || n_matched < n_lidar {
            log::warn!(
                "文件匹配：camera={} lidar={} 匹配={}（缺失 {} 帧）",
                n_camera, n_lidar, n_matched,
                n_lidar.saturating_sub(n_matched),
            );
        } else {
            info!("文件匹配：camera={} lidar={} 匹配={}", n_camera, n_lidar, n_matched);
        }

        Ok(target)
    }

    /// 列出独立路径中的文件（独立路径模式）
    fn list_files_independent(&self, image_path: &str, pcd_path: &str) -> io::Result<Vec<Vec<String>>> {
        if !std::path::Path::new(image_path).is_dir() {
            log::warn!("image 目录不存在: {}，跳过数据加载", image_path);
            return Ok(Vec::new());
        }
        if !std::path::Path::new(pcd_path).is_dir() {
            log::warn!("pcd 目录不存在: {}，跳过数据加载", pcd_path);
            return Ok(Vec::new());
        }

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

        let n_image = image_files.len();
        let n_pcd = pcd_files.len();

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

        // 按文件名排序，确保帧顺序一致
        matched_pairs.sort_by(|a, b| a[0].cmp(&b[0]));

        let n_matched = matched_pairs.len();
        if n_matched < n_image || n_matched < n_pcd {
            log::warn!(
                "文件匹配：image={} pcd={} 匹配={}（缺失 {} 帧）",
                n_image, n_pcd, n_matched,
                n_pcd.saturating_sub(n_matched),
            );
        } else {
            info!("文件匹配：image={} pcd={} 匹配={}", n_image, n_pcd, n_matched);
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
            info!("预加载 {} 帧数据到内存...", n);
            let (images, clouds) = preload_parallel(
                &self.files,
                &self.target_path,
                self.image_path.as_deref(),
                self.pcd_path.as_deref(),
            );
            self.images = images;
            self.clouds = clouds;
            info!("预加载完成，共 {} 帧", n);
        }

        if self.current_index >= self.images.len() {
            return Ok(false);
        }

        let idx = self.current_index;
        self.current_index += 1;

        // 从内存写入流（无 I/O 等待）
        let mut clr_stream = self.clr_stream.lock().unwrap();
        if let Err(e) = clr_stream.write(self.images[idx].clone()) {
            log::warn!("图像流写入失败 (帧 {}): {:?}", idx, e);
        }
        drop(clr_stream);

        let mut cld_stream = self.cld_stream.lock().unwrap();
        if let Err(e) = cld_stream.write(self.clouds[idx].clone()) {
            log::warn!("点云流写入失败 (帧 {}): {:?}", idx, e);
        }

        Ok(true)
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
        info!("预加载 {} 帧数据到内存...", n);
        let (images, clouds) = preload_parallel(
            &self.files,
            &self.target_path,
            self.image_path.as_deref(),
            self.pcd_path.as_deref(),
        );
        self.images = images;
        self.clouds = clouds;
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
                    let mut clr_stream = self.clr_stream.lock().unwrap();
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
                    let mut clr_stream = self.clr_stream.lock().unwrap();
                    match image_result {
                        Ok(image) => { let _ = clr_stream.write(image); }
                        Err(e) => {
                            error!("加载图像 {} 时出错：{}", camera_file, e);
                            let _ = clr_stream.write_direct(|slot| *slot = None);
                        }
                    }
                }

                {
                    let mut cld_stream = self.cld_stream.lock().unwrap();
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
