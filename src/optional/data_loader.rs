use image::DynamicImage;
use log::{error, info, warn};
use pcd_rs::DynReader;
use std::collections::HashMap;
use std::{fs, io, sync::Arc, thread, time::Duration};

use crate::{
    color::load_image,
    swapl::global_swapl,
    utils::stream::{Eap, Stream, StreamError},
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

    /// 加载单个数据文件到流中
    pub async fn load(&mut self) -> io::Result<()> {
        info!("开始加载数据...");
        if self.files.is_empty() {
            self.files = self.list_files()?;
        }

        for file_pair in &self.files {
            // 构建完整的文件路径
            let (camera_file, lidar_file) = if self.image_path.is_some() && self.pcd_path.is_some() {
                // 独立路径模式
                (
                    format!("{}/{}", self.image_path.as_ref().unwrap(), &file_pair[0]),
                    format!("{}/{}", self.pcd_path.as_ref().unwrap(), &file_pair[1]),
                )
            } else {
                // 旧格式
                (
                    format!("{}/camera/{}", self.target_path, &file_pair[0]),
                    format!("{}/lidar/{}", self.target_path, &file_pair[1]),
                )
            };

            // 获取数据流锁
            let mut clr_stream = self.clr_stream.lock().await;
            let mut cld_stream = self.cld_stream.lock().await;

            // 加载图像文件并写入流
            match load_image(&camera_file) {
                Ok(image) => {
                    if let Err(StreamError::BufferFull) = clr_stream.write(image) {
                        warn!("颜色流缓冲区已满");
                    }
                }
                Err(e) => {
                    error!("加载图像 {} 时出错：{}", camera_file, e);
                    if let Err(StreamError::BufferFull) =
                        clr_stream.write_direct(|slot| *slot = None)
                    {
                        warn!("写入 None 时颜色流缓冲区已满");
                    }
                }
            }

            // 解析 PCD 文件并写入流
            match DynReader::open(&lidar_file) {
                Ok(mut reader) => {
                    let lifra = load_cloud(&mut reader);
                    if let Err(StreamError::BufferFull) = cld_stream.write(lifra) {
                        warn!("点云流缓冲区已满");
                    }
                }
                Err(e) => {
                    error!("打开 PCD 文件 {} 时出错：{}", lidar_file, e);
                    if let Err(StreamError::BufferFull) =
                        cld_stream.write_direct(|slot| *slot = None)
                    {
                        warn!("写入 None 时点云流缓冲区已满");
                    }
                }
            }
        }

        info!("数据加载完成");
        Ok(())
    }

    /// 循环加载数据文件到流中
    /// 该方法会按照 20 帧的速度无限循环加载数据，除非遇到 I/O 错误
    pub async fn load_loop(&mut self) -> io::Result<()> {
        if self.files.is_empty() {
            self.files = self.list_files()?;
        }

        loop {
            for file_pair in &self.files {
                // 构建完整的文件路径
                let (camera_file, lidar_file) = if self.image_path.is_some() && self.pcd_path.is_some() {
                    // 独立路径模式
                    (
                        format!("{}/{}", self.image_path.as_ref().unwrap(), &file_pair[0]),
                        format!("{}/{}", self.pcd_path.as_ref().unwrap(), &file_pair[1]),
                    )
                } else {
                    // 旧格式
                    (
                        format!("{}/camera/{}", self.target_path, &file_pair[0]),
                        format!("{}/lidar/{}", self.target_path, &file_pair[1]),
                    )
                };

                // 先执行耗时的 I/O 操作，不持有锁
                let image_result = load_image(&camera_file);
                let cloud_result =
                    DynReader::open(&lidar_file).map(|mut reader| load_cloud(&mut reader));

                // 然后获取锁并快速写入数据
                {
                    let mut clr_stream = self.clr_stream.lock().await;
                    match image_result {
                        Ok(image) => {
                            if let Err(StreamError::BufferFull) = clr_stream.write(image) {
                                warn!("颜色流缓冲区已满");
                            }
                        }
                        Err(e) => {
                            error!("加载图像 {} 时出错：{}", camera_file, e);
                            if let Err(StreamError::BufferFull) =
                                clr_stream.write_direct(|slot| *slot = None)
                            {
                                warn!("写入 None 时颜色流缓冲区已满");
                            }
                        }
                    }
                } // 在这里自动释放 clr_stream 的锁

                {
                    let mut cld_stream = self.cld_stream.lock().await;
                    match cloud_result {
                        Ok(lifra) => {
                            if let Err(StreamError::BufferFull) = cld_stream.write(lifra) {
                                warn!("点云流缓冲区已满");
                            }
                        }
                        Err(e) => {
                            error!("打开 PCD 文件 {} 时出错：{}", lidar_file, e);
                            if let Err(StreamError::BufferFull) =
                                cld_stream.write_direct(|slot| *slot = None)
                            {
                                warn!("写入 None 时点云流缓冲区已满");
                            }
                        }
                    }
                } // 在这里自动释放 cld_stream 的锁

                // 延迟以达到20帧/秒的速度 (50ms per frame)
                thread::sleep(Duration::from_millis(50));
            }
        }
    }
}
