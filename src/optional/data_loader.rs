use image::DynamicImage;
use log::{error, info};
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
/// DataLoader负责从文件系统加载2D和3D检测数据，并将它们写入相应的数据流中。
pub struct DataLoader {
    /// 2D检测结果数据流
    clr_stream: Eap<Stream<DynamicImage>>,
    /// 3D检测结果数据流
    cld_stream: Eap<Stream<Vec<[f32; 3]>>>,
    /// 数据文件路径
    target_path: String,

    files: Vec<Vec<String>>,
}

impl DataLoader {
    /// 创建一个新的数据加载器
    ///
    /// 通过Swapl数据中枢获取所需的数据流并克隆为独立引用
    pub fn new(target_path: String) -> Self {
        let swapl = global_swapl();
        let clr_stream = Arc::clone(&swapl.colors);
        let cld_stream = Arc::clone(&swapl.clouds);
        let files = vec![];

        Self {
            clr_stream,
            cld_stream,
            target_path,
            files,
        }
    }

    /// 列出目标路径中的所有文件
    ///
    /// 返回在camera和lidar目录中都存在的文件对列表，
    /// 每个元素包含[文件名, 文件名]，为了未来扩展兼容
    pub fn list_files(&self) -> io::Result<Vec<Vec<String>>> {
        let lidar_path = format!("{}/lidar", self.target_path);
        let camera_path = format!("{}/camera", self.target_path);

        // 读取camera目录中的所有文件，构建文件干名到完整文件名的映射
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

        // 读取lidar目录中的所有文件，构建文件干名到完整文件名的映射
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

    /// 加载单个数据文件到流中
    pub async fn load(&mut self) -> io::Result<()> {
        info!("开始加载数据...");
        if self.files.is_empty() {
            self.files = self.list_files()?;
        }

        for file_pair in &self.files {
            let camera_file = format!("{}/camera/{}", self.target_path, &file_pair[0]);
            let lidar_file = format!("{}/lidar/{}", self.target_path, &file_pair[1]);

            // 获取数据流锁
            let mut clr_stream = self.clr_stream.lock().await;
            let mut cld_stream = self.cld_stream.lock().await;

            // 加载图像文件并写入流
            match load_image(&camera_file) {
                Ok(image) => {
                    if let Err(StreamError::BufferFull) = clr_stream.write(image) {
                        error!("颜色流缓冲区已满");
                    }
                }
                Err(e) => {
                    error!("加载图像 {} 时出错：{}", camera_file, e);
                    if let Err(StreamError::BufferFull) =
                        clr_stream.write_direct(|slot| *slot = None)
                    {
                        error!("写入None时颜色流缓冲区已满");
                    }
                }
            }

            // 解析 PCD 文件并写入流
            match DynReader::open(&lidar_file) {
                Ok(mut reader) => {
                    let lifra = load_cloud(&mut reader);
                    if let Err(StreamError::BufferFull) = cld_stream.write(lifra) {
                        error!("点云流缓冲区已满");
                    }
                }
                Err(e) => {
                    error!("打开PCD文件 {} 时出错：{}", lidar_file, e);
                    if let Err(StreamError::BufferFull) =
                        cld_stream.write_direct(|slot| *slot = None)
                    {
                        error!("写入None时点云流缓冲区已满");
                    }
                }
            }
        }

        info!("数据加载完成");
        Ok(())
    }

    /// 循环加载数据文件到流中
    /// 该方法会按照20帧的速度无限循环加载数据，除非遇到I/O错误
    pub async fn load_loop(&mut self) -> io::Result<()> {
        if self.files.is_empty() {
            self.files = self.list_files()?;
        }

        loop {
            for file_pair in &self.files {
                let camera_file = format!("{}/camera/{}", self.target_path, &file_pair[0]);
                let lidar_file = format!("{}/lidar/{}", self.target_path, &file_pair[1]);

                // 先执行耗时的I/O操作，不持有锁
                let image_result = load_image(&camera_file);
                let cloud_result =
                    DynReader::open(&lidar_file).map(|mut reader| load_cloud(&mut reader));

                // 然后获取锁并快速写入数据
                {
                    let mut clr_stream = self.clr_stream.lock().await;
                    match image_result {
                        Ok(image) => {
                            if let Err(StreamError::BufferFull) = clr_stream.write(image) {
                                error!("颜色流缓冲区已满");
                            }
                        }
                        Err(e) => {
                            error!("加载图像 {} 时出错：{}", camera_file, e);
                            if let Err(StreamError::BufferFull) =
                                clr_stream.write_direct(|slot| *slot = None)
                            {
                                error!("写入None时颜色流缓冲区已满");
                            }
                        }
                    }
                } // 在这里自动释放clr_stream的锁

                {
                    let mut cld_stream = self.cld_stream.lock().await;
                    match cloud_result {
                        Ok(lifra) => {
                            if let Err(StreamError::BufferFull) = cld_stream.write(lifra) {
                                error!("点云流缓冲区已满");
                            }
                        }
                        Err(e) => {
                            error!("打开PCD文件 {} 时出错：{}", lidar_file, e);
                            if let Err(StreamError::BufferFull) =
                                cld_stream.write_direct(|slot| *slot = None)
                            {
                                error!("写入None时点云流缓冲区已满");
                            }
                        }
                    }
                } // 在这里自动释放cld_stream的锁

                // 延迟以达到20帧/秒的速度 (50ms per frame)
                thread::sleep(Duration::from_millis(50));
            }
        }
    }
}
