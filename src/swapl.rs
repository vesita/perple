use std::sync::{Arc, LazyLock, Mutex};

use image::DynamicImage;


use crate::cloud::CldBud;
use crate::color::ClrBud;
use crate::tracker::output::Target;
use crate::utils::sight::Sight;
use crate::utils::stream::{DualBuf, Eap, Stream, new_dual_buf, new_eap};

// 使用LazyLock创建全局单例
static GLOBAL_SWAPL: LazyLock<Swapl> = LazyLock::new(Swapl::new);

/// 获取全局Swapl实例
pub fn global_swapl() -> &'static Swapl {
    &GLOBAL_SWAPL
}

/// 系统数据交换中枢
///
/// Swapl作为整个系统的数据中枢，负责管理所有的数据流。
/// 其他模块通过访问Swapl来进行数据交互，实现了松耦合的架构设计。
/// 所有的数据流都是线程安全的(Eap<Stream<T>>)，可以在多个线程间安全共享。
///
/// 跨阶段说明：检测阶段（Lidar/Camera）写入 producer，后融合阶段（Fuse/Tracker）
/// 读取 consumer，通过 DualBuffer::swap() 在串行点切换，消除跨阶段锁竞争。
pub struct Swapl {
    /// 点云数据输入流
    pub clouds: Eap<Stream<Vec<[f32; 3]>>>,
    pub clouds_out: Eap<Stream<Vec<[f32; 3]>>>,
    /// 地面滤除后的点云（双缓冲：检测写producer / 跟踪读consumer）
    pub clouds_filtered: DualBuf<Vec<[f32; 3]>>,
    /// 点云检测结果输出流（双缓冲：分类写producer / Fuse读consumer）
    pub cld_buds_raw: DualBuf<Vec<CldBud>>,
    /// 点云检测结果输出流（经 Fuse 融合后，Fuse写 / Tracker读，同阶段无竞争）
    pub cld_objs: Eap<Stream<Vec<CldBud>>>,
    /// 地面 Bud 独立流（双缓冲：检测写producer / 后融合读consumer）
    pub ground_buds: DualBuf<Vec<CldBud>>,
    /// 墙体 Bud 独立流（双缓冲：检测写producer / 后融合读consumer）
    pub wall_buds: DualBuf<Vec<CldBud>>,
    /// 图像数据输入流
    pub colors: Eap<Stream<DynamicImage>>,
    /// 图像检测结果输出流（双缓冲：Camera写producer / Fuse读consumer）
    pub clr_objs: DualBuf<Vec<ClrBud>>,
    /// 3D投影结果输出流
    pub sights: Eap<Stream<Vec<Sight>>>,
    /// 目标检测结果输出流
    pub targets: Eap<Stream<Vec<Target>>>,
    /// 地面平面方程流 [a, b, c, d] (a*x + b*y + c*z + d = 0)
    pub ground_plane: Eap<Stream<[f32; 4]>>,
    /// 最新 YOLO 检测结果（供 classify YOLO refine 读取，非 DualBuf 避免跨阶段污染）
    pub last_yolo: Arc<Mutex<Vec<ClrBud>>>,
}

impl Swapl {
    /// 创建一个新的数据交换中枢
    pub fn new() -> Self {
        Swapl {
            clouds: new_eap(Stream::new()),
            clouds_out: new_eap(Stream::new()),
            clouds_filtered: new_dual_buf(),
            cld_buds_raw: new_dual_buf(),
            cld_objs: new_eap(Stream::new()),
            ground_buds: new_dual_buf(),
            wall_buds: new_dual_buf(),
            colors: new_eap(Stream::new()),
            clr_objs: new_dual_buf(),
            sights: new_eap(Stream::new()),
            targets: new_eap(Stream::new()),
            ground_plane: new_eap(Stream::new()),
            last_yolo: Arc::new(Mutex::new(Vec::new())),
        }
    }
}