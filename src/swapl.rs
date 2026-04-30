use std::sync::LazyLock;

use image::DynamicImage;


use crate::cloud::CldBud;
use crate::color::ClrBud;
use crate::tracker::output::Target;
use crate::utils::sight::Sight;
use crate::utils::stream::{Eap, Stream, new_eap};

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
/// 所有的数据流都是线程安全的(Eap<Stream<T>>>)，可以在多个线程间安全共享。
pub struct Swapl {
    /// 点云数据输入流
    pub clouds: Eap<Stream<Vec<[f32; 3]>>>,
    pub clouds_out: Eap<Stream<Vec<[f32; 3]>>>,
    /// 地面滤除后的点云（用于跟踪器点云投票）
    pub clouds_filtered: Eap<Stream<Vec<[f32; 3]>>>,
    /// 点云检测结果输出流（原始，未融合）
    pub cld_buds_raw: Eap<Stream<Vec<CldBud>>>,
    /// 点云检测结果输出流（经 Fuse 融合后）
    pub cld_objs: Eap<Stream<Vec<CldBud>>>,
    /// 图像数据输入流
    pub colors: Eap<Stream<DynamicImage>>,
    /// 图像检测结果输出流
    pub clr_objs: Eap<Stream<Vec<ClrBud>>>,
    /// 3D投影结果输出流
    pub sights: Eap<Stream<Vec<Sight>>>,
    /// 目标检测结果输出流
    pub targets: Eap<Stream<Vec<Target>>>,
    /// 地面平面方程流 [a, b, c, d] (a*x + b*y + c*z + d = 0)
    pub ground_plane: Eap<Stream<[f32; 4]>>,
}

impl Swapl { 
    /// 创建一个新的数据交换中枢
    pub fn new() -> Self {
        Swapl {
            clouds: new_eap(Stream::new()),
            clouds_out: new_eap(Stream::new()),
            clouds_filtered: new_eap(Stream::new()),
            cld_buds_raw: new_eap(Stream::new()),
            cld_objs: new_eap(Stream::new()),
            colors: new_eap(Stream::new()),
            clr_objs: new_eap(Stream::new()),
            sights: new_eap(Stream::new()),
            targets: new_eap(Stream::new()),
            ground_plane: new_eap(Stream::new()),
        }
    }
}