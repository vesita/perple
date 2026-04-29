/// Perple ROS 节点入口
///
/// 数据流：
///   ROS /perple/input/cloud → RosBridge → Swapl → Perple pipeline → Swapl → RosBridge → ROS topics
///
/// ROS 模式启动：
///   cargo run --features ros1
///
/// 数据来源：ROS 话题订阅（无需文件加载），与 visualize 示例的文件模式不同。

#[cfg(feature = "ros1")]
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(feature = "ros1")]
use std::sync::Arc;

#[cfg(feature = "ros1")]
use perple::ros_bridge::{RosBridge, RosBridgeConfig};
#[cfg(feature = "ros1")]
use perple::Perple;

#[cfg(feature = "ros1")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();

    log::info!("Perple ROS 节点启动中...");

    // ── ROS 桥接初始化（创建发布器 + 订阅器，内部调用 rosrust::init） ──
    let mut bridge = RosBridge::new(RosBridgeConfig::default());
    bridge.init()?;
    log::info!("ROS 桥接初始化完成，发布器/订阅器已创建");

    // ── 创建 tokio 运行时供 Perple 管线使用 ──
    let rt = tokio::runtime::Runtime::new()?;

    // ── 启动 Perple 管线（LiDAR 聚类 + 跟踪 + 融合） ──
    // Perple::run() 以 Signal 模式启动各子循环（无限运行），
    // 数据经 Swapl 全局共享，无需手动文件加载。
    let _perple_handle = rt.spawn(async {
        let mut perple = Perple::new();
        if let Err(e) = perple.run().await {
            log::error!("Perple 管线错误: {}", e);
        }
    });
    log::info!("Perple 检测管线已启动");

    // ── Ctrl+C 信号处理（tokio 后台线程） ──
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    std::thread::spawn(move || {
        // 创建独立 tokio 运行时等待 Ctrl+C
        if let Ok(rt) = tokio::runtime::Runtime::new() {
            rt.block_on(async {
                tokio::signal::ctrl_c().await.ok();
            });
        }
        log::info!("收到 Ctrl+C，正在关闭...");
        r.store(false, Ordering::SeqCst);
    });

    // ── ROS spin 线程：在后台处理订阅回调 ──
    // rosrust 0.9 没有 spin_once()，spin() 会阻塞处理回调直到节点关闭
    std::thread::spawn(|| {
        rosrust::spin();
    });

    // ── 主循环：发布（20 Hz） ──
    let rate = std::time::Duration::from_millis(50);
    log::info!("进入主循环 (20 Hz)");

    while running.load(Ordering::SeqCst) {
        // 发布跟踪目标 / 检测框 / 自车速度
        bridge.publish_all();

        std::thread::sleep(rate);
    }

    // ── 优雅关闭 ──
    // 丢弃 tokio 运行时 → 自动取消 Perple 管线任务
    drop(rt);
    log::info!("Perple ROS 节点已关闭");
    Ok(())
}

#[cfg(not(feature = "ros1"))]
fn main() {
    println!("欢迎使用Perple目标检测工具！");
    println!();
    println!("可用示例:");
    println!("  cargo run --example counter         # 目标计数示例");
    println!("  cargo run --example color_detection # 颜色检测和可视化示例");
    println!("  cargo run --example image_test      # 图像测试示例");
    println!();
    println!("ROS1 模式: cargo run --features ros1");
}
