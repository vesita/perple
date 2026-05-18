use super::*;
use tokio::time::{sleep, Duration};

/// 测试 Perple 实例的创建
#[test]
fn test_perple_new() {
    let perple = Perple::new();

    // 验证所有字段都已初始化
    assert!(perple.camera.try_lock().is_ok());
    assert!(perple.lidar.try_lock().is_ok());
    assert!(perple.tracker.try_lock().is_ok());
}

/// 测试 run 方法能否成功启动所有循环
#[tokio::test]
async fn test_run_starts_all_loops() {
    let mut perple = Perple::new();

    // 调用 run 方法
    let result = perple.run().await;

    // 验证 run 成功返回
    assert!(result.is_ok(), "run() 应该成功返回");

    // 验证所有循环都已启动
    assert!(perple.is_color_running().await.unwrap_or(false), "color 循环应该已启动");
    assert!(perple.is_lidar_running().await.unwrap_or(false), "lidar 循环应该已启动");
    assert!(perple.is_tracker_running().await.unwrap_or(false), "tracker 循环应该已启动");

    // 清理：停止所有循环
    let _ = perple.stop_color_loop().await;
    let _ = perple.stop_lidar_loop().await;
    let _ = perple.stop_tracker_loop().await;
}

/// 测试 run 方法启动后循环能正常执行
#[tokio::test]
async fn test_run_loops_execute() {
    let mut perple = Perple::new();

    // 启动所有循环
    let run_result = perple.run().await;
    assert!(run_result.is_ok());

    // 等待一小段时间，让循环有机会执行
    sleep(Duration::from_millis(100)).await;

    // 验证循环仍在运行
    assert!(perple.is_color_running().await.unwrap_or(false));
    assert!(perple.is_lidar_running().await.unwrap_or(false));
    assert!(perple.is_tracker_running().await.unwrap_or(false));

    // 清理：停止所有循环
    let _ = perple.stop_color_loop().await;
    let _ = perple.stop_lidar_loop().await;
    let _ = perple.stop_tracker_loop().await;

    // 等待一小段时间确保循环已停止
    sleep(Duration::from_millis(50)).await;

    // 验证循环已停止
    assert!(!perple.is_color_running().await.unwrap_or(false));
    assert!(!perple.is_lidar_running().await.unwrap_or(false));
    assert!(!perple.is_tracker_running().await.unwrap_or(false));
}

/// 测试多次调用 run 方法的错误处理
#[tokio::test]
async fn test_run_error_on_already_running() {
    let mut perple = Perple::new();

    // 第一次调用 run
    let result1 = perple.run().await;
    assert!(result1.is_ok());

    // 第二次调用 run 应该失败
    let result2 = perple.run().await;
    assert!(result2.is_err(), "第二次调用 run 应该返回错误");

    // 清理
    let _ = perple.stop_color_loop().await;
    let _ = perple.stop_lidar_loop().await;
    let _ = perple.stop_tracker_loop().await;
}

/// 测试单独启动各个循环的功能
#[tokio::test]
async fn test_individual_loop_start() {
    let mut perple = Perple::new();

    // 只启动 color 循环
    let color_result = perple.start_color_loop().await;
    assert!(color_result.is_ok());
    assert!(perple.is_color_running().await.unwrap_or(false));

    // 只启动 lidar 循环
    let lidar_result = perple.start_lidar_loop().await;
    assert!(lidar_result.is_ok());
    assert!(perple.is_lidar_running().await.unwrap_or(false));

    // 只启动 tracker 循环
    let tracker_result = perple.start_tracker_loop().await;
    assert!(tracker_result.is_ok());
    assert!(perple.is_tracker_running().await.unwrap_or(false));

    // 清理
    let _ = perple.stop_color_loop().await;
    let _ = perple.stop_lidar_loop().await;
    let _ = perple.stop_tracker_loop().await;
}

/// 测试按次数循环的模式
#[tokio::test]
async fn test_count_mode_loop() {
    let mut perple = Perple::new();

    // 启动一个只执行 3 次的循环
    let result = perple.start_color_loop_count(3).await;
    assert!(result.is_ok());

    // 等待循环完成（3 次 * 100ms = 300ms，加上一些余量）
    sleep(Duration::from_millis(500)).await;

    // 验证循环已自动停止
    assert!(!perple.is_color_running().await.unwrap_or(false));
}

/// 测试按时长循环的模式
#[tokio::test]
async fn test_duration_mode_loop() {
    let mut perple = Perple::new();

    // 启动一个运行 200ms 的循环
    let result = perple.start_color_loop_duration(200).await;
    assert!(result.is_ok());

    // 立即检查，应该还在运行
    assert!(perple.is_color_running().await.unwrap_or(false));

    // 等待超过指定时长
    sleep(Duration::from_millis(300)).await;

    // 验证循环已自动停止
    assert!(!perple.is_color_running().await.unwrap_or(false));
}

/// 测试信号控制的循环模式
#[tokio::test]
async fn test_signal_mode_loop() {
    let mut perple = Perple::new();

    // 启动信号控制的循环
    let result = perple.start_color_loop().await;
    assert!(result.is_ok());

    // 验证正在运行
    assert!(perple.is_color_running().await.unwrap_or(false));

    // 手动停止
    let stop_result = perple.stop_color_loop().await;
    assert!(stop_result.is_ok());

    // 等待一小段时间
    sleep(Duration::from_millis(50)).await;

    // 验证已停止
    assert!(!perple.is_color_running().await.unwrap_or(false));
}

/// 测试并发访问安全性
#[tokio::test]
async fn test_concurrent_access() {
    let mut perple = Perple::new();

    // 启动循环
    let run_result = perple.run().await;
    assert!(run_result.is_ok());

    // 并发检查多个循环的状态
    let camera_check = perple.camera.lock().unwrap();
    let lidar_check = perple.lidar.lock().unwrap();
    let tracker_check = perple.tracker.lock().unwrap();

    // 验证都能成功获取锁（没有死锁）
    drop(camera_check);
    drop(lidar_check);
    drop(tracker_check);

    // 清理
    let _ = perple.stop_color_loop().await;
    let _ = perple.stop_lidar_loop().await;
    let _ = perple.stop_tracker_loop().await;
}
