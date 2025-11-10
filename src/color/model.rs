use ort::session::{builder::GraphOptimizationLevel, Session};

/// 加载YOLO模型（只检测person类别）
/// 
/// 加载ONNX格式的YOLO模型，并应用优化配置。
/// 
/// # 参数
/// * `model_path` - 模型文件路径
/// 
/// # 返回值
/// 返回加载的Session对象
/// 
/// # 错误处理
/// 如果模型加载失败会返回Err
/// 
/// # 示例
/// 
/// ```
/// use perple::color::model::load_model;
/// 
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let model = load_model("path/to/model.onnx")?;
/// # Ok(())
/// # }
/// ```
pub fn load_model(model_path: &str) -> Result<Session, ort::Error> {
    let model = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(4)?
        .commit_from_file(model_path)?;
    Ok(model)
}