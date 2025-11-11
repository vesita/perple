//! 模型加载模块
//! 
//! 提供加载ONNX格式YOLO模型的功能。

use ort::session::{builder::GraphOptimizationLevel, Session};

/// 加载YOLO模型（只检测person类别）
/// 
/// 加载ONNX格式的YOLO模型，并应用优化配置。
/// 模型默认配置为使用4个线程进行推理，这对于大多数场景已经足够。
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

/// 从内存数据加载YOLO模型
/// 
/// 从字节数组加载ONNX格式的YOLO模型，适用于静态嵌入模型的场景。
/// 
/// # 参数
/// * `model_data` - 模型文件的字节数组
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
/// use perple::color::model::load_model_from_memory;
/// 
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// // 静态嵌入模型文件
/// // const MODEL_BYTES: &[u8] = include_bytes!("../../module/color/yolo11n.onnx");
/// // let model = load_model_from_memory(MODEL_BYTES)?;
/// # Ok(())
/// # }
/// ```
pub fn load_model_from_memory(model_data: &[u8]) -> Result<Session, ort::Error> {
    let model = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(4)?
        .commit_from_memory(&model_data)?;
    Ok(model)
}

/// 静态加载YOLO模型（示例）
/// 
/// 展示如何使用include_bytes!宏静态嵌入模型文件并加载
/// 注意：这只是一种示例实现，实际使用时需要根据模型文件的实际路径调整
/// 
/// # 返回值
/// 返回加载的Session对象
/// 
/// # 错误处理
/// 如果模型加载失败会返回Err
#[allow(dead_code)]
pub fn load_static_model() -> Result<Session, ort::Error> {
    // 使用include_bytes!宏在编译时将模型文件嵌入到二进制文件中
    // 注意：需要根据实际的模型文件路径进行调整
    // const MODEL_BYTES: &[u8] = include_bytes!("../../module/color/yolo11n.onnx");
    // load_model_from_memory(MODEL_BYTES)
    
    // 为了防止编译错误，这里暂时返回一个错误
    Err(ort::Error::new("Static model not configured. Please adjust the path in the source code."))
}