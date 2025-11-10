//! 图像处理模块
//! 
//! 提供图像加载、调整大小、转换为张量等图像处理功能。

use image::{DynamicImage, GenericImageView, imageops::FilterType};
use ndarray::{Array, Array4};
use ort::value::{Tensor, TensorValueType, Value};
use std::path::Path;


pub struct ScaleMessage {
    pub o_width: u32,
    pub o_height: u32,
    pub s_width: u32,
    pub s_height: u32,
}


/// 加载图像文件
/// 
/// 从指定路径加载图像文件。
/// 
/// # 参数
/// * `path` - 图像文件路径
/// 
/// # 返回值
/// 返回加载的DynamicImage对象
/// 
/// # 错误处理
/// 如果图像加载失败会返回Err，包含错误信息
/// 
/// # 示例
/// 
/// ```
/// use perple::color::image::load_image;
/// 
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let image = load_image("path/to/image.jpg")?;
/// # Ok(())
/// # }
/// ```
pub fn load_image(path: &str) -> Result<DynamicImage, Box<dyn std::error::Error>> {
    // 验证路径是否有效
    let path = Path::new(path);
    if !path.exists() {
        return Err(format!("图像文件不存在: {:?}", path).into());
    }

    // 加载图像
    let img = image::open(path).map_err(|e| format!("无法加载图像: {}", e))?;
    Ok(img)
}

/// 调整图像大小以适应模型输入
/// 
/// 使用CatmullRom插值算法将图像调整为指定尺寸。
/// 
/// # 参数
/// * `img` - 原始图像
/// * `width` - 目标宽度
/// * `height` - 目标高度
/// 
/// # 返回值
/// 返回调整大小后的图像
pub fn resize_image(img: &DynamicImage, width: u32, height: u32) -> DynamicImage {
    img.resize_exact(width, height, FilterType::CatmullRom)
}

pub fn scale_image(img: &DynamicImage, target_width: u32, target_height: u32) -> (DynamicImage, ScaleMessage) {
    let original_width = img.width();
    let original_height = img.height();
    
    let scale_width = target_width;
    let scale_height = target_height;
    
    let resized_img = img.resize_exact(target_width, target_height, FilterType::CatmullRom);
    
    let scale_message = ScaleMessage {
        o_width: original_width,
        o_height: original_height,
        s_width: scale_width,
        s_height: scale_height,
    };
    
    (resized_img, scale_message)
}

/// 将图像转换为模型输入张量
/// 
/// 将图像转换为模型所需的四维张量格式，包括：
/// 1. 归一化像素值到[0, 1]范围
/// 2. 调整通道顺序为RGB
/// 3. 调整维度顺序为NCHW格式
/// 
/// # 参数
/// * `img` - 图像
/// * `input_height` - 输入图像高度
/// * `input_width` - 输入图像宽度
/// 
/// # 返回值
/// 返回形状为(1, 3, height, width)的四维张量，通道顺序为RGB，像素值范围[0, 1]
pub fn image_to_tensor(img: &DynamicImage, input_height: usize, input_width: usize) -> Array4<f32> {
    // 创建用于模型输入的张量，形状为(1, 3, input_height, input_width)
    let mut tensor = Array::zeros((1, 3, input_height, input_width));
    
    // 获取图像的RGB数据，避免多次调用to_rgb8()
    let rgb_img = img.to_rgb8();
    
    // 使用enumerate来同时获取坐标和像素值，避免像素坐标转换开销
    for (y, row) in rgb_img.rows().enumerate() {
        for (x, pixel) in row.enumerate() {
            let [r, g, b] = pixel.0;
            
            // 将RGB通道分别存储在对应的通道维度中
            tensor[[0, 0, y, x]] = (r as f32) / 255.0;  // R通道
            tensor[[0, 1, y, x]] = (g as f32) / 255.0;  // G通道
            tensor[[0, 2, y, x]] = (b as f32) / 255.0;  // B通道
        }
    }
    
    // 返回处理好的图像张量
    tensor
}

pub fn input_image(img: &DynamicImage, input_height: usize, input_width: usize) -> Value<TensorValueType<f32>> {
    // 调整图像大小以适应模型输入
    let resized_img = resize_image(img, input_width as u32, input_height as u32);
    
    // 预分配准确大小的向量并初始化为0
    let mut nchw_data = vec![0.0f32; input_height * input_width * 3];
    
    // 获取RGB图像数据
    let rgb_img = resized_img.to_rgb8();
    
    // 一次性遍历所有像素，并直接按NCHW格式写入
    for (y, row) in rgb_img.rows().enumerate() {
        for (x, pixel) in row.enumerate() {
            let [r, g, b] = pixel.0;
            
            // 直接按照NCHW格式写入数据
            // R 通道 (channel 0)
            let r_index = y * input_width + x;
            nchw_data[r_index] = r as f32 / 255.0;
            
            // G 通道 (channel 1)
            let g_index = input_height * input_width + y * input_width + x;
            nchw_data[g_index] = g as f32 / 255.0;
            
            // B 通道 (channel 2)
            let b_index = 2 * input_height * input_width + y * input_width + x;
            nchw_data[b_index] = b as f32 / 255.0;
        }
    }
    
    // 创建 ONNX Tensor
    Tensor::from_array(([1, 3, input_height, input_width], nchw_data)).unwrap()
}