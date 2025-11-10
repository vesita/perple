use image::{DynamicImage, GenericImageView, imageops::FilterType};
use ndarray::{Array, Array4};
use std::path::Path;

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
    
    // 遍历所有像素，将RGB值归一化到[0, 1]范围并存储到张量中
    for pixel in img.pixels() {
        let x = pixel.0 as usize;
        let y = pixel.1 as usize;
        let [r, g, b, _] = pixel.2.0;
        
        // 将RGB通道分别存储在对应的通道维度中
        tensor[[0, 0, y, x]] = (r as f32) / 255.0;  // R通道
        tensor[[0, 1, y, x]] = (g as f32) / 255.0;  // G通道
        tensor[[0, 2, y, x]] = (b as f32) / 255.0;  // B通道
    }
    
    // 返回处理好的图像张量
    tensor
}