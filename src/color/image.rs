//! 图像处理模块
//!
//! 提供图像加载、调整大小、转换为张量等图像处理功能。

use image::{DynamicImage, ImageBuffer, Rgb, imageops::FilterType};
use ndarray::{Array, Array4};
use ort::value::{Tensor, TensorValueType, Value};
use std::path::Path;

pub struct ScaleMessage {
    pub o_width: u32,
    pub o_height: u32,
    pub s_width: u32,
    pub s_height: u32,
    pub pad_x: f32,    // letterbox: horizontal padding on each side (model input pixels)
    pub pad_y: f32,    // letterbox: vertical padding on each side (model input pixels)
    pub scale: f32,    // letterbox: scale factor preserving aspect ratio (image→model)
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
/// ```ignore
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
    img.resize_exact(width, height, FilterType::Nearest)
}

pub fn scale_image(
    img: &DynamicImage,
    target_width: u32,
    target_height: u32,
) -> (DynamicImage, ScaleMessage) {
    let original_width = img.width();
    let original_height = img.height();

    // Letterbox: maintain aspect ratio, pad to target with gray (YOLO convention)
    let scale = (target_width as f32 / original_width as f32)
        .min(target_height as f32 / original_height as f32);
    let new_w = (original_width as f32 * scale).round() as u32;
    let new_h = (original_height as f32 * scale).round() as u32;
    let pad_x = (target_width - new_w) as f32 / 2.0;
    let pad_y = (target_height - new_h) as f32 / 2.0;
    let x_off = pad_x.round() as u32;
    let y_off = pad_y.round() as u32;

    let resized = img.resize_exact(new_w, new_h, FilterType::CatmullRom).to_rgb8();
    let mut canvas =
        image::ImageBuffer::from_pixel(target_width, target_height, image::Rgb([114, 114, 114]));
    image::imageops::overlay(&mut canvas, &resized, x_off as i64, y_off as i64);

    (
        DynamicImage::ImageRgb8(canvas),
        ScaleMessage {
            o_width: original_width,
            o_height: original_height,
            s_width: target_width,
            s_height: target_height,
            pad_x,
            pad_y,
            scale,
        },
    )
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
            tensor[[0, 0, y, x]] = (r as f32) / 255.0; // R通道
            tensor[[0, 1, y, x]] = (g as f32) / 255.0; // G通道
            tensor[[0, 2, y, x]] = (b as f32) / 255.0; // B通道
        }
    }

    // 返回处理好的图像张量
    tensor
}

pub fn input_image(
    img: &DynamicImage,
    input_height: usize,
    input_width: usize,
) -> Value<TensorValueType<f32>> {
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

/// 填充预创建的Value<TensorValueType<f32>>对象，避免返回时的拷贝
///
/// # 参数
/// * `img` - 输入图像
/// * `input_height` - 输入图像高度
/// * `input_width` - 输入图像宽度
/// * `tensor_value` - 预创建的Tensor Value对象，会被直接填充
pub fn fill_input_image(
    img: &DynamicImage,
    input_height: usize,
    input_width: usize,
    tensor_value: &mut Value<TensorValueType<f32>>,
) {
    let (dst_w, dst_h) = (input_width, input_height);
    let src_w = img.width() as usize;
    let src_h = img.height() as usize;

    // Letterbox: maintain aspect ratio, pad to target with gray (YOLO convention)
    let scale = (dst_w as f32 / src_w as f32).min(dst_h as f32 / src_h as f32);
    let new_w = (src_w as f32 * scale).round() as usize;
    let new_h = (src_h as f32 * scale).round() as usize;
    let x_offset = ((dst_w - new_w) as f32 / 2.0).round() as usize;
    let y_offset = ((dst_h - new_h) as f32 / 2.0).round() as usize;

    let src_bytes = img.as_bytes();
    let src_bpp = if src_bytes.len() >= src_w * src_h * 4 {
        4
    } else {
        3
    };

    // Fill with gray padding (YOLO convention: 114/255)
    let pad_value = 114.0 / 255.0;
    let mut nchw_data = vec![pad_value; dst_h * dst_w * 3];

    for dy in 0..new_h {
        let sy = ((dy as f32) / scale) as usize;
        let sy = sy.min(src_h - 1);
        for dx in 0..new_w {
            let sx = ((dx as f32) / scale) as usize;
            let sx = sx.min(src_w - 1);

            let px = (sy * src_w + sx) * src_bpp;
            let tx = x_offset + dx;
            let ty = y_offset + dy;
            let idx = ty * dst_w + tx;

            nchw_data[idx] = src_bytes[px] as f32 / 255.0;
            nchw_data[dst_h * dst_w + idx] = src_bytes[px + 1] as f32 / 255.0;
            nchw_data[2 * dst_h * dst_w + idx] = src_bytes[px + 2] as f32 / 255.0;
        }
    }

    *tensor_value = Tensor::from_array(([1, 3, dst_h, dst_w], nchw_data)).unwrap();
}

/// 图像去畸变映射（Brown-Conrady 模型）
///
/// 预计算畸变像素坐标 → 原始像素坐标的映射表，避免每帧重复计算。
/// 对每个输出像素 (u, v)，计算其在畸变图像中的采样位置。
pub struct UndistortMap {
    map_x: Vec<f32>,
    map_y: Vec<f32>,
    width: u32,
    height: u32,
}

impl UndistortMap {
    /// 创建去畸变映射表
    ///
    /// # 参数
    /// * `intrinsic` - 3×3 内参矩阵 [fx, 0, cx; 0, fy, cy; 0, 0, 1]
    /// * `dist` - 畸变系数 [k1, k2, p1, p2, k3]
    /// * `width` - 图像宽度
    /// * `height` - 图像高度
    pub fn new(intrinsic: &[[f32; 3]; 3], dist: &[f32; 5], width: u32, height: u32) -> Self {
        let fx = intrinsic[0][0];
        let fy = intrinsic[1][1];
        let cx = intrinsic[0][2];
        let cy = intrinsic[1][2];
        let (k1, k2, p1, p2, k3) = (dist[0], dist[1], dist[2], dist[3], dist[4]);
        if k1.abs() < 1e-6 && k2.abs() < 1e-6 && k3.abs() < 1e-6 && p1.abs() < 1e-6 && p2.abs() < 1e-6 {
            return Self { map_x: Vec::new(), map_y: Vec::new(), width, height };
        }

        let n = (width * height) as usize;
        let mut map_x = Vec::with_capacity(n);
        let mut map_y = Vec::with_capacity(n);

        for vy in 0..height {
            for vx in 0..width {
                // 输出像素 → 归一化坐标（去畸变空间）
                let x = (vx as f32 - cx) / fx;
                let y = (vy as f32 - cy) / fy;

                // 应用正向畸变（Brown-Conrady）
                let r2 = x * x + y * y;
                let r4 = r2 * r2;
                let r6 = r4 * r2;
                let radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;
                let x_dist = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
                let y_dist = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;

                // 归一化坐标 → 畸变图像像素坐标
                map_x.push(fx * x_dist + cx);
                map_y.push(fy * y_dist + cy);
            }
        }

        Self { map_x, map_y, width, height }
    }

    /// 对图像应用去畸变（双线性插值）
    pub fn apply(&self, img: &DynamicImage) -> DynamicImage {
        if self.map_x.is_empty() {
            return img.clone();
        }

        let rgb = img.to_rgb8();
        let mut output = ImageBuffer::new(self.width, self.height);

        for y in 0..self.height {
            for x in 0..self.width {
                let idx = (y * self.width + x) as usize;
                let src_x = self.map_x[idx];
                let src_y = self.map_y[idx];

                let pixel = if src_x >= 0.0 && src_y >= 0.0
                    && src_x < self.width as f32 - 1.0
                    && src_y < self.height as f32 - 1.0
                {
                    bilinear_interpolate(&rgb, src_x, src_y)
                } else {
                    Rgb([114, 114, 114]) // 越界填充灰色
                };
                output.put_pixel(x, y, pixel);
            }
        }

        DynamicImage::ImageRgb8(output)
    }
}

/// 双线性插值采样
fn bilinear_interpolate(img: &ImageBuffer<Rgb<u8>, Vec<u8>>, x: f32, y: f32) -> Rgb<u8> {
    let x0 = x.floor() as u32;
    let y0 = y.floor() as u32;
    let x1 = x0 + 1;
    let y1 = y0 + 1;
    let dx = x - x0 as f32;
    let dy = y - y0 as f32;

    let p00 = img.get_pixel(x0, y0);
    let p10 = img.get_pixel(x1, y0);
    let p01 = img.get_pixel(x0, y1);
    let p11 = img.get_pixel(x1, y1);

    Rgb([
        (p00[0] as f32 * (1.0 - dx) * (1.0 - dy)
            + p10[0] as f32 * dx * (1.0 - dy)
            + p01[0] as f32 * (1.0 - dx) * dy
            + p11[0] as f32 * dx * dy) as u8,
        (p00[1] as f32 * (1.0 - dx) * (1.0 - dy)
            + p10[1] as f32 * dx * (1.0 - dy)
            + p01[1] as f32 * (1.0 - dx) * dy
            + p11[1] as f32 * dx * dy) as u8,
        (p00[2] as f32 * (1.0 - dx) * (1.0 - dy)
            + p10[2] as f32 * dx * (1.0 - dy)
            + p01[2] as f32 * (1.0 - dx) * dy
            + p11[2] as f32 * dx * dy) as u8,
    ])
}
