use ndarray::{Array2, s};
use ort::{session::Session, value::Tensor, inputs};
use ndarray::Array4;

/// 运行模型推理
/// 
/// 使用ONNX模型对输入张量进行推理，返回处理后的结果。
/// 
/// # 参数
/// * `model` - ONNX模型Session
/// * `input` - 输入张量，形状应为(1, 3, height, width)
/// 
/// # 返回值
/// 返回处理后的模型输出，形状为(num_boxes, 5)，包含[x, y, w, h, conf]
/// 
/// # 错误处理
/// 如果推理过程中发生错误会返回Err
/// 
/// # 示例
/// 
/// ```
/// use ort::session::Session;
/// use ndarray::Array4;
/// use perple::color::infer::run_inference;
/// 
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let model = Session::builder()?.commit_from_file("path/to/model.onnx")?;
/// let input = Array4::<f32>::zeros((1, 3, 640, 640)); // 示例输入
/// let output = run_inference(&mut model, &input)?;
/// # Ok(())
/// # }
/// ```
pub fn run_inference(model: &mut Session, input: &Array4<f32>) -> Result<Array2<f32>, Box<dyn std::error::Error>> {
    // 运行模型推理
    let shape: Vec<usize> = input.shape().to_vec();
    let (data, _offset) = input.clone().into_raw_vec_and_offset();
    let input_tensor = Tensor::from_array((
        [shape[0], shape[1], shape[2], shape[3]],
        data
    ))?;
    let outputs = model.run(inputs!["images" => input_tensor])?;
    
    // 提取输出并处理
    let output = outputs[0].try_extract_tensor::<f32>()?;
    let shape = output.0.clone();
    
    // 验证输出形状
    if shape.len() != 3 || shape[0] != 1 {
        return Err("模型输出形状不符合预期".into());
    }
    
    // YOLO模型输出形状为 [1, num_boxes, num_params]
    // 其中num_params通常为6: [x, y, w, h, conf, class_conf] 
    let data = output.1.to_vec();
    
    // 将数据重塑为二维数组 [num_boxes, num_params]
    let array_2d = Array2::from_shape_vec((shape[1] as usize, shape[2] as usize), data)?;
    
    // 只取前5列 [x, y, w, h, conf]
    let array = array_2d.slice(s![.., 0..5]).to_owned();
    
    Ok(array)
}