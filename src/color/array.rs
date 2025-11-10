//! 数组处理模块
//! 
//! 提供将ndarray数组转换为ONNX Runtime张量的功能。

use ndarray::Array4;
use ort::value::{Tensor, TensorValueType, Value};

/// 将ndarray数组转换为ONNX Runtime张量
/// 
/// # 参数
/// * `mats` - 四维数组，形状为(1, 3, height, width)
/// 
/// # 返回值
/// 返回对应的ONNX Runtime张量
pub fn to_input(mats: &Array4<f32>) -> Value<TensorValueType<f32>> {
    let shape: Vec<usize> = mats.shape().to_vec();
    let (data, _offset) = mats.clone().into_raw_vec_and_offset();
    let result = Tensor::from_array((
        [shape[0], shape[1], shape[2], shape[3]],
        data
    )).unwrap();
    result
}