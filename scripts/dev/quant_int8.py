#.venv/bin/python3
"""
INT8 动态量化 ONNX 模型
"""
import sys
from pathlib import Path
from onnxruntime.quantization import quantize_dynamic, QuantType

SRC = Path("model/quantized/yolo11n.onnx")
DST = Path("model/quantized/yolo11n_int8.onnx")

assert SRC.exists(), f"找不到 {SRC}"
print(f"输入: {SRC} ({SRC.stat().st_size/1e6:.1f} MB)")

# 动态量化：所有 MatMul 和 Add 节点转为 int8
quantize_dynamic(
    model_input=str(SRC),
    model_output=str(DST),
    per_channel=True,
    weight_type=QuantType.QInt8,
)

print(f"输出: {DST} ({DST.stat().st_size/1e6:.1f} MB)")
print("Done")
