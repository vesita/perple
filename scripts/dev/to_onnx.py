#.venv/bin/python3
"""
YOLO 模型导出到 ONNX 格式脚本
自动选择最佳模型权重进行导出
"""

import os
from pathlib import Path
import sys
from ultralytics import YOLO
import shutil

# 添加项目根目录和py-scripts目录到Python路径
project_root = Path(__file__).parent.parent.parent.absolute()
py_scripts_path = project_root / "py-scripts"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(py_scripts_path))


def find_latest_model_weights(model_records_path):
    """
    查找最新的模型权重文件
    按修改时间排序，选择最新的目录中的best.pt文件
    """
    # 获取所有训练记录目录
    record_dirs = [d for d in model_records_path.iterdir() if d.is_dir()]
    
    if not record_dirs:
        return None
    
    # 按修改时间排序，获取最新的目录
    latest_dir = max(record_dirs, key=lambda x: x.stat().st_mtime)
    best_pt_path = latest_dir / "weights" / "best.pt"
    
    # 检查best.pt是否存在
    if best_pt_path.exists():
        return best_pt_path
    
    return None


def auto_select_model():
    """
    自动选择最佳模型权重进行导出
    """
    # 路径设置
    root_dir = Path(__file__).parent.parent.absolute()
    model_path = root_dir / "model"
    
    # 初始化模型 - 优先使用最新的best.pt，备选使用original下的yolo11n.pt
    model_records_path = model_path / "records"
    original_model_path = model_path / "original" / "yolo11n.pt"
    
    # 尝试查找最新的best.pt
    latest_model_file = find_latest_model_weights(model_records_path)
    
    if latest_model_file and latest_model_file.exists():
        model_file = latest_model_file
        print(f"使用最新的模型权重: {model_file}")
    elif original_model_path.exists():
        model_file = original_model_path
        print(f"使用原始模型权重: {model_file}")
    else:
        raise FileNotFoundError(f"未找到任何可用的模型文件")
        
    return str(model_file)


def main():
    # 自动选择模型
    model_file = auto_select_model()
    
    # 加载YOLO11模型
    model = YOLO(model_file)
    
    # 导出自定义名称的ONNX模型到指定目录
    model.export(format="onnx",  
                nms=True,
    )
    
    # 获取最新模型权重所在的目录
    root_dir = Path(__file__).parent.parent.absolute()
    model_records_path = root_dir / "model" / "records"
    latest_weight_path = find_latest_model_weights(model_records_path)
    onnx_model_dir = latest_weight_path.parent if latest_weight_path else None
    
    if onnx_model_dir is None:
        raise FileNotFoundError("无法找到导出的ONNX模型目录")
    
    # 加载导出的ONNX模型
    exported_onnx_path = onnx_model_dir / "best.onnx"
    
    # 创建目标目录（在项目根目录下）
    target_dir = project_root / "module" / "color"
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # 移动并重命名ONNX模型
    final_onnx_path = target_dir / "yolo11n.onnx"
    shutil.move(str(exported_onnx_path), str(final_onnx_path))
    print(f"ONNX模型已移动至: {final_onnx_path}")
    
    # 加载移动后的ONNX模型
    onnx_model = YOLO(str(final_onnx_path), task="detect")
    
    # 运行推理
    results = onnx_model("data/color/valid/images/PennPed00051.png")
    
    print("ONNX模型导出和测试完成!")


if __name__ == "__main__":
    main()