#.venv/bin/python3
"""
YOLO 模型评估脚本
用于评估训练完成的目标检测模型性能
"""

import os
from pathlib import Path

import torch
from ultralytics import YOLO
import yaml


def main():
    """主评估函数"""
    # 设备选择
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    hyper_path = "scripts/hyper"
    model_path = "scripts/model"
    
    # 配置文件路径
    data_config = hyper_path / "dataset.yaml"
    
    # 查找最佳模型权重文件
    best_model_path = None
    weights_dir = Path("runs" / "detect")
    
    # 遍历所有训练运行目录，找到最新的yolo11n5运行
    yolo11n5_runs = list(weights_dir.glob("yolo11n5*"))
    if yolo11n5_runs:
        # 选择最新的运行目录
        latest_run = max(yolo11n5_runs, key=os.path.getctime)
        best_model_path = latest_run / "weights" / "best.pt"
        last_model_path = latest_run / "weights" / "last.pt"
        
        # 检查best.pt是否存在，否则使用last.pt
        if best_model_path.exists():
            model_file = best_model_path
        elif last_model_path.exists():
            model_file = last_model_path
        else:
            raise FileNotFoundError(f"未找到模型权重文件在 {latest_run}")
    else:
        # 如果没有找到yolo11n5运行，尝试使用原始模型
        model_file = model_path / "original" / "yolo11n.pt"
        if not model_file.exists():
            raise FileNotFoundError(f"未找到模型文件: {model_file}")
    
    print(f"加载模型: {model_file}")
    
    # 加载模型
    model = YOLO(str(model_file))
    
    # 执行验证
    print("开始评估模型...")
    results = model.val(
        data=str(data_config),
        imgsz=640,
        batch=4,
        device=device,
        workers=4,
        verbose=True
    )
    
    # 输出评估结果
    print("\n=== 模型评估结果 ===")
    print(f"Precision: {results.box.mp:.4f}")
    print(f"Recall: {results.box.mr:.4f}")
    print(f"mAP50: {results.box.map50:.4f}")
    print(f"mAP50-95: {results.box.map:.4f}")
    
    return results


if __name__ == "__main__":
    try:
        eval_results = main()
        print("\n评估成功完成！")
    except Exception as e:
        print(f"评估过程中发生错误: {e}")
        raise