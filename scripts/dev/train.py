#.venv/bin/python3
"""
YOLO 模型训练脚本
用于训练自定义数据集的目标检测模型
"""

from pathlib import Path
import sys

import torch
from ultralytics import YOLO
import yaml

# 使用统一的路径处理方法
# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(project_root))

# 导入自定义工具
from scripts.dev.archive import archive_training_results


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


def list_available_models(model_records_path, original_model_path):
    """
    列出所有可用的模型供用户选择
    """
    models = []
    
    # 添加原始模型
    if original_model_path.exists():
        models.append(("original", str(original_model_path)))
    
    # 添加训练记录中的模型
    if model_records_path.exists():
        record_dirs = [d for d in model_records_path.iterdir() if d.is_dir()]
        # 按修改时间排序，最新的在前面
        record_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        for i, record_dir in enumerate(record_dirs):
            best_pt_path = record_dir / "weights" / "best.pt"
            if best_pt_path.exists():
                models.append((record_dir.name, str(best_pt_path)))
    
    return models


def select_model_interactive(models):
    """
    交互式选择模型
    """
    if not models:
        return None
    
    print("\n可用的模型:")
    print("0. 自动选择 (默认，使用最新的模型)")
    for i, (name, path) in enumerate(models, 1):
        print(f"{i}. {name} ({path})")
    
    print("\n请选择要使用的模型 (按Enter使用默认选择): ", end="")
    try:
        choice = input().strip()
        if not choice:  # 按Enter键使用默认选择
            return None  # 返回None表示使用默认逻辑
        
        choice_idx = int(choice)
        if choice_idx == 0:
            return None  # 使用默认逻辑
        elif 1 <= choice_idx <= len(models):
            return models[choice_idx - 1][1]  # 返回模型路径
        else:
            print("无效选择，使用默认逻辑")
            return None
    except (ValueError, IndexError):
        print("输入无效，使用默认逻辑")
        return None


def main():
    """主训练函数"""
    # 设备选择
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")    
    hyper_path = Path("scripts/hyper")
    model_path = Path("scripts/model")
    
    # 配置文件路径
    data_config = hyper_path / "dataset.yaml"
    hyp_config_path = hyper_path / "hyp.yaml"  # 使用统一的超参数配置文件
    
    # 加载超参数配置
    with open(hyp_config_path, "r") as f:
        hyp_config = yaml.safe_load(f)
    
    # 初始化模型 - 优先使用最新的best.pt，备选使用original下的yolo11n.pt
    model_records_path = model_path / "records"
    original_model_path = model_path / "original" / "yolo11n.pt"
    
    # 获取所有可用模型并让用户选择
    available_models = list_available_models(model_records_path, original_model_path)
    selected_model_path = select_model_interactive(available_models)
    
    # 根据用户选择或默认逻辑确定模型文件
    if selected_model_path:
        model_file = selected_model_path
        print(f"使用用户选择的模型权重: {model_file}")
    else:
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
    
    model = YOLO(str(model_file))
    
    # 准备数据增强参数
    augment_params = hyp_config.get("augment", {})
    
    # 准备优化器参数
    optimizer_name = hyp_config["optimizer"]
    optimizer_params = hyp_config.get(optimizer_name, {})
    
    # 合并所有训练参数，避免重复传递
    train_params = {}
    # 添加训练参数（除了优化器相关）
    for key, value in hyp_config.items():
        if key not in ["optimizer", "Adam", "SGD", "AdamW", "augment"]:
            train_params[key] = value
    train_params.update(augment_params)  # 数据增强参数
    train_params.update(optimizer_params)  # 优化器参数
    
    # 执行训练
    print("开始训练...")
    results = model.train(
        data=str(data_config),
        optimizer=optimizer_name,  # 传递优化器名称
        device=device,
        pretrained=True,
        name="yolo11n",
        save=True,
        **train_params  # 传递合并后的所有参数
    )
    
    # 训练完成后自动归档结果
    print("训练完成，正在归档结果...")
    try:
        if archive_training_results():
            print("训练结果归档成功")
        else:
            print("训练结果归档失败")
    except Exception as e:
        print(f"归档过程中发生错误: {e}")
    
    return results


if __name__ == "__main__":
    try:
        training_results = main()
        print("训练成功完成！")
    except Exception as e:
        print(f"训练过程中发生错误: {e}")
        raise