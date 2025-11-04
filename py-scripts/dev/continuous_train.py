#!/usr/bin/env python3
"""
YOLO 模型持续训练脚本
用于实现更好的训练策略，避免反复重启训练
"""

import os
from pathlib import Path
import sys
import time
import yaml
from datetime import datetime

import torch
from ultralytics import YOLO

# 添加项目根目录和py-scripts目录到Python路径
project_root = Path(__file__).parent.parent.parent.absolute()
py_scripts_path = project_root / "py-scripts"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(py_scripts_path))

# 导入自定义工具
from utils import archive_training_results


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


def evaluate_training_progress(results_dir):
    """
    评估训练进度，检查是否应该继续训练
    """
    try:
        # 检查results目录是否存在
        if not results_dir.exists():
            return True  # 如果没有结果目录，继续训练
        
        # 查找results.csv文件
        results_csv = results_dir / "results.csv"
        if not results_csv.exists():
            return True  # 如果没有结果文件，继续训练
            
        # 读取结果文件，检查最近几次的mAP值
        with open(results_csv, 'r') as f:
            lines = f.readlines()
            
        if len(lines) < 2:  # 至少需要标题行和一行数据
            return True
            
        # 解析最后一行数据
        last_line = lines[-1].strip().split(',')
        # mAP50通常在特定列，这里简单处理
        if len(last_line) >= 5:
            # 假设mAP50在第5列（索引4）
            try:
                map50 = float(last_line[4])
                print(f"当前mAP50值: {map50}")
                
                # 如果mAP50还没有达到满意的水平，继续训练
                if map50 < 0.85:  # 可以根据需求调整这个阈值
                    return True
                else:
                    return False
            except ValueError:
                # 如果解析失败，默认继续训练
                return True
        
        return True
    except Exception as e:
        print(f"评估训练进度时出错: {e}")
        return True  # 出错时默认继续训练


def continuous_train(max_cycles=3):
    """
    持续训练函数，可以多次循环训练直到满足条件
    
    Args:
        max_cycles: 最大训练循环次数
    """
    cycle = 0
    
    while cycle < max_cycles:
        cycle += 1
        print(f"\n开始第 {cycle}/{max_cycles} 轮训练")
        
        # 设备选择
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {device}")

        # 路径设置
        script_dir = Path(__file__).parent.absolute()
        root_dir = script_dir.parent
        
        hyper_path = root_dir / "hyper"
        model_path = root_dir / "model"
        
        # 配置文件路径
        data_config = hyper_path / "dataset.yaml"
        train_config_path = hyper_path / "train.yaml"
        optimizer_config_path = hyper_path / "optimizer.yaml"
        augment_config_path = hyper_path / "amt.yaml"
        
        # 加载训练配置
        with open(train_config_path, "r") as f:
            train_config = yaml.safe_load(f)
        
        # 加载优化器配置
        with open(optimizer_config_path, "r") as f:
            optimizer_hyper = yaml.safe_load(f)
            
        # 加载数据增强配置
        with open(augment_config_path, "r") as f:
            augment_config = yaml.safe_load(f)
        
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
        
        # 创建唯一的训练名称，避免覆盖之前的训练结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        train_name = f"yolo11n_continuous_{timestamp}"
        
        model = YOLO(str(model_file))
        
        # 准备数据增强参数
        augment_params = augment_config.get("augment", {})
        
        # 执行训练
        print("开始训练...")
        results = model.train(
            data=str(data_config),
            epochs=train_config["epochs"],
            imgsz=train_config["image_size"],
            workers=train_config["workers"],
            batch=train_config["batch"],
            optimizer=optimizer_hyper["optimizer"],  # 传递优化器名称而非实例
            device=device,
            pretrained=True,
            name=train_name,
            save=True,
            save_period=train_config["save_period"],
            cos_lr=train_config.get("cos_lr", False),  # 启用余弦退火学习率调度
            patience=train_config.get("patience", 100),  # 早停耐心值
            **augment_params  # 将数据增强参数传递给训练函数
        )
        
        # 检查训练结果目录
        runs_dir = project_root / "runs" / "detect" / train_name
        if evaluate_training_progress(runs_dir):
            print("模型还需要继续训练，准备下一轮训练...")
            
            # 训练完成后自动归档结果
            print("正在归档本轮训练结果...")
            try:
                if archive_training_results():
                    print("训练结果归档成功")
                else:
                    print("训练结果归档失败")
            except Exception as e:
                print(f"归档过程中发生错误: {e}")
                
            if cycle < max_cycles:
                print(f"等待10秒后开始下一轮训练...")
                time.sleep(10)  # 等待一段时间再开始下一轮训练
        else:
            print("模型训练已达到满意效果，停止训练")
            break
    
    # 最终归档
    print("正在进行最终归档...")
    try:
        if archive_training_results():
            print("最终训练结果归档成功")
        else:
            print("最终训练结果归档失败")
    except Exception as e:
        print(f"归档过程中发生错误: {e}")
    
    return results


if __name__ == "__main__":
    try:
        training_results = continuous_train(max_cycles=3)
        print("训练成功完成！")
    except Exception as e:
        print(f"训练过程中发生错误: {e}")
        raise