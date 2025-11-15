#.venv/bin/python3
# -*- coding: utf-8 -*-
"""
自动归档脚本
将训练结果从runs目录归档到records目录下
"""

import shutil
import sys
from pathlib import Path
from datetime import datetime


def archive_training_results():
    """
    自动归档训练结果
    将runs/detect目录下的最新训练结果归档到scripts/model/records目录下
    只保留best.pt权重文件以节省空间
    命名格式为：年月日-顺序编号
    """
    try:        
        # 定义源目录和目标目录
        runs_detect_dir = Path("runs/detect")
        records_dir = Path("scripts/model/records")
        
        # 确保目标目录存在
        records_dir.mkdir(parents=True, exist_ok=True)
        
        # 检查源目录是否存在训练结果
        if not runs_detect_dir.exists() or not any(runs_detect_dir.iterdir()):
            print("未找到任何训练结果")
            return False
        
        # 获取最新的训练目录（按修改时间排序）
        train_dirs = [d for d in runs_detect_dir.iterdir() if d.is_dir()]
        if not train_dirs:
            print("未找到任何训练目录")
            return False
            
        # 按修改时间排序，获取最新的训练目录
        latest_train_dir = max(train_dirs, key=lambda p: p.stat().st_mtime)
        print(f"找到最新训练目录: {latest_train_dir.name}")
        
        # 生成归档目录名称：年月日-顺序编号
        today = datetime.now().strftime("%y_%m_%d")
        
        # 查找当天已有的归档目录，确定编号
        existing_archives = list(records_dir.glob(f"{today}*"))
        next_index = len(existing_archives)
        archive_name = f"{today}_{next_index:02d}"
        archive_dir = records_dir / archive_name
        
        # 创建归档目录
        archive_dir.mkdir(exist_ok=True)
        
        # 复制所有文件和目录到归档目录
        print(f"正在归档到: {archive_dir}")
        
        for item in latest_train_dir.iterdir():
            if item.is_dir():
                # 对于weights目录，只复制best.pt文件
                if item.name == "weights":
                    weights_dir = archive_dir / item.name
                    weights_dir.mkdir(exist_ok=True)
                    
                    # 只复制best.pt文件
                    best_pt = item / "best.pt"
                    if best_pt.exists():
                        shutil.copy2(best_pt, weights_dir / "best.pt")
                        print(f"已复制: best.pt")
                else:
                    # 复制其他目录
                    shutil.copytree(item, archive_dir / item.name)
            else:
                # 复制非目录文件
                shutil.copy2(item, archive_dir / item.name)
        
        print(f"训练结果已成功归档到: {archive_dir}")
        return True
        
    except Exception as e:
        print(f"归档过程中发生错误: {e}")
        return False


if __name__ == "__main__":
    archive_training_results()