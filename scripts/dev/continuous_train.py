#.venv/bin/python3
"""
YOLO 模型持续训练脚本
多次循环训练直到满足精度或达到最大轮次
"""
from pathlib import Path
import sys
import time
from datetime import datetime

import torch
from ultralytics import YOLO
import yaml

project_root = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(project_root))

from scripts.dev.model_utils import resolve_model_file, build_train_params
from scripts.dev.post_process import post_process


def evaluate_training_progress(results_dir: Path) -> bool:
    """检查是否应继续训练（True=继续）"""
    try:
        results_csv = results_dir / "results.csv"
        if not results_csv.exists():
            return True
        lines = results_csv.read_text().strip().splitlines()
        if len(lines) < 2:
            return True
        cols = lines[-1].strip().split(",")
        if len(cols) >= 8:
            map50 = float(cols[7])
            print(f"当前 mAP50: {map50:.4f}")
            return map50 < 0.92
        return True
    except Exception as e:
        print(f"评估进度异常（默认继续）: {e}")
        return True


def continuous_train(max_cycles=3):
    cycle = 0
    results = None

    while cycle < max_cycles:
        cycle += 1
        print(f"\n=== 第 {cycle}/{max_cycles} 轮训练 ===")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"设备: {device}")

        hyp_config_path = Path("scripts/hyper/hyp.yaml")
        data_config = hyp_config_path.parent / "dataset_new.yaml"

        with open(hyp_config_path, encoding="utf-8") as f:
            hyp_config = yaml.safe_load(f)

        model_file = resolve_model_file(
            model_records_path=Path("scripts/model/records"),
            original_model_path=Path("scripts/model/original/yolo11n.pt"),
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        train_name = f"yolo11n_continuous_{timestamp}"

        model = YOLO(model_file)
        train_params = build_train_params(hyp_config)

        print("开始训练...")
        results = model.train(
            data=str(data_config),
            device=device,
            pretrained=True,
            name=train_name,
            save=True,
            **train_params,
        )

        runs_dir = Path("runs") / "detect" / train_name

        if evaluate_training_progress(runs_dir):
            print("需要继续训练...")
            post_process(export_onnx_flag=False)
            if cycle < max_cycles:
                print("等待 10 秒后下一轮...")
                time.sleep(10)
        else:
            print("精度已达标，停止训练")
            break

    print("最终归档中...")
    post_process()

    return results


if __name__ == "__main__":
    try:
        continuous_train(max_cycles=3)
        print("训练成功完成！")
    except Exception as e:
        print(f"训练失败: {e}")
        raise
