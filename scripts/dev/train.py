#.venv/bin/python3
"""
YOLO 模型训练脚本
"""
from pathlib import Path
import sys

import torch
from ultralytics import YOLO
import yaml

project_root = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(project_root))

from scripts.dev.model_utils import resolve_model_file, build_train_params
from scripts.dev.post_process import post_process


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    hyper_path = Path("scripts/hyper")
    data_config = hyper_path / "dataset.yaml"
    hyp_config_path = hyper_path / "hyp.yaml"

    with open(hyp_config_path, encoding="utf-8") as f:
        hyp_config = yaml.safe_load(f)

    model_path = Path("scripts/model")
    model_file = resolve_model_file(
        model_records_path=model_path / "records",
        original_model_path=model_path / "original" / "yolo11n.pt",
    )

    model = YOLO(model_file)
    train_params = build_train_params(hyp_config)

    # 如果从已有的 best.pt 继续训练，覆盖 name 避免混乱
    train_name = "yolo11n"

    print("开始训练...")
    results = model.train(
        data=str(data_config),
        device=device,
        pretrained=True,
        name=train_name,
        save=True,
        **train_params,
    )

    post_process()

    return results


if __name__ == "__main__":
    try:
        main()
        print("训练成功完成！")
    except Exception as e:
        print(f"训练失败: {e}")
        raise
