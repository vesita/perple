#.venv/bin/python3
"""
用新标注数据 fine-tune，强制适配当前相机视角
"""
from pathlib import Path
import sys

import torch
from ultralytics import YOLO

project_root = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(project_root))

from scripts.dev.model_utils import resolve_model_file
from scripts.dev.post_process import post_process

NEW_DATA_YAML = str(project_root / "scripts" / "hyper" / "dataset_new.yaml")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")

    model_path = project_root / "scripts" / "model"
    model_file = resolve_model_file(
        model_records_path=model_path / "records",
        original_model_path=model_path / "original" / "yolo11n.pt",
    )
    model = YOLO(model_file)

    results = model.train(
        data=NEW_DATA_YAML,
        device=device,
        epochs=100,
        patience=20,
        batch=8,
        imgsz=640,
        workers=2,
        cos_lr=True,
        optimizer="Adam",
        lr0=0.0001,          # fine-tune 用小学习率
        lrf=0.01,
        weight_decay=0.05,
        warmup_epochs=2,
        close_mosaic=10,
        pretrained=True,
        name="yolo11n_finetune_new",
        save=True,
    )

    post_process()


if __name__ == "__main__":
    main()
