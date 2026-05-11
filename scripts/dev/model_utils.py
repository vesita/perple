"""
YOLO 模型工具函数
被 train.py 和 continuous_train.py 共用
"""
from pathlib import Path
from typing import Optional

from ultralytics import YOLO


def find_latest_model_weights(model_records_path: Path) -> Optional[Path]:
    """按修改时间找最新的 best.pt"""
    record_dirs = [d for d in model_records_path.iterdir() if d.is_dir()]
    if not record_dirs:
        return None
    latest_dir = max(record_dirs, key=lambda x: x.stat().st_mtime)
    best_pt = latest_dir / "weights" / "best.pt"
    return best_pt if best_pt.exists() else None


def list_available_models(model_records_path: Path, original_model_path: Path):
    """列出所有可用模型 [(名称, 路径), ...]"""
    models = []
    if original_model_path.exists():
        models.append(("original", str(original_model_path)))

    if model_records_path.exists():
        dirs = sorted(
            (d for d in model_records_path.iterdir() if d.is_dir()),
            key=lambda x: x.stat().st_mtime, reverse=True,
        )
        for d in dirs:
            best = d / "weights" / "best.pt"
            if best.exists():
                models.append((d.name, str(best)))
    return models


def select_model_interactive(models):
    """交互选择，Enter=自动"""
    if not models:
        return None
    print("\n可用的模型:")
    print("0. 自动选择 (默认)")
    for i, (name, path) in enumerate(models, 1):
        print(f"{i}. {name} ({path})")
    print("\n请选择模型 (Enter=默认): ", end="")
    try:
        choice = input().strip()
        if not choice:
            return None
        idx = int(choice)
        if idx == 0:
            return None
        if 1 <= idx <= len(models):
            return models[idx - 1][1]
        print("无效选择，使用默认")
        return None
    except (ValueError, IndexError):
        print("输入无效，使用默认")
        return None


def resolve_model_file(
    model_records_path: Path,
    original_model_path: Path,
) -> str:
    """确定模型权重路径：交互→最新best→原始"""
    models = list_available_models(model_records_path, original_model_path)
    selected = select_model_interactive(models)
    if selected:
        print(f"使用选择的模型: {selected}")
        return selected

    latest = find_latest_model_weights(model_records_path)
    if latest and latest.exists():
        print(f"使用最新权重: {latest}")
        return str(latest)

    if original_model_path.exists():
        print(f"使用原始权重: {original_model_path}")
        return str(original_model_path)

    raise FileNotFoundError("未找到任何可用模型")


def build_train_params(hyp_config: dict) -> dict:
    """从 hyp.yaml 解析合并训练参数"""
    skip_keys = {"optimizer", "Adam", "SGD", "AdamW", "augment"}
    params = {k: v for k, v in hyp_config.items() if k not in skip_keys}
    params.update(hyp_config.get("augment", {}))
    params.update(hyp_config.get(hyp_config.get("optimizer", "Adam"), {}))
    return params
