#.venv/bin/python3
"""
导出 YOLO ONNX：分别用两个进程避免路径覆盖
"""
import subprocess, sys
from pathlib import Path

BASE = Path(__file__).parent.parent.parent
OUT = BASE / "model/quantized"
OUT.mkdir(parents=True, exist_ok=True)

WEIGHTS = "runs/detect/yolo11n_finetune_new/weights/best.pt"

# 先确保 onnx 依赖
subprocess.run([sys.executable, "-m", "pip", "install", "onnx", "onnxruntime", "-q"])

# ── FP32 ──────────────────────────────────────────────────────────────
subprocess.run([
    sys.executable, "-c", f"""
from ultralytics import YOLO
model = YOLO(r"{WEIGHTS}")
model.export(format="onnx", imgsz=640, half=False)
import shutil
shutil.copy2("runs/detect/yolo11n_finetune_new/weights/best.onnx", r"{OUT / 'yolo11n.onnx'}")
print("FP32 done")
"""], check=True)

# ── FP16 ──────────────────────────────────────────────────────────────
subprocess.run([
    sys.executable, "-c", f"""
from ultralytics import YOLO
model = YOLO(r"{WEIGHTS}")
model.export(format="onnx", imgsz=640, half=True)
import shutil
shutil.copy2("runs/detect/yolo11n_finetune_new/weights/best.onnx", r"{OUT / 'yolo11n_fp16.onnx'}")
print("FP16 done")
"""], check=True)

print("\n结果:")
for f in sorted(OUT.glob("yolo11n*.onnx")):
    print(f"  {f.name}  ({f.stat().st_size/1e6:.1f} MB)")
