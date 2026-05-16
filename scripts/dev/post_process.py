#.venv/bin/python3
"""
训练后处理统一入口：归档完整训练目录 → 导出 ONNX
被 train.py / continuous_train.py / finetune_new.py 共用

功能：
- 归档完整训练记录（权重 + 图表 + 指标），目录名含 mAP
- 导出 ONNX 到 model/quantized/
- 提供信号处理器，Ctrl+C 时自动归档
"""
import shutil, subprocess, sys, signal
from pathlib import Path
from datetime import datetime


def _read_map50(run_dir: Path) -> float | None:
    """从训练目录的 results.csv 读取最新 mAP50"""
    csv = run_dir / "results.csv"
    if not csv.exists():
        return None
    try:
        lines = csv.read_text().strip().splitlines()
        if len(lines) < 2:
            return None
        cols = lines[-1].split(",")
        if len(cols) >= 8:
            return float(cols[7])  # mAP50 在第 8 列
    except (ValueError, IndexError):
        pass
    return None


def _archive_name(run_dir: Path) -> str:
    """生成归档目录名: yy_mm_dd_idx_mAP"""
    today = datetime.now().strftime("%y_%m_%d")
    records_dir = Path("scripts/model/records")
    existing = sorted(records_dir.glob(f"{today}_*"))
    idx = len(existing)
    map50 = _read_map50(run_dir)
    tag = f"_mAP{int(map50 * 100):02d}" if map50 is not None else ""
    return f"{today}_{idx:02d}{tag}"


def archive_run(run_dir: Path) -> Path | None:
    """
    归档完整训练目录（weights + 图表 + 指标）到 scripts/model/records/。
    即使训练被打断也能归档已有结果。
    """
    records_dir = Path("scripts/model/records")
    records_dir.mkdir(parents=True, exist_ok=True)

    if not run_dir.exists():
        print(f"  [归档] 目录不存在: {run_dir}")
        return None

    name = _archive_name(run_dir)
    dst = records_dir / name
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copytree(run_dir, dst, dirs_exist_ok=True)
    print(f"  [归档] {run_dir.name} → {dst}")

    # 更新 latest.pt
    best_pt = run_dir / "weights" / "best.pt"
    if best_pt.exists():
        latest_link = records_dir / "latest.pt"
        shutil.copy2(best_pt, latest_link)

    return dst


def archive_latest_run() -> Path | None:
    """归档 runs/detect/ 下最新的训练目录"""
    runs_dir = Path("runs/detect")
    if not runs_dir.exists():
        print("  [归档] runs/detect/ 不存在")
        return None
    dirs = [d for d in runs_dir.iterdir() if d.is_dir()]
    if not dirs:
        print("  [归档] 无训练目录")
        return None
    latest = max(dirs, key=lambda p: p.stat().st_mtime)
    return archive_run(latest)


def _export_one(pt_path: str, out_dir: Path, half: bool, suffix: str) -> bool:
    """导出单个 ONNX 文件，返回是否成功"""
    import tempfile
    from ultralytics import YOLO

    with tempfile.TemporaryDirectory() as tmp:
        tmp_pt = Path(tmp) / "model.pt"
        shutil.copy2(pt_path, str(tmp_pt))
        model = YOLO(str(tmp_pt))
        model.export(format="onnx", imgsz=640, half=half)
        onnx_src = tmp_pt.with_suffix(".onnx")
        if onnx_src.exists():
            dst = out_dir / suffix
            shutil.copy2(str(onnx_src), str(dst))
            print(f"  [导出] {suffix} done")
            return True
        print(f"  [导出] {suffix} 失败：未找到 ONNX 文件")
        return False


def export_onnx(pt_path: Path | str = "scripts/model/records/latest.pt",
                int8: bool = True) -> None:
    """从 best.pt 导出 ONNX (FP32 + FP16 + 可选 INT8) 到 model/quantized/"""
    out_dir = Path("model/quantized")
    out_dir.mkdir(parents=True, exist_ok=True)
    pt_path = str(pt_path)

    if not Path(pt_path).exists():
        print(f"  [导出] 权重不存在: {pt_path}")
        return

    fp32_ok = _export_one(pt_path, out_dir, half=False, suffix="yolo11n.onnx")
    _export_one(pt_path, out_dir, half=True, suffix="yolo11n_fp16.onnx")

    # INT8 动态量化基于 FP32 ONNX
    if int8 and fp32_ok:
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            src = out_dir / "yolo11n.onnx"
            dst = out_dir / "yolo11n_int8.onnx"
            quantize_dynamic(
                model_input=str(src), model_output=str(dst),
                per_channel=True, weight_type=QuantType.QInt8,
            )
            print(f"  [导出] yolo11n_int8.onnx ({dst.stat().st_size/1e6:.1f} MB)")
        except ImportError:
            print("  [导出] INT8 量化跳过（需要 onnxruntime）")

    print(f"  [导出] ONNX → {out_dir}/")


def post_process(export_onnx_flag: bool = True) -> None:
    """训练后处理：归档最新训练目录 + 可选导出 ONNX"""
    print("\n--- 训练后处理 ---")
    archive_dir = archive_latest_run()
    if archive_dir and export_onnx_flag:
        export_onnx()
    print("--- 后处理完成 ---\n")


# ── 信号处理器：Ctrl+C 时自动归档 ────────────────────────────────────
_SAFE_RUN_DIR: Path | None = None


def _on_interrupt(sig, frame):
    """Ctrl+C 时归档当前训练结果后退出"""
    print("\n\n捕获中断信号，正在归档训练结果...")
    if _SAFE_RUN_DIR and _SAFE_RUN_DIR.exists():
        archive_run(_SAFE_RUN_DIR)
    else:
        archive_latest_run()
    print("已归档，安全退出。")
    sys.exit(0)


def install_interrupt_handler(run_dir: Path | None = None):
    """注册 Ctrl+C 处理器，中断时自动归档 run_dir"""
    global _SAFE_RUN_DIR
    _SAFE_RUN_DIR = run_dir
    signal.signal(signal.SIGINT, _on_interrupt)


if __name__ == "__main__":
    post_process()
