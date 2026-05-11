"""
墙体管线对比测试 - 自动执行脚本

运行墙体管线对比 bench，汇总结果并生成分析图。
支持 quick/full 模式。

用法:
  python scripts/run_wall_pipeline.py                        # quick 模式
  python scripts/run_wall_pipeline.py --mode=full            # full 模式
  python scripts/run_wall_pipeline.py --analysis-only        # 仅从已有数据重绘图
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def run_bench(mode: str, release: bool = False):
    """执行 wall_pipeline_bench。"""
    exe_name = "wall_pipeline_bench"
    profile = "release" if release else "debug"

    print("构建 {} ({})...".format(exe_name, profile))
    cmd = ["cargo", "build"]
    if release:
        cmd.append("--release")
    cmd.extend(["--example", exe_name])
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    if result.returncode != 0:
        print("构建失败:\n{}".format(result.stderr), file=sys.stderr)
        sys.exit(1)

    target_dir = REPO_ROOT / "target" / profile
    if os.name == "nt":
        exe_path = target_dir / "examples" / "{}.exe".format(exe_name)
    else:
        exe_path = target_dir / "examples" / exe_name

    print("运行 {} --mode={}...\n".format(exe_name, mode))
    ret = subprocess.run(
        [str(exe_path), "--mode={}".format(mode)],
        cwd=REPO_ROOT, capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    if ret.returncode != 0:
        print("运行失败: returncode={}".format(ret.returncode), file=sys.stderr)
        sys.exit(1)

    print(ret.stdout)
    if ret.stderr:
        print(ret.stderr, file=sys.stderr)


def analysis_only():
    """仅从已有数据重绘分析图（未来扩展）。"""
    output_dir = REPO_ROOT / "output" / "bench" / "wall_pipeline"
    json_files = list(output_dir.glob("*/info.json"))
    if not json_files:
        print("没有找到已有数据 ({})".format(output_dir))
        return

    print("找到 {} 个策略的数据文件".format(len(json_files)))
    print("分析图生成（待实现）")


def main():
    parser = argparse.ArgumentParser(description="墙体管线对比测试")
    parser.add_argument("--mode", choices=["quick", "full"], default="quick",
                       help="测试模式 (默认: quick)")
    parser.add_argument("--release", action="store_true",
                       help="使用 release 模式运行")
    parser.add_argument("--analysis-only", action="store_true",
                       help="仅从已有数据重绘图")
    args = parser.parse_args()

    os.chdir(REPO_ROOT)

    if args.analysis_only:
        analysis_only()
        return

    run_bench(args.mode, args.release)


if __name__ == "__main__":
    main()
