"""聚类参数实验批量运行 — 自动修改 config → 运行 eval → 收集结果"""
import subprocess
import json
import shutil
import re
from pathlib import Path

CONFIG_PATH = Path("config/default.toml")
BACKUP_PATH = Path("config/default.toml.bak")

# 实验组合: (name, changes_dict)
EXPERIMENTS = [
    ("baseline", {}),
    ("min_pts_5", {"cluster.min_points_per_cluster": "5"}),
    ("min_pts_3", {"cluster.min_points_per_cluster": "3"}),
    ("voxel_005", {"cluster.voxel_size": "0.05"}),
    ("voxel_005_min_pts_5", {"cluster.voxel_size": "0.05", "cluster.min_points_per_cluster": "5"}),
    ("merge_005", {"cluster.merge_patience": "0.05"}),
    ("merge_015", {"cluster.merge_patience": "0.15"}),
    ("merge_020", {"cluster.merge_patience": "0.20"}),
    ("lvdot", {"cluster.strategy": '"lvdot"'}),
    ("xy_dbscan", {"cluster.strategy": '"xy_dbscan"'}),
]

def patch_config(changes):
    """直接按行替换修改 config 文件（避免 TOML 解析的编码问题）"""
    with open(CONFIG_PATH, encoding="utf-8") as f:
        lines = f.readlines()

    for key, val in changes.items():
        parts = key.split(".")
        target_key = parts[-1]
        # 只处理最末级的 key
        found = False
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("#") or "=" not in stripped:
                continue
            k = stripped.split("=")[0].strip()
            if k == target_key:
                lines[i] = f"{target_key} = {val}\n"
                found = True
                break
        if not found:
            print(f"  [WARN] 未找到配置项: {key}")

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        f.writelines(lines)


def run_eval(name):
    """运行 eval_labeled 并解析结果"""
    print(f"\n{'='*60}")
    print(f"  实验: {name}")
    print(f"{'='*60}")

    result = subprocess.run(
        ["cargo", "run", "--example", "eval_labeled", "--",
         "--center-dist", "0.5", "--output", f"./output/experiment_{name}"],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300
    )
    output = (result.stdout or "") + (result.stderr or "")
    # 打印最后几行
    for line in output.splitlines():
        if any(kw in line for kw in ["Precision", "Recall", "F1", "进度:", ">>", "总体", "检测:", "╔"]):
            print(f"  {line}")

    p = re.search(r'Precision:\s*([\d.]+)%', output)
    r = re.search(r'Recall:\s*([\d.]+)%', output)
    f1 = re.search(r'F1:\s*([\d.]+)', output)
    tp = re.search(r'TP:\s*(\d+)', output)
    fp = re.search(r'FP:\s*(\d+)', output)
    fn = re.search(r'FN:\s*(\d+)', output)
    det = re.search(r'检测:\s*(\d+)', output)

    return {
        "name": name,
        "precision": float(p.group(1)) if p else None,
        "recall": float(r.group(1)) if r else None,
        "f1": float(f1.group(1)) if f1 else None,
        "tp": int(tp.group(1)) if tp else None,
        "fp": int(fp.group(1)) if fp else None,
        "fn": int(fn.group(1)) if fn else None,
        "n_det": int(det.group(1)) if det else None,
    }


# ─── 主流程 ─────────────────────────────────────────────────────────────────
shutil.copy2(CONFIG_PATH, BACKUP_PATH)

results = []
try:
    for name, changes in EXPERIMENTS:
        # 还原配置
        shutil.copy2(BACKUP_PATH, CONFIG_PATH)
        # 应用修改
        if changes:
            patch_config(changes)

        res = run_eval(name)
        results.append(res)

        p = res["precision"]
        r = res["recall"]
        f = res["f1"]
        print(f"  >> {name}: P={p}%  R={r}%  F1={f}")

finally:
    shutil.copy2(BACKUP_PATH, CONFIG_PATH)
    BACKUP_PATH.unlink(missing_ok=True)

# ─── 汇总表 ─────────────────────────────────────────────────────────────────
print("\n")
print("="*70)
print("  实验汇总")
print("="*70)
header = f"  {'实验名':<20} {'Precision':<10} {'Recall':<10} {'F1':<10} {'TP':<6} {'FP':<6} {'FN':<6} {'检测数':<6}"
print(header)
print(f"  {'-'*18} {'-'*8} {'-'*8} {'-'*8} {'-'*4} {'-'*4} {'-'*4} {'-'*4}")
for r in results:
    p = r['precision'] or 0
    rv = r['recall'] or 0
    f = r['f1'] or 0
    print(f"  {r['name']:<20} {p:<8.1f}% {rv:<8.1f}% {f:<8.4f} "
          f"{r['tp']:<6} {r['fp']:<6} {r['fn']:<6} {r['n_det']:<6}")

out = Path("output/experiment_summary.json")
out.parent.mkdir(exist_ok=True)
with open(out, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\n结果已保存到 {out}")
