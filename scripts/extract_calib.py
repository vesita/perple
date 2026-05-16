#!/usr/bin/env python3
"""从 STPoints image.json 提取标定参数并生成 TOML 配置片段"""
import json, sys

with open("config/stpoints/image.json") as f:
    calib = json.load(f)

# 内参 3x3
intr = calib["intrinsic"]
print("# 从 config/stpoints/image.json 提取的相机标定参数")
print("\n[camera]")
print(f"intrinsic = [")
for r in range(3):
    row = intr[r*3:(r+1)*3]
    print(f"  [ {row[0]:>12.8f}, {row[1]:>12.8f}, {row[2]:>12.8f} ],")
print("]")

# 外参 4x4
extr = calib["extrinsic"]
print(f"\nextrinsic = [")
for r in range(4):
    row = extr[r*4:(r+1)*4]
    print(f"  [ {row[0]:>12.8f}, {row[1]:>12.8f}, {row[2]:>12.8f}, {row[3]:>12.8f} ],")
print("]")

# 畸变系数
if "dist_coeffs" in calib and calib["dist_coeffs"]:
    dc = calib["dist_coeffs"]
    # Pad to 5 if needed
    while len(dc) < 5:
        dc.append(0.0)
    print(f"\ndist_coeffs = [ {dc[0]:>8.6f}, {dc[1]:>8.6f}, {dc[2]:>9.7f}, {dc[3]:>9.7f}, {dc[4]:>9.7f} ]")
else:
    print("\n# dist_coeffs = None  # 无畸变")
