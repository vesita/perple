"""导入 D:\windows_v1.8.1\extra 标注数据到训练集，加 ext_ 前缀避免命名冲突"""
import xml.etree.ElementTree as ET
import shutil
from pathlib import Path

SRC_IMG = Path("D:/windows_v1.8.1/extra/image")
SRC_XML = Path("D:/windows_v1.8.1/extra/label")
DST_IMG = Path("data/yolo_new/train/images")
DST_LBL = Path("data/yolo_new/train/labels")

DST_IMG.mkdir(parents=True, exist_ok=True)
DST_LBL.mkdir(parents=True, exist_ok=True)

count = 0
for xml_path in sorted(SRC_XML.glob("*.xml")):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find("size")
    iw = int(size.find("width").text)
    ih = int(size.find("height").text)
    fname = root.find("filename").text  # e.g. 000000.jpg

    # Build ext_ prefixed names
    stem = fname.rsplit(".", 1)[0]
    dst_name = f"ext_{stem}.jpg"
    label_name = f"ext_{stem}.txt"

    img_src = SRC_IMG / fname
    if not img_src.exists():
        print(f"  MISSING: {fname}")
        continue

    # Copy with ext_ prefix
    shutil.copy2(img_src, DST_IMG / dst_name)

    label_lines = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        if name != "person":
            continue
        bb = obj.find("bndbox")
        xmin = float(bb.find("xmin").text)
        ymin = float(bb.find("ymin").text)
        xmax = float(bb.find("xmax").text)
        ymax = float(bb.find("ymax").text)
        cx = (xmin + xmax) / 2 / iw
        cy = (ymin + ymax) / 2 / ih
        w = (xmax - xmin) / iw
        h = (ymax - ymin) / ih
        label_lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    (DST_LBL / label_name).write_text("\n".join(label_lines) + "\n")
    count += 1
    print(f"  {fname}  ->  {dst_name}  /  {label_name}  ({len(label_lines)} obj)")

print(f"\nDone: {count} images + labels added")
imgs = sorted(DST_IMG.glob("*.jpg"))
lbls = sorted(DST_LBL.glob("*.txt"))
print(f"Total train images: {len(imgs)}")
print(f"Total train labels: {len(lbls)}")
