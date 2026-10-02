# 任务 1-1：扫描并隔离损坏/无法打开的图片
#
# ⚠️ 原则：原始数据只读。发现坏图的处置方式是「移动到隔离区」+「写报告」，
#          绝不删除，随时可以移回。
#
# 判定思路（两层检查，缺一不可）：
#   1) img.verify() —— 只校验文件头/校验和，速度快，能抓"伪 jpg / 头损坏"；
#      但它不解码像素，有些"头正常、数据截断"的图会漏网。
#   2) 重新 open 后 img.load() —— 真正解码全部像素，能抓"能打开但中途断掉"的图。
#      注意：verify() 之后的文件对象不能再用了，必须重新打开，这是 PIL 的坑。
#
# 运行（在任意终端可原样粘贴）：
#   D:\ML\.venv_gpu\Scripts\python.exe src\data\clean.py

import csv
import shutil
import sys
from pathlib import Path

from PIL import Image

# 把 src/ 加进模块搜索路径，好 import 上一层目录的 config
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (IMAGE_EXTS, QUARANTINE_DIR, RAW_TEST_DIR, RAW_TRAIN_DIR,
                    REPORT_DIR)


def iter_images(root: Path):
    """按类名文件夹遍历 root 下所有图片，返回 (路径, 所属split根目录)。"""
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p


def check_image(path: Path) -> str | None:
    """检查单张图片。正常返回 None；损坏返回错误原因字符串。"""
    try:
        with Image.open(path) as img:
            img.verify()  # 第一层：文件头校验
    except Exception as e:
        return f"verify失败: {type(e).__name__}: {e}"
    try:
        with Image.open(path) as img:  # verify 后必须重新打开（PIL 的坑）
            img.load()  # 第二层：完整解码，抓数据截断
    except Exception as e:
        return f"decode失败: {type(e).__name__}: {e}"
    return None


def quarantine(src: Path, split_root: Path) -> Path:
    """把坏图从原始数据移动到隔离区，保持 <split>/<类名>/ 的目录结构。"""
    rel = src.relative_to(split_root)               # 例如 daisy\xxx.jpg
    dst = QUARANTINE_DIR / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():  # 重名兜底：加序号，绝不覆盖
        dst = dst.with_name(f"{dst.stem}_1{dst.suffix}")
    shutil.move(str(src), str(dst))
    return dst


def main() -> None:
    splits = [("train", RAW_TRAIN_DIR), ("test", RAW_TEST_DIR)]
    total, bad = 0, []

    # ---- 第一步：扫描（只读） ----
    for split_name, root in splits:
        n_split, n_bad_split = 0, 0
        for p in iter_images(root):
            total += 1
            n_split += 1
            reason = check_image(p)
            if reason is not None:
                bad.append((split_name, p, reason))
                n_bad_split += 1
        print(f"[扫描] {split_name}: {n_split} 张, 坏图 {n_bad_split} 张")

    print(f"[扫描] 合计 {total} 张, 坏图 {len(bad)} 张")

    # ---- 第二步：隔离 + 写报告 ----
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / "bad_files.csv"
    with open(report_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["原始路径", "所属集合", "错误原因", "处理动作"])
        for split_name, p, reason in bad:
            dst = quarantine(p, RAW_TRAIN_DIR if split_name == "train" else RAW_TEST_DIR)
            writer.writerow([str(p), split_name, reason, f"已移动到 {dst}"])
            print(f"[隔离] {p.name} -> {dst}  ({reason})")

    print(f"\n报告已写入: {report_path}")
    print("隔离区（可随时移回）: ", QUARANTINE_DIR)


if __name__ == "__main__":
    main()
