# 任务 1-2：划分数据集
#
# 背景：题目给的 train/ 目录要切两刀 —— 大部分用来训练，留一部分当「验证集」，
#       用来在训练过程中监控泛化效果、选超参数（Titanic 里你已经这么干过）。
#       test/ 是题目给的最终考卷，只在整个流程最后碰一次，绝不用来调参。
#
# 策略：按「类」循环，在每个类内部按 VAL_RATIO 切分，再复制到
#       data/processed/train/<类名>/ 和 data/processed/val/<类名>/
#
# 运行（先跑 clean.py）：
#   D:\ML\.venv_gpu\Scripts\python.exe src\data\split.py

import shutil
import sys
from pathlib import Path

from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import PROCESSED_DIR, RAW_TRAIN_DIR, SEED, VAL_RATIO


def list_class_files(cls_dir: Path) -> list[Path]:
    """列出某个类文件夹下的所有图片文件（已由 clean.py 保证都是好图）。"""
    return sorted(p for p in cls_dir.iterdir() if p.is_file())


def copy_files(files: list[Path], dst_dir: Path) -> None:
    """把一批文件复制到目标文件夹（复制而非移动：原始 train 保持完整）。"""
    dst_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.copy2(str(f), str(dst_dir / f.name))


def reset_output() -> None:
    """清空上次的划分结果。管道要能反复跑（幂等），就必须先清场。

    为什么必须做：copy2 只会「覆盖同名文件」，永远不会「删除多余文件」。
    上一次跑出来的文件只要这次没被同名覆盖到，就会一直躺在目录里 ——
    数据看着变多了，比例也会失真（这正是 tulip 显示 24.4% 的原因）。
    """
    for sub in ("train", "val"):
        d = PROCESSED_DIR / sub
        if d.exists():
            shutil.rmtree(d)
    print(f"已清空旧输出: {PROCESSED_DIR}")


def main() -> None:
    class_dirs = sorted(d for d in RAW_TRAIN_DIR.iterdir() if d.is_dir())
    print(f"发现 {len(class_dirs)} 个类别: {[d.name for d in class_dirs]}")

    reset_output()

    for cls_dir in class_dirs:
        files = list_class_files(cls_dir)

        # ---- 切分（已完成）----
        # 在「每个类内部」按 VAL_RATIO 切，天然就是分层的：
        # 每类内部都是 8:2，整体的类别比例自动保持，不需要再传 stratify=y。
        train_files, val_files = train_test_split(
            files, test_size=VAL_RATIO, random_state=SEED)

        # 备选方案 B（不调库，想手动复现随机划分时用）：
        #   import random
        #   random.seed(SEED)          # 固定种子，可复现
        #   files = files[:]           # 别动原列表
        #   random.shuffle(files)
        #   n_val = int(len(files) * VAL_RATIO)
        #   val_files, train_files = files[:n_val], files[n_val:]

        # 复制到目标目录（train 一份、val 一份）
        copy_files(train_files, PROCESSED_DIR / "train" / cls_dir.name)
        copy_files(val_files,   PROCESSED_DIR / "val"   / cls_dir.name)

    # ---- 校验：打印每类数量和实际比例，肉眼确认没有切歪 ----
    print("\n类别            train   val    val占比")
    for cls_dir in class_dirs:
        n_train = len(list((PROCESSED_DIR / "train" / cls_dir.name).glob("*")))
        n_val = len(list((PROCESSED_DIR / "val" / cls_dir.name).glob("*")))
        ratio = n_val / (n_train + n_val) if (n_train + n_val) else 0
        print(f"{cls_dir.name:<14} {n_train:>5}   {n_val:>4}    {ratio:.1%}")
    print(f"\n划分完成，输出目录: {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
