# 任务 1-4 收尾：把处理好的图片装进 Dataset / DataLoader
#
# 概念一句话（详见 docs/概念题.md 第 3 题）：
#   Dataset  = "仓库"，规定「一共多少张(__len__)」和「按编号取一张(__getitem__)」
#   DataLoader = "卡车"，在仓库外面套一层，负责：按 batch_size 打包、
#                shuffle 打乱、num_workers 多进程搬货
#
# 我们直接用现成的 ImageFolder（torchvision 提供的 Dataset 子类）：
#   它按「文件夹名 = 类别标签」自动打标签，并生成 class_to_idx 映射表。

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (BATCH_SIZE, NUM_CLASSES, NUM_WORKERS, PROCESSED_DIR,
                    RAW_TEST_DIR)
from data.transforms import eval_transform, train_transform


def make_loaders(batch_size: int = BATCH_SIZE,
                 num_workers: int = NUM_WORKERS):
    """构建三个 DataLoader，返回 (train_loader, val_loader, test_loader, class_to_idx)"""
    train_ds = datasets.ImageFolder(str(PROCESSED_DIR / "train"),
                                    transform=train_transform)
    val_ds = datasets.ImageFolder(str(PROCESSED_DIR / "val"),
                                  transform=eval_transform)
    # test 直接用原始目录：clean.py 已把坏图移走，剩下的都是好图
    test_ds = datasets.ImageFolder(str(RAW_TEST_DIR),
                                   transform=eval_transform)

    # 三份数据的类别 -> 标签编号映射必须一致，先自查
    assert train_ds.class_to_idx == val_ds.class_to_idx == test_ds.class_to_idx, \
        "三个数据集的类别映射不一致，检查 processed 目录是否完整"

    # ==================== TODO（由你完成）====================

    train_loader = DataLoader(train_ds, batch_size=batch_size,
        shuffle=True,num_workers=num_workers)  # 为什么必须打乱？（提示：
    #                                                     # 不打乱，模型会按文件夹
    #                                                     # 顺序"背答案"）
    #   val/test 两个：shuffle 要不要 True？为什么？
    #   再想想 drop_last（最后凑不满一个 batch 的尾巴要不要扔）——可先不传。
    # =========================================================
    val_loader =  DataLoader(val_ds, batch_size=batch_size,
        shuffle=False,num_workers=num_workers)
    test_loader =  DataLoader(test_ds, batch_size=batch_size,
        shuffle=False,num_workers=num_workers)

    return train_loader, val_loader, test_loader, train_ds.class_to_idx


# ---------------- 自检：不训练，只看数据通不通 ----------------
if __name__ == "__main__":
    print(f" NUM_CLASSES = {NUM_CLASSES}")
    train_loader, val_loader, test_loader, class_to_idx = make_loaders()
    print(f"类别映射 class_to_idx = {class_to_idx}")
    print(f"train 共 {len(train_loader.dataset)} 张 / {len(train_loader)} 个 batch")
    print(f"val   共 {len(val_loader.dataset)} 张 / {len(val_loader)} 个 batch")
    print(f"test  共 {len(test_loader.dataset)} 张 / {len(test_loader)} 个 batch")

    # 抓一个 batch 出来验货：形状必须是 [B, 3, 224, 224]
    images, labels = next(iter(train_loader))
    print(f"\n一个 batch: images.shape = {tuple(images.shape)}, "
          f"labels = {labels[:8].tolist()}")
    print(f"像素范围（Normalize 后应有负数）: {images.min():.2f} ~ {images.max():.2f}")
