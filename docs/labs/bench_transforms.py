# 诊断：数据增强管道到底慢在哪一步
#
# 背景：v2 训练时 GPU 利用率长期接近 0，一个 epoch 要 4 分钟以上。
# 怀疑瓶颈不在显卡，而在 CPU 上做增强。这个脚本串行跑各个算子，
# 打印「每张图多少毫秒」，直接定位元凶，再决定砍谁。
#
# 运行：
#   D:\ML\.venv_gpu\Scripts\python.exe docs\labs\bench_transforms.py

import sys
import time
from pathlib import Path

from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from config import IMG_SIZE, PROCESSED_DIR
from data.transforms import IMAGENET_MEAN, IMAGENET_STD, train_transform
from data.transforms_v2 import train_transform_v2

train_dir = Path(PROCESSED_DIR) / "train"
paths = [p for p in sorted(train_dir.rglob("*"))
         if p.suffix.lower() in {".jpg", ".jpeg", ".png"}][:60]
imgs = [Image.open(p).convert("RGB") for p in paths]
print(f"样本 {len(imgs)} 张，逐张串行计时\n")


def bench(tf, n: int = 60) -> float:
    """返回每张图的平均耗时（毫秒）"""
    t0 = time.time()
    for im in imgs[:n]:
        tf(im)
    return (time.time() - t0) / n * 1000


# ---------- 1. 单个算子 ----------
single = [
    ("Resize(224)（v1 底座）", transforms.Resize((IMG_SIZE, IMG_SIZE))),
    ("RandomResizedCrop", transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0))),
    ("RandomHorizontalFlip", transforms.RandomHorizontalFlip(p=0.5)),
    ("RandomRotation(15)", transforms.RandomRotation(15)),
    ("ColorJitter(0.25x3)", transforms.ColorJitter(0.25, 0.25, 0.25)),
    ("RandAugment(2, 7)", transforms.RandAugment(num_ops=2, magnitude=7)),
]
print("单个算子：")
for name, op in single:
    print(f"  {name:<26} {bench(op):8.2f} ms")

# ---------- 2. 完整管道对比 ----------
# 无 RandAugment 版：其余算子与 v2 完全一致，只把它摘掉
no_ra = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0), ratio=(0.8, 1.25)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
])

print("\n完整管道：")
for name, tf in [("v1 train_transform", train_transform),
                 ("v2 去掉 RandAugment", no_ra),
                 ("v2 完整（当前配置）", train_transform_v2)]:
    ms = bench(tf)
    print(f"  {name:<26} {ms:8.2f} ms/张")

print("\n换算：每小时单进程能处理的图片数（粗略上界）")
for name, tf in [("v1", train_transform), ("v2 去掉 RandAugment", no_ra),
                 ("v2 完整", train_transform_v2)]:
    print(f"  {name:<26} {3600 / (bench(tf) / 1000):>8.0f} 张/小时")
