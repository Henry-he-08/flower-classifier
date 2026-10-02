# 任务 1-3 进阶版：B 档数据增强（不改动 transforms.py，v1 那套原样保留）
#
# v1 的三个算子（水平翻转 / 旋转 / 颜色抖动）全部保留，另外加三个更强的：
#
#   RandomResizedCrop —— 随机裁一块再放大回 224。v1 的 Resize 是"整张图压扁"，
#                        模型见到的构图永远一样；这个让它见惯"远近 + 偏心"的构图
#   RandAugment       —— 自动从 14 个算子的池子里随机挑 2 个执行，一行顶十几个算子
#   RandomErasing     —— 随机涂掉一小块，逼模型别只盯某个局部特征（防"只认花瓣颜色"）
#
# 三者叠加后，同一张图连续两次几乎不可能重复 —— 等效于把 2425 张训练图
# 变成"每一轮都是新图"的无限流，这是小数据集上最划算的正则手段。
#
# 铁律不变：这三个只能进 train_transform_v2。
#           eval_transform 一点随机性都不能沾，否则成绩不可信。

import sys
from pathlib import Path

from torchvision import transforms

# 单独运行本文件时，手动把 src/ 放进搜索路径（和 transforms.py 同一个处理）
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import IMG_SIZE
from data.transforms import IMAGENET_MEAN, IMAGENET_STD

# 注意三个阶段的顺序不能乱：
#   PIL 阶段（几何 + 颜色 + RandAugment）  ->  ToTensor  ->  张量阶段（RandomErasing）
train_transform_v2 = transforms.Compose([
    # 1) 随机裁剪 + 缩放回 224：替代 v1 的固定 Resize
    #    scale=(0.7, 1.0) 表示"至少保留原图 70% 的面积"，避免裁得太狠把花丢了
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0), ratio=(0.8, 1.25)),

    # 2) v1 的几何增强，保留
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),

    # 3) v1 的颜色增强，幅度略微调大（0.2 -> 0.25）
    transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25),

    # 4) 自动增强：每次随机挑 2 个算子，magnitude=7 属中等强度（上限 9）
    #    强度别再往上加：randaugment 里有 solarize / posterize 这类会破坏
    #    颜色信息的算子，调太猛会让"黄花"和"橙花"变得难以区分
    transforms.RandAugment(num_ops=2, magnitude=7),

    # 5) 转张量 + 标准化（必须排在所有 PIL 算子之后）
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),

    # 6) 随机擦除：必须在 ToTensor 之后，它吃的是张量不是 PIL 图
    #    p=0.25 表示每 4 张图擦 1 张，擦掉的面积占 2%~15%
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
])


# ---------------- 自检：不训练，只验证增强管道 ----------------
# 验收标准：
#   1) train 形状必须是 (3, 224, 224)
#   2) 连跑 5 次两两不同 —— 只要有一个随机算子生效就该全不同
#   3) eval 跑两次必须完全一致
if __name__ == "__main__":
    import torch
    from PIL import Image

    from config import PROCESSED_DIR
    from data.transforms import eval_transform

    train_dir = PROCESSED_DIR / "train"
    img_path = next(p for p in sorted(train_dir.rglob("*"))
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    img = Image.open(img_path).convert("RGB")   # 只读，不动原文件
    print(f"抽到一张训练图: {img_path.parent.name}/{img_path.name}  原尺寸 {img.size}")

    a = train_transform_v2(img)
    print(f"v2 train_transform 形状  : {tuple(a.shape)}   dtype={a.dtype}")
    print(f"v2 像素范围              : {a.min():.3f} ~ {a.max():.3f}")

    outs = [train_transform_v2(img) for _ in range(5)]
    same = sum(torch.allclose(outs[0], o) for o in outs[1:])
    print(f"连跑 5 次与第 1 次相同的次数: {same}   <- 必须是 0")

    e1, e2 = eval_transform(img), eval_transform(img)
    print(f"eval_transform 跑两次相同吗: {torch.allclose(e1, e2)}   <- 必须是 True")
    print("自检通过" if same == 0 and torch.allclose(e1, e2) else "自检未通过，检查算子")
