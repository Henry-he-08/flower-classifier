# 任务 1-3（数据增强）+ 任务 1-4（统一尺寸）
#
# 两条铁律：
#   1) 增强算子只加在 train_transform —— 训练集每个 epoch「看起来都不一样」，
#      相当于免费扩充样本；验证/测试集必须「原样」进网络，不能带随机性，
#      否则评估结果不可信。
#   2) Resize 放最后进网络前统一尺寸：CNN 的全连接层要求输入形状固定，
#      所以所有图都必须变成 IMG_SIZE x IMG_SIZE（本题 224，配合结构图）。
#
# Normalize 用 ImageNet 的均值/标准差：我们的模型从头训练，输入分布和
# 预训练时代一致能收敛更稳；这是社区通用基准值。

import sys
from pathlib import Path

from torchvision import transforms

# 单独运行本文件时（python src\data\transforms.py），Python 只把「脚本自己所在的
# 目录」D:\ML.2\src\data 放进搜索路径，够不到上一层的 src\config.py，所以手动补一行。
# 被 dataset.py import 时这行是多余的（它已经补过了），但留着不冲突。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import IMG_SIZE, PROCESSED_DIR

# ImageNet 数据集上统计出来的像素均值/标准差（RGB 三通道各一组）
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ---------- 验证 / 测试用：不做任何随机操作，顺序固定 ----------
eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),   # 任务 1-4：统一到 CNN 输入尺寸
    transforms.ToTensor(),                     # HWC(0~255整数) -> CHW(0~1浮点)
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# ---------- 训练用：随机增强，每个 epoch 同一张图都不一样 ----------
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    # ==================== TODO（任务 1-3 的核心，由你完成）====================
    # 从下面挑 2~3 个加进来（都是 torchvision.transforms 里的现成类）：
    #
    transforms.RandomHorizontalFlip(p=0.5),   
    transforms.RandomRotation(degrees=15),   # 随机旋转 ±15°。拍摄角度差异
    transforms.ColorJitter(brightness=0.2,   # 亮度/对比度/饱和度抖动。
                              contrast=0.2,     # 模拟光照、相机差异
                              saturation=0.2),
   
    #
    # 思考题（写进提交说明很加分）：
    #   - 为什么不做上下翻转 RandomVerticalFlip？（提示：花一般怎么拍？）
    #   - 增强太猛会发生什么？（提示：Rotate 90° 后郁金香可能看起来像别的类？）
    # =========================================================================
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),])


# ---------------- 自检：不训练，只看增强管道通不通 ----------------
# 验收标准（任务 1-3 + 1-4）：
#   1) train 输出形状必须是 (3, 224, 224)
#   2) 「同一张图跑两次结果相同吗」必须是 False —— 只要有一个随机算子生效就会 False；
#      如果打出 True，说明你挑的算子全是确定性的，等于没做增强
#   3) eval 那两行必须是 True —— 验证/测试集混进随机性，成绩就不可信了
if __name__ == "__main__":
    import torch
    from PIL import Image

    train_dir = PROCESSED_DIR / "train"
    img_path = next(p for p in sorted(train_dir.rglob("*"))
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    img = Image.open(img_path).convert("RGB")   # 只读，不动原文件
    print(f"抽到一张训练图: {img_path.parent.name}/{img_path.name}  原尺寸 {img.size}")

    a = train_transform(img)
    b = train_transform(img)
    print(f"train_transform 形状      : {tuple(a.shape)}   dtype={a.dtype}")
    print(f"train 像素范围            : {a.min():.3f} ~ {a.max():.3f}")
    print(f"同一张图跑两次结果相同吗  : {torch.allclose(a, b)}"
          f"   <- 必须是 False")

    e1 = eval_transform(img)
    e2 = eval_transform(img)
    print(f"eval_transform 形状       : {tuple(e1.shape)}")
    print(f"eval 跑两次结果相同吗     : {torch.allclose(e1, e2)}"
          f"   <- 必须是 True")
