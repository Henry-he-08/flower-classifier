# 学习率 / 优化器 对照实验室
#
# 为什么做这个实验
# ----------------
# 你第一次训练的结果（SGD lr=1e-4, momentum=0.9, 15 轮）：
#     train loss  1.6099 -> 1.6010     只降了 0.0089
#     train acc   0.177  -> 0.297
#     val   acc   0.182  -> 0.294      最佳 0.3186
#
# 而 ln(5) = 1.6094 —— 这正是「五分类瞎猜」的损失值。
# 注意 train acc 和 val acc 几乎一样（0.297 / 0.294）：这是**欠拟合**，不是过拟合。
# 模型连训练集都没记住，说明参数几乎没被更新过。
#
# 本实验把「数据」和「初始权重」都钉死，只换优化器/学习率，
# 这样就能把「学不动」的责任精确地落到 lr 上，而不是怪模型或数据。
#
# 运行：D:\ML\.venv_gpu\Scripts\python.exe docs\labs\lr_lab.py

import sys
import time
from pathlib import Path

# 单独运行本文件时，把 src 补进搜索路径（本文件在 docs\labs\ 下，所以往上两级）
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch
from torch import nn

from config import NUM_CLASSES, SEED
from data.dataset import make_loaders
from model import MyModel

STEPS = 30          # 每组跑多少个 batch（1 个 epoch = 76 个 batch）


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    train_loader, _, _, _ = make_loaders()

    # 预先取 STEPS 个 batch 搬到显卡上，之后每组都复用同一批数据
    batches = []
    for i, (x, y) in enumerate(train_loader):
        if i >= STEPS:
            break
        batches.append((x.to(device), y.to(device)))
    print(f"prepared {len(batches)} batches (fixed data, same for every config)")
    print()

    criterion = nn.CrossEntropyLoss()

    configs = [
        ("SGD lr=1e-4 mom=0.9  <-- YOURS", lambda m: torch.optim.SGD(m.parameters(), lr=1e-4, momentum=0.9)),
        ("SGD lr=1e-3 mom=0.9", lambda m: torch.optim.SGD(m.parameters(), lr=1e-3, momentum=0.9)),
        ("SGD lr=1e-2 mom=0.9", lambda m: torch.optim.SGD(m.parameters(), lr=1e-2, momentum=0.9)),
        ("SGD lr=1e-1 mom=0.9", lambda m: torch.optim.SGD(m.parameters(), lr=1e-1, momentum=0.9)),
        ("Adam lr=1e-4", lambda m: torch.optim.Adam(m.parameters(), lr=1e-4)),
        ("Adam lr=1e-3", lambda m: torch.optim.Adam(m.parameters(), lr=1e-3)),
    ]

    print(f"{'config':32} {'start':>8} {'step5':>8} {'end':>8} {'total drop':>12}")
    print("-" * 74)

    for tag, build_opt in configs:
        torch.manual_seed(SEED)                 # 每组都从同一份初始权重出发
        model = MyModel(NUM_CLASSES).to(device)
        opt = build_opt(model)

        losses = []
        t0 = time.time()
        for x, y in batches:
            opt.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            opt.step()
            losses.append(loss.item())

        print(f"{tag:32} {losses[0]:8.4f} {losses[4]:8.4f} {losses[-1]:8.4f} "
              f"{losses[0] - losses[-1]:12.6f}   ({time.time() - t0:.1f}s)")

    print()
    print("reference: random-guess loss = ln(5) = 1.6094")
    print()
    print("读法:")
    print("  'total drop' 接近 0        -> 这组参数几乎没动，就是你现在的情况")
    print("  'total drop' 明显变大      -> 这组的步长配得上这个网络")
    print("  但 drop 也不是越大越好    -> lr=1e-1 那行如果 loss 变 NaN 或乱跳，就是炸了")


if __name__ == "__main__":
    main()
