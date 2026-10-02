# 训练 v2：A + B 档提升实验（结构一行没动，完全符合"按结构图搭 AlexNet"的题目要求）
#
# 运行：
#   D:\ML\.venv_gpu\Scripts\python.exe src\train_v2.py
#
# 设计原则：本文件与 train.py 完全独立，跑完不覆盖 v1 的任何产物：
#   权重 -> checkpoints/best_model_v2.pth        （不动 best_model.pth）
#   历史 -> docs/logs/history_AdamW_v2.json
#   曲线 -> docs/figures/fig_curve_AdamW_v2.png
#
# 相比 v1（Adam + 15 轮 + 基础增强）的 6 处改动：
#   A1 轮数     15 -> 35              v1 到第 15 轮 train/val 都还在涨，是被 EPOCHS 截断的
#   A2 调度器   CosineAnnealingLR     学习率从 1e-4 余弦衰减到约 0，专治 v1 末尾的锯齿震荡
#   A3 损失     label_smoothing=0.1   不逼模型把正确类 logit 打到极端，泛化更稳
#   A4 优化器   AdamW + wd=1e-4       把权重衰减解耦出来，轻微正则
#   A5 复用     train.py 的 train_one_epoch / pick_device / set_seed，不重复造轮子
#   B1 增强     data/transforms_v2.py 的 RandomResizedCrop + RandAugment + RandomErasing
#
# 最后额外做一次 TTA 评估（原图 + 水平翻转取平均），纯推理，不用重训。

import json
import time

import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets

from config import (BATCH_SIZE, HISTORY_DIR, NUM_CLASSES, NUM_WORKERS,
                    PROCESSED_DIR, PROJECT_ROOT, RAW_TEST_DIR, SEED)
from data.transforms import eval_transform
from data.transforms_v2 import train_transform_v2
from evaluate import evaluate, plot_history
from model import MyModel
from train import pick_device, set_seed, train_one_epoch

# ---------- v2 实验专属超参（不改 config.py，v1 的配置原样保留）----------
EPOCHS_V2 = 35
LR_V2 = 1e-4
WEIGHT_DECAY_V2 = 1e-4
LABEL_SMOOTHING = 0.1
TAG = "AdamW_v2"
CHECKPOINT_V2 = PROJECT_ROOT / "checkpoints" / "best_model_v2.pth"
V1_TEST_ACC = 0.7128          # v1 提交成绩，用来打印提升幅度


# 数据加载的工程参数（这里踩过一个大坑，结论反直觉，值得记进笔记）
#
# 现象：train_transform_v2 单张只要 3.67ms（2425 张约 9 秒），但 DataLoader
#       读 320 张却要 31.5 秒。tools/bench_transforms.py 和 tools/bench_loader.py
#       两个脚本量出来的结论是：
#
#         num_workers=0（主进程直接读）  320 张   1.8 秒   <- 最快
#         num_workers=6                  320 张  31.5 秒   <- 慢 17 倍
#
# 原因在 Windows：多进程走的是 spawn 而不是 fork，每个 worker 都要重新 import
# 一遍 torch（几百 ms 起步），每个 epoch 还要重建，再加上跨进程 pickle 搬运
# 大张量（一个 batch 就是 32x3x224x224 的 float，很贵）。
# 而这台机器是 SSD + 强单核，串行读图本来就到 183 张/秒，根本喂得饱 GPU。
#
# 所以：Windows 上的小数据集训练，num_workers=0 往往比多进程快得多。
#       Linux 上没有这个问题（fork 几乎零成本），那边才该开多进程。
LOAD_WORKERS = 0


def build_loaders(batch_size: int = BATCH_SIZE, num_workers: int = LOAD_WORKERS):
    """与 data.dataset.make_loaders 逻辑一致，区别只有两处：
       train 换成 v2 强增强管道；DataLoader 开了 pin_memory 加速显存搬运。
    """
    train_ds = datasets.ImageFolder(str(PROCESSED_DIR / "train"),
                                    transform=train_transform_v2)
    val_ds = datasets.ImageFolder(str(PROCESSED_DIR / "val"),
                                  transform=eval_transform)
    test_ds = datasets.ImageFolder(str(RAW_TEST_DIR),
                                   transform=eval_transform)

    assert train_ds.class_to_idx == val_ds.class_to_idx == test_ds.class_to_idx, \
        "三个数据集的类别映射不一致，检查 processed 目录是否完整"

    # pin_memory=True：把 CPU 内存里的 batch 钉住，搬到显存时走更快的通道
    common = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, **common)
    return train_loader, val_loader, test_loader, train_ds.class_to_idx


def tta_accuracy(model, loader, device) -> float:
    """TTA：原图 + 水平翻转各推理一次，logits 相加后再取 argmax。纯推理，不训练。

    为什么能涨点：水平翻转不改变"这是朵什么花"，但会让模型的小误差换一个方向，
    两次平均相当于把随机错误抵消掉一部分。
    """
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs) + model(torch.flip(inputs, dims=[3]))
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return correct / total


def main() -> None:
    set_seed(SEED)
    device = pick_device()
    train_loader, val_loader, test_loader, class_to_idx = build_loaders()
    print(f"类别映射: {class_to_idx}")

    model = MyModel(NUM_CLASSES).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_V2,
                                  weight_decay=WEIGHT_DECAY_V2)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS_V2)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    CHECKPOINT_V2.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, EPOCHS_V2 + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, device, criterion, optimizer)
        val_loss, val_acc = evaluate(model, val_loader, device, criterion)

        scheduler.step()          # 注意：调度器用 epoch 的 val 结果更新，不是每个 batch
        lr_now = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), CHECKPOINT_V2)

        print(f"epoch {epoch:>2}/{EPOCHS_V2} | lr {lr_now:.2e} | "
              f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.4f} | {time.time() - t0:.1f}s")

    print(f"\n最佳验证准确率 {best_val_acc:.4f}（v1 Adam 是 0.6946）")

    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    hist_path = HISTORY_DIR / f"history_{TAG}.json"
    hist_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"训练历史已存: {hist_path}")

    plot_history(history, tag=TAG)

    # 用最优权重在 test 上出最终成绩。test loss 用不带 smoothing 的普通损失，
    # 这样和 v1 的成绩能直接对比（smoothing 会把 loss 抬高，不可比）。
    model.load_state_dict(torch.load(CHECKPOINT_V2, map_location=device))
    plain_loss, plain_acc = evaluate(model, test_loader, device, nn.CrossEntropyLoss())
    tta_acc = tta_accuracy(model, test_loader, device)

    print(f"\n最终成绩（test 集）  : loss {plain_loss:.4f}, 准确率 {plain_acc:.4f}")
    print(f"TTA（原图+水平翻转）  : 准确率 {tta_acc:.4f}")
    print(f"对比 v1 提交成绩 {V1_TEST_ACC:.4f} -> {plain_acc:.4f} "
          f"({plain_acc - V1_TEST_ACC:+.4f})")


if __name__ == "__main__":
    main()
