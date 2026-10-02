# 任务 3 收尾：评估 + 画准确率曲线
#
# 本文件三个职责：
#   1) evaluate()        —— 在任意 loader 上算 (平均loss, 准确率)，训练中被 val 用
#   2) plot_history()    —— 画 train/val 两条准确率（和 loss）随 epoch 的曲线
#   3) main()            —— 训练完成后：加载最优模型，在 test 上出最终成绩

import matplotlib
import matplotlib.pyplot as plt
import torch
from torch import nn          # main() 里要用 nn.CrossEntropyLoss()

from config import (CHECKPOINT_PATH, DEVICE_AUTO, FIGURES_DIR, NUM_CLASSES)
from data.dataset import make_loaders
from model import MyModel

# 无显示环境也能出图（Windows 下无所谓，留着不碍事）
matplotlib.use("Agg")


def evaluate(model, loader, device, criterion) -> tuple[float, float]:
    

    model.eval()
    correct = total = total_loss = 0
    for inputs, labels in loader:
          inputs, labels = inputs.to(device), labels.to(device)

          with torch.no_grad():          # 评估不需要梯度，省显存又提速
              outputs = model(inputs)    # [B, num_classes] logits
              loss = criterion(outputs, labels)

          total_loss += loss.item() * inputs.size(0)   # 按样本数加权累加
          correct += (outputs.argmax(dim=1) == labels).sum().item()
          total += labels.size(0)



    return total_loss / total, correct / total
    
    


def plot_history(history: dict, tag: str = "") -> None:
    """画 train/val 的 accuracy 与 loss 曲线，存到 docs/ 下。

    两个子图各两条线：acc 图用 history["train_acc"] / ["val_acc"]，
    loss 图用 history["train_loss"] / ["val_loss"]。
    记得给 label（"train" / "val"）—— 不给 label 图例就是空的。

    tag: 可选标签（如优化器名 "Adam"）。给了就存成 fig_curve_<tag>.png，
         换一组实验不会把上一组的图覆盖掉；不给则沿用旧文件名。
    """
    epochs = range(1, len(history["train_acc"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(epochs, history["train_acc"], label="train")
    ax1.plot(epochs, history["val_acc"],   label="val")
    ax2.plot(epochs, history["train_loss"], label="train")
    ax2.plot(epochs, history["val_loss"],   label="val")

    ax1.set_title("Accuracy")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("acc")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax2.set_title("Loss")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("loss")
    ax2.legend()
    ax2.grid(alpha=0.3)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / (f"fig_curve_{tag}.png" if tag else "fig_accuracy_curve.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"曲线图已保存: {out}")


def main() -> None:
    # 同 train.pick_device：想要 cuda，没有显卡就回落 cpu
    if DEVICE_AUTO == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(DEVICE_AUTO)
    _, _, test_loader, _ = make_loaders()

    model = MyModel(NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    print(f"已加载最优模型: {CHECKPOINT_PATH}")

    # test 上的损失用 CrossEntropyLoss 算（和训练同一个函数即可）
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate(model, test_loader, device, criterion)
    print(f"\n最终成绩（test 集）: loss {test_loss:.4f}, 准确率 {test_acc:.4f}")


if __name__ == "__main__":
    main()
