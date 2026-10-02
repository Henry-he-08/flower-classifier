# 任务 3：训练 + 记录准确率曲线
#
# 流程：读数据 -> 建模型 -> （你定）损失函数/优化器 -> 逐 epoch 训练+验证
#       -> 保存最优模型 -> 画曲线
#
# 运行（先完成 model.py / dataset.py 的 TODO）：
#   D:\ML\.venv_gpu\Scripts\python.exe src\train.py

import json
import time

import torch
from torch import nn

from config import (CHECKPOINT_PATH, DEVICE_AUTO, EPOCHS, HISTORY_DIR,
                    LEARNING_RATE, NUM_CLASSES, SEED)
from data.dataset import make_loaders
from evaluate import evaluate, plot_history
from model import MyModel


def set_seed(seed: int) -> None:
    """固定所有随机源，结果可复现（Titanic 的老规矩）。"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device() -> torch.device:
    # config 里默认想要 cuda，但如果这台机器没有可用显卡，自动回落 cpu
    if DEVICE_AUTO == "cuda" and not torch.cuda.is_available():
        print("未检测到可用显卡，回落到 CPU（训练会慢，建议排查驱动）")
        return torch.device("cpu")
    device = torch.device(DEVICE_AUTO)
    print(f"计算设备: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    return device


def train_one_epoch(model, loader, device, criterion, optimizer) -> tuple[float, float]:
    model.train()                            # 切训练模式（Dropout 生效）
    total_loss = correct = total = 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)

        
        optimizer.zero_grad()               # ① 清上一步的梯度（PyTorch 梯度是累加的！
                                            #    Titanic 实验里你量化过不清零的后果）
        outputs = model(inputs)             # ② 前向 -> [B, num_classes] 的 logits
        loss = criterion(outputs, labels)   # ③ 算损失
        loss.backward()                     # ④ 反向传播
        optimizer.step()                    # ⑤ 更新参数（严格说是第五步）

        total_loss += loss.item() * inputs.size(0)   # .item() 脱离计算图，省显存
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += inputs.size(0)
    return total_loss / total, correct / total
   
    
    

def main() :
    
    set_seed(SEED)
    device = pick_device()
    train_loader, val_loader, test_loader, class_to_idx = make_loaders()
    print(f"类别映射: {class_to_idx}")

    model = MyModel(NUM_CLASSES).to(device)

    # ==================== TODO（任务三的"超参数"实验主战场）====================
    # 1) criterion：多分类 + 输出是 logits -> nn.CrossEntropyLoss()
    #    为什么不是 Titanic 的 BCEWithLogitsLoss？（那是二分类；这是 5 分类）
    #    为什么不是 MSELoss？（回归用）
    criterion = nn.CrossEntropyLoss()
    #
    # 2) optimizer：建议至少跑两组对比（写进提交，就是你的实验记录）：
    #      torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    #      torch.optim.SGD(model.parameters(), lr=..., momentum=0.9)
    #   Titanic 里 Adam 碾压 SGD 17.9 个百分点；图像任务上结论未必一样，
    #    这正是值得实验的点
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    # =========================================================================

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, device, criterion, optimizer)

        # 验证：不更新参数，只看泛化（evaluate 内部已包 no_grad）
        val_loss, val_acc = evaluate(model, val_loader, device, criterion)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # 只保存验证集上最好的那一版（防止最后一轮恰好过拟合）
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), CHECKPOINT_PATH)

        print(f"epoch {epoch:>2}/{EPOCHS} | "
              f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.4f} | "
              f"{time.time() - t0:.1f}s")

    print(f"\n训练结束，最佳验证准确率 {best_val_acc:.4f}")
    print(f"最优模型已存: {CHECKPOINT_PATH}")

    # 把 history 落盘：以后想重画曲线不用再训一遍，也方便和下一组实验（如 SGD）对比。
    # 文件名自动带上优化器名，所以 Adam 和 SGD 的结果不会互相覆盖。
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    hist_path = HISTORY_DIR / f"history_{type(optimizer).__name__}.json"
    hist_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"训练历史已存: {hist_path}")

    plot_history(history, tag=type(optimizer).__name__)


if __name__ == "__main__":
    main()
