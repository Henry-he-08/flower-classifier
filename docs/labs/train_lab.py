# 训练循环实验室 —— 把 train.py 要写的四样东西，逐个用玩具数据看清楚
#
# 这个文件不碰你的数据集，全部用假数据。它只解释「机制」，
# 你的 train_one_epoch() 仍然要自己写 —— 那才是任务三的核心。
#
# 四个实验各自对应你 train.py 里的一处 TODO：
#   实验 A -> main() 里的 criterion
#   实验 B -> train_one_epoch() 的第一步（为什么必须清零）
#   实验 C -> train_one_epoch() 的最后一步（optimizer 改了什么）
#   实验 D -> model.train() / model.eval() 到底在切什么开关
#
# 运行：D:\ML\.venv_gpu\Scripts\python.exe docs\labs\train_lab.py

import torch
from torch import nn

torch.manual_seed(0)
LINE = "=" * 64


# ---------------------------------------------------------------- 实验 A
print(LINE)
print("Experiment A : what does CrossEntropyLoss actually eat?")
print(LINE)

# 模型最后输出的原始实数，叫 logits —— 没经过任何归一化
logits = torch.tensor([[2.0, 1.0, 0.1]])
y = torch.tensor([0])                      # 正确答案 = 第 0 类

crit = nn.CrossEntropyLoss()
loss_raw = crit(logits, y)                                   # 正确：直接喂 logits
loss_double = crit(torch.softmax(logits, dim=1), y)          # 错误：喂了 softmax 之后再套

prob = torch.softmax(logits, dim=1)[0]
print(f"logits               = {logits.tolist()}")
print(f"softmax(logits)      = {[round(v, 4) for v in prob.tolist()]}")
print()
print(f"feed logits          -> loss = {loss_raw:.4f}      <-- correct")
print(f"feed softmax(logits) -> loss = {loss_double:.4f}      <-- softmax twice, loss inflated")
print(f"manual -log(p_true)  = {-torch.log(prob[0]):.4f}      <-- matches the correct one")
print()
print("Takeaway: CrossEntropyLoss already contains log_softmax inside.")
print("          So NEVER add Softmax at the end of your forward().")


# ---------------------------------------------------------------- 实验 B
print()
print(LINE)
print("Experiment B : gradients ACCUMULATE -- why zero_grad() first?")
print(LINE)

w = torch.tensor([1.0], requires_grad=True)

(w * 2).backward()
print(f"after 1st backward          w.grad = {w.grad.item():.1f}")

(w * 3).backward()                 # 不清零，直接再来一次
print(f"2nd backward, no zeroing    w.grad = {w.grad.item():.1f}   <-- 2 + 3, old one still here!")

w.grad = None                      # 这就是 zero_grad() 干的事
(w * 3).backward()
print(f"zeroed, then 3rd backward   w.grad = {w.grad.item():.1f}   <-- only this step")
print()
print("Takeaway: .backward() does grad += , NOT grad = .")
print("          Forget zero_grad() and every step carries all history.")


# ---------------------------------------------------------------- 实验 C
print()
print(LINE)
print("Experiment C : what does optimizer.step() change?")
print(LINE)

# 只有一个参数 w，损失 L = w^2，所以 dL/dw = 2w
w = torch.tensor([5.0], requires_grad=True)
opt = torch.optim.SGD([w], lr=0.1)          # 学习率 0.1

print(f"start w = {w.item():.4f}")
for i in range(3):
    opt.zero_grad()                          # (1) 清梯度
    loss = w ** 2                            # (2) 前向 + 算损失
    loss.backward()                          # (3) 反向：w.grad = 2w
    g = w.grad.item()
    opt.step()                               # (4) 更新：w <- w - lr * grad
    print(f"  step {i+1}: grad={g:7.4f}  ->  w = {w.item():7.4f}")
print()
print("Takeaway: SGD update rule is just  w <- w - lr * grad .")
print("          Adam/SGD-momentum differ only in HOW they turn grad into that step.")


# ---------------------------------------------------------------- 实验 D
print()
print(LINE)
print("Experiment D : what does model.train() / model.eval() switch?")
print(LINE)

drop = nn.Dropout(p=0.5)
x = torch.ones(1, 8)

drop.train()
print(f"train() mode : {drop(x).tolist()[0]}")
print(f"train() mode : {drop(x).tolist()[0]}   <-- different each call, half zeroed")

drop.eval()
print(f"eval()  mode : {drop(x).tolist()[0]}   <-- passes through unchanged, reproducible")
print()
print("Takeaway: train() turns Dropout/RandomFlip-like randomness ON;")
print("          eval()  turns it OFF. Forget this line and your val score lies.")


# ---------------------------------------------------------------- 收尾
print()
print(LINE)
print("So your four hooks are:")
print("  A -> criterion = ???          (in main())")
print("  B -> the first line inside the batch loop  (zero_grad)")
print("  C -> the last line inside the batch loop   (step)")
print("  D -> one line at the top of train_one_epoch()")
print(LINE)
