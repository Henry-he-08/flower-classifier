# 尺寸实验室（讲题配套，不是作业答案）
#
# 只回答一个问题：Conv2d 的 kernel / stride / padding 怎么决定输出尺寸？
#     输出边长 = floor((输入边长 + 2*padding - kernel) / stride) + 1
#
# 运行：
#   D:\ML\.venv_gpu\Scripts\python.exe docs\labs\size_lab.py
#
# 用法：改下面任何参数重跑，眼睛盯着形状变化，公式就长在手里了。

import torch
import torch.nn as nn

print("=" * 62)
print("实验 A：在 32x32 的图上看三个参数各自怎么改尺寸（与题目无关）")
print("=" * 62)
x = torch.randn(1, 3, 32, 32)
print("输入               :", tuple(x.shape))

demo = [
    ("基准   k=3 s=1 p=0", dict(kernel_size=3, stride=1, padding=0)),
    ("加pad  k=3 s=1 p=1", dict(kernel_size=3, stride=1, padding=1)),
    ("加pad  k=3 s=1 p=2", dict(kernel_size=3, stride=1, padding=2)),
    ("大步长 k=3 s=2 p=0", dict(kernel_size=3, stride=2, padding=0)),
    ("大核   k=5 s=1 p=0", dict(kernel_size=5, stride=1, padding=0)),
    ("大核   k=5 s=1 p=2", dict(kernel_size=5, stride=1, padding=2)),
]
for name, kw in demo:
    conv = nn.Conv2d(3, 8, **kw)
    out = conv(x)
    # 手算公式对照：PyTorch 的实际行为 vs 公式，两边必须一致
    h = x.shape[2]
    k, s, p = kw["kernel_size"], kw["stride"], kw["padding"]
    formula = (h + 2 * p - k) // s + 1
    real = out.shape[2]
    flag = "OK" if formula == real else "!! 公式和实际不一致"
    print(f"{name} : {tuple(out.shape)}   公式={formula}  {flag}")

print()
print("=" * 62)
print("实验 B：池化的尺寸公式和卷积同款（kernel=3, stride=2）")
print("=" * 62)
pool = nn.MaxPool2d(kernel_size=3, stride=2)
for h in (55, 27, 13):
    y = pool(torch.randn(1, 1, h, h))
    formula = (h - 3) // 2 + 1
    print(f"pool 输入 {h:>2} -> 输出 {tuple(y.shape)[2]:>2}   公式={formula}")

print()
print("=" * 62)
print("实验 C：你的推算台（改这里的数字，验证你给 conv1/conv2 定的参数）")
print("=" * 62)
# 目标：让 conv1 输出 55x55、pool1 后 27x27、conv2 输出 27x27。
# 结构图的已知条件：conv1 k=11 s=4（96 通道）；pool k=3 s=2；conv2 k=5（256 通道）。
# 你要自己定的是 conv1 和 conv2 的 padding。改空格里的数，跑到和图一致为止。
CONV1_PADDING = 0   # TODO: 换成你的推算
CONV2_PADDING = 0   # TODO: 换成你的推算

z = torch.randn(1, 3, 224, 224)
c1 = nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=CONV1_PADDING)
z1 = c1(z)
print(f"conv1 (k11 s4 pad={CONV1_PADDING}) : {tuple(z1.shape)}   <- 图要 (1, 96, 55, 55)")
z2 = nn.MaxPool2d(3, 2)(z1)
print(f"pool1 (k3 s2)            : {tuple(z2.shape)}   <- 图要 (1, 96, 27, 27)")
c2 = nn.Conv2d(96, 256, kernel_size=5, padding=CONV2_PADDING)
z3 = c2(z2)
print(f"conv2 (k5  pad={CONV2_PADDING}) : {tuple(z3.shape)}   <- 图要 (1, 256, 27, 27)")

ok = z1.shape[2] == 55 and z2.shape[2] == 27 and z3.shape[2] == 27
print("\n结论:", "和结构图对上了，把这两个 padding 抄进 model.py"
      if ok else "还没对上，回头检查推算（提示：conv1 那个'差 1'的陷阱）")
