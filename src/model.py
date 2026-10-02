# 任务 2：按结构图补全网络（AlexNet 变体，输出层换成我们的类别数）
#
# 结构图 -> PyTorch 的翻译规则（只有一条公式）：
#
#     输出边长 = floor((输入边长 + 2*padding - kernel) / stride) + 1
#
# 反推 padding 的三步（这是本题唯一需要「算」的地方）：
#   1) 抄：从结构图上抄下 In / K / S，以及它要求的 Out
#   2) 试：把 P = 0, 1, 2, 3, ... 逐个代进公式，看哪个能让 Out 对上
#   3) 选：因为公式里有向下取整，可能不止一个 P 都能命中 —— 取最小的那个
#
# 把图上「写明的数字」抄下来（通道数、核大小、步长、每层进出尺寸），
# 图上「没写的」padding 用公式反推 —— 这就是这个题的全部工作量：
#
#   输入 224x224x3（图）
#   C1:  k=11 s=4, 96 通道  -> 图要 55x55x96   ⚠️ 陷阱：pad=0 代公式得 54，
#                                                    差 1！怎么补回来？(docs\size_lab.py)
#   pool1: k=3 s=2          -> 图要 27x27x96
#   C2:  k=5, 256 通道      -> 图要 27x27x256  (尺寸不变 -> padding 反推)
#   pool2: k=3 s=2          -> 图要 13x13x256
#   C3:  k=3, 384 通道      -> 图要 13x13x384  (尺寸不变 -> 反推)
#   C4:  k=3, 384 通道      -> 图要 13x13x384  (同上)
#   C5:  k=3, 256 通道      -> 图要 13x13x256  (同上)
#   pool3: k=3 s=2          -> 图要 6x6x256
#   展平: 256 * 6 * 6 = 9216 -> FC6(4096) -> FC7(4096) -> FC8(类别数)
#
# 结构图的 FC8 是 18，是出题人模板的设定；本数据集只有 5 类，
# 输出维度必须 = NUM_CLASSES（config 里），这才是"分类头"的意义。

import torch
import torch.nn as nn

from config import NUM_CLASSES


class MyModel(nn.Module):
    """按结构图实现的 CNN。输入 [B, 3, 224, 224]，输出 [B, num_classes]。"""

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()

        # ---------- 卷积部分 ----------
        # C1: 3 通道 -> 96 张特征图，11x11 大核 + 步长 4（大步长快速压缩，经典开局）
        #
        # ⚠️ self.conv1 里要装的是「一个层对象」，不是一串数字。
        #    参数顺序：in_channels, out_channels, kernel_size, stride, padding
        
        # ⚠️ kernel_size=11 这种 key=value 写法，只有写在「函数调用的括号里」才合法
        #    —— 括号前面必须有 nn.Conv2d 这个名字。少了名字，就是语法错误。
        # TODO: 写出 self.conv1（3 -> 96，kernel_size=11，stride=4，padding 你自己反推）
        self.conv1 = nn.Conv2d (3 ,96,kernel_size=11,stride=4,padding=2)

        # 池化核 3、步长 2 -> 55 -> 27。池化层没有可学习参数，
        # 所以五段卷积共用同一个 self.pool 即可（想想为什么能共用）
        # TODO: 写出 self.pool。注意类名大小写（Pool 的 P 大写，不是 Maxpool）
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)


        # C2: 96 -> 256 通道，5x5 核。图上 27 -> 27 尺寸不变，
        #     代公式反推：padding 该填几？（用 docs\size_lab.py 验证你的推算）
        # TODO
        self.conv2 =nn.Conv2d (96,256,kernel_size=5,padding=2)

        # C3、C4: 256 -> 384 -> 384，3x3 核。图上 13 -> 13，
        #     同样用公式反推 padding（3x3 核保尺寸的 padding 有个口诀，试出来）
        # TODO
        self.conv3 =nn.Conv2d(256,384,kernel_size=3,padding=1)
        self.conv4 =nn.Conv2d (384,384,kernel_size=3,padding=1)

        # C5: 384 -> 256，3x3 核，图上 13 -> 13
        # TODO
        self.conv5 =nn.Conv2d (384,256,kernel_size=3,padding=1)

        # ---------- 全连接部分 ----------
        # 展平后 256*6*6 = 9216 -> 4096 -> 4096 -> num_classes
        # 建议用 nn.Sequential 把三层 Linear 和激活拼在一起，或拆成 fc1/fc2/fc3
        # 思考（写进笔记）：两个 4096 层之间要不要加 nn.Dropout(p=0.5)？
        #                   AlexNet 原论文加了 —— 4096x4096 的全连接是参数大户，
        #                   小数据集上不dropout基本必过拟合
        
        self.classifier = nn.Sequential(
            nn.Linear(9216, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),

            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),

            nn.Linear(4096, num_classes),
)


        # 所有卷积层后面都要接非线性激活，整体复用这一个即可
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO：按结构图把上面的层串起来。骨架提示：
        #
        x = self.pool(self.relu(self.conv1(x)))   # -> [B, 96, 27, 27]
        x = self.pool(self.relu(self.conv2(x)))   # -> [B, 256, 13, 13]
        x=self.relu(self.conv3(x))  
        x=self.relu(self.conv4(x))                                        # C3、C4 不接池化
        x = self.pool(self.relu(self.conv5(x)))   # -> [B, 256, 6, 6]
        #
        x = torch.flatten(x, 1)                   # [B, 256, 6, 6] -> [B, 9216]
        #                                             # 参数 1 = 从第 1 维开始展平，
        #                                             # 保留 batch 维！
        x = self.classifier(x)                    # -> [B, num_classes]
        return x
        #
        # 注意：最后不要加 Softmax！CrossEntropyLoss 内部自带
        #       log_softmax（Titanic 的 BCEWithLogitsLoss 同理）
        


# ---------------- 自检探针：模型写完跑这个，全对才能开训 ----------------
if __name__ == "__main__":
    model = MyModel()
    print(model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n可学习参数总量: {n_params:,}")

    # 喂一张假图过一遍前向，任何一层尺寸接错都会在这里当场报错
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    print(f"输入 {tuple(x.shape)} -> 输出 {tuple(out.shape)}")
    assert out.shape == (1, NUM_CLASSES), f"输出形状不对，应为 (1, {NUM_CLASSES})"
    print("自检通过 ✓ 可以开训")
