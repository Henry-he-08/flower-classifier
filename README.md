# 花卉小能手 · 花卉图片分类器（CNN）

从零实现的一个 **5 分类卷积神经网络**（AlexNet 变体），完整覆盖
「数据清洗 → 划分 → 增强 → 建模 → 训练 → 评估」全流程。

> 凌睿工作室招新 · 机器学习方向 · 任务二 / 任务三

---

## 结果速览

| 项目 | 结果 |
|---|---|
| 数据集 | 5 类花卉（daisy / dandelion / rose / sunflower / tulip），原始 3827 张 |
| 坏图处理 | 扫描出 **13 张**损坏图片，全部隔离（不删除）+ CSV 报告 |
| 数据划分 | train **2425** / val **609**（每类各 20%，分层保持） |
| 输入尺寸 | 统一 `3 × 224 × 224` |
| 数据增强 | RandomHorizontalFlip + RandomRotation(±15°) + ColorJitter |
| 模型规模 | AlexNet 变体，**58,301,829** 个可学习参数 |
| 最佳验证准确率 | **69.46%**（Adam, epoch 12） |
| **测试集准确率** | **71.28%** |
| 随机基线 | 20% |

**优化器对照实验**（固定种子 42，唯一变量是优化器与学习率）：

| 配置 | 末轮 val acc | 结论 |
|---|---:|---|
| SGD lr=1e-4 | 0.2939 | ❌ 完全没学到（lr 与优化器不配对） |
| **Adam lr=1e-4** | **0.6749**（最佳 0.6946） | ✅ 最优 |
| SGD lr=1e-2 | 0.6732 | ✅ 正常收敛 |

![优化器对照曲线](docs/figures/fig_compare_optimizers.png)

---

## 环境

本项目在以下环境验证通过：

| 项 | 版本 |
|---|---|
| 操作系统 | Windows |
| Python | 3.12.14 |
| PyTorch | 2.14.0+cu130 |
| torchvision | 0.29.1+cu130 |
| matplotlib / numpy / Pillow / scikit-learn | 见 `requirements.txt` |
| GPU | NVIDIA GeForce RTX 5060 Laptop（CUDA 可用） |

**注意**：本项目使用了机器上的独立虚拟环境，可按自己的环境调整解释器路径。
下文命令中出现的 `D:\ML\.venv_gpu\Scripts\python.exe` 是作者本机的解释器路径，
**请替换为你自己的 Python 解释器**（关键是要装好 `requirements.txt` 里的依赖）。

---

## 数据

数据**不包含在本仓库中**（体积大且属第三方数据）。目录结构要求：

```
<数据根目录>\
├── train\
│   ├── daisy\
│   ├── dandelion\
│   ├── rose\
│   ├── sunflower\
│   └── tulip\
└── test\
    ├── daisy\
    ├── ...（同上 5 类）
```

数据根目录在 `src/config.py` 里配置（`RAW_TRAIN_DIR` / `RAW_TEST_DIR`）。

> **数据集只读**：处理过程中的坏图是"移动"到隔离区，不删除原始文件。

---

## 快速开始

```bash
# 0) 安装依赖
pip install -r requirements.txt

# 1) 数据处理（按顺序）
python src/data/clean.py        # 扫描并隔离坏图，生成 CSV 报告
python src/data/split.py        # 划分 train / val（幂等：每次重跑先清场）
python src/data/dataset.py      # 自检数据管道（可选）
python src/model.py             # 自检网络尺寸（可选，全绿才建议开训）

# 2) 训练 + 评估
python src/train.py             # 训练并落盘曲线、训练历史
python src/evaluate.py          # 在 test 集上出最终成绩

# 3) 重画多组对照曲线（读 docs/logs/*.json，不必重训）
python docs/labs/compare_history.py
```

每个关键脚本都带 `if __name__ == "__main__":` 自检探针，
**不依赖真实数据即可验证尺寸 / 形状 / 批次逻辑**。

---

## 目录结构

```
ML.2/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── config.py              全局配置：路径 + 全部超参数（调参只动这里）
│   ├── model.py               网络结构（AlexNet 变体）
│   ├── train.py               训练主入口
│   ├── evaluate.py            评估 + 画曲线
│   └── data/
│       ├── clean.py           1-1 坏图扫描 / 隔离
│       ├── split.py           1-2 划分 train / val
│       ├── transforms.py      1-3 增强 + 1-4 尺寸统一
│       └── dataset.py         1-5 Dataset / DataLoader
├── tools/
│   └── md_to_pdf.py           文档 md → PDF（系统自带 Edge 无头模式，零安装）
└── docs/
    ├── 提交报告.md             ★ 主报告（完整版）
    ├── 训练过程记录.md          三组训练的完整配置、逐轮数据、结论
    ├── 学习过程记录.md          概念理解路径、验证脚本、踩坑与修正
    ├── 概念题.md               概念问答（详版，含推导与代码）
    ├── 学习笔记.md              个人整理笔记（概念问答初稿）
    ├── labs/                  过程验证脚本（见下）
    ├── figures/               曲线图
    └── logs/                  训练历史 json（逐轮 loss / acc）
```

**运行后生成**（已在 `.gitignore` 中排除）：
`data/processed/`、`data/quarantine/`、`data/report/`、`checkpoints/`。

---

## 过程验证脚本（`docs/labs/`）

这些脚本是理解每个概念时**用最小实验把机制跑出来**的记录：

| 脚本 | 验证了什么 |
|---|---|
| `size_lab.py` | 用真 torch 逐层验算输出尺寸，反推每层 padding（含 conv1 的"差 1"陷阱） |
| `dataloader_lab.py` | 用假数据验证 `batch_size` / `shuffle` / `drop_last` / `num_workers` 各自改变了什么 |
| `train_lab.py` | 验证 `backward()` 是**累加**（所以必须 `zero_grad()`）、`step()` 如何改参数、`train()`/`eval()` 的差别 |
| `lr_lab.py` | 固定数据与初始权重，只换优化器与 lr，定位"学不动"的根因 |
| `compare_history.py` | 从落盘的训练历史重画多组对照曲线，不必重训 |

---

## 文档索引

| 想了解什么 | 看哪份 |
|---|---|
| 完整的方法、推导、实验与结论 | [`docs/提交报告.md`](docs/提交报告.md) |
| 每组训练怎么配、每一轮数字是多少 | [`docs/训练过程记录.md`](docs/训练过程记录.md) |
| 我是怎么一步步搞懂的、踩过哪些坑 | [`docs/学习过程记录.md`](docs/学习过程记录.md) |
| 8 道概念题的完整回答 | [`docs/概念题.md`](docs/概念题.md) |

---

## 复现要点

- **随机种子固定为 42**（`src/config.py` 的 `SEED`），任何人任何时间跑都能得到相同结果；
- **数据划分幂等**：`split.py` 开头先 `reset_output()` 清空旧输出，跑一次和跑十次结果一致；
- **增强只加在训练集**：`train_transform` 含随机算子，`eval_transform` 不含（保证评估可复现）；
- **训练历史自动落盘**：`docs/logs/history_<优化器名>.json`，方便对比不同配置。

---

## 已知问题与后续改进

1. **检查点未按优化器命名** —— `checkpoints/best_model.pth` 每次训练会被覆盖；
   改进：按 `history_*.json` 的方式加上优化器名。
2. **`EPOCHS=15` 偏少** —— val 曲线到末轮仍在上升，可试 25 轮。
3. **数据增强仅 3 个算子** —— 可试 `RandomResizedCrop` 替代 `Resize`，或加入更强正则。
4. **未做迁移学习** —— 当前是从零训练；用预训练权重预期能大幅提升。
