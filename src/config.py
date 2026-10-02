# 花卉小能手 —— 全局配置（本文件在 src/ 下，与源码同级）
# 所有路径、超参数集中在这里，其他文件只 import，不改数。
# 调参实验时：只动这个文件，方便对比记录。

from pathlib import Path

# ============ 路径 ============
# 原始数据（百度网盘下载的那两份，只读，永不修改里面的文件）
RAW_TRAIN_DIR = Path(r"D:\BaiduNetdiskDownload\train")
RAW_TEST_DIR = Path(r"D:\BaiduNetdiskDownload\test")

# 项目输出（处理结果都放在本项目 data/ 下）
PROJECT_ROOT = Path(r"D:\ML.2")
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"   # 划分后的 train/val
QUARANTINE_DIR = PROJECT_ROOT / "data" / "quarantine"  # 损坏图片的隔离区（移动，不删除）
REPORT_DIR = PROJECT_ROOT / "data" / "report"          # 损坏图片清单 CSV
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "best_model.pth"
FIGURES_DIR = PROJECT_ROOT / "docs" / "figures"        # 曲线图输出
HISTORY_DIR = PROJECT_ROOT / "docs" / "logs"           # 训练历史 json 输出

# ============ 数据处理超参数 ============
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}  # 题目默认 jpg，稳妥起见把常见格式都算上
VAL_RATIO = 0.2                          # 训练集里划出 20% 做验证
SEED = 42                                # 随机种子：保证划分、训练可复现（Titanic 同款做法）

# ============ 设备 ============
DEVICE_AUTO = "cuda"   # "cuda" 优先；机器上没有显卡时 train.py 会自动回落到 cpu

# ============ 模型 / 训练超参数（任务三的重点实验对象）============
IMG_SIZE = 224          # 结构图输入是 224x224，resize 到这个尺寸
NUM_CLASSES = 5         # ⚠️ 结构图 FC8 输出 18，但本数据集只有 5 类花
                        #    （daisy/dandelion/rose/sunflower/tulip）
                        #    思考：为什么输出维度必须等于类别数？
BATCH_SIZE = 32         # 批大小：受显存限制；报 CUDA out of memory 就调小
NUM_WORKERS = 2         # DataLoader 后台读图进程数；Windows 下别开太大
EPOCHS = 15             # 轮数：先跑通再调。观察曲线：还在涨 → 加；验证集开始掉 → 减
LEARNING_RATE = 1e-4    # 初始学习率建议 1e-4（Adam）；太大不收敛、太小太慢
