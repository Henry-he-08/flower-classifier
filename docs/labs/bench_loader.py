# 诊断：DataLoader 各种 worker 配置下，实际吞吐差多少
#
# 现象：train_transform_v2 单张只要 3.67ms（2425 张约 9 秒），
#       但 DataLoader 读 320 张却要 31.5 秒。说明瓶颈不在 transform，
#       而在多进程搬运 / 磁盘 IO。
#
# 这个脚本扫一遍常见配置，找出真正快的那组。
#
# 运行：
#   D:\ML\.venv_gpu\Scripts\python.exe docs\labs\bench_loader.py

import sys
import time
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from config import PROCESSED_DIR
from data.transforms_v2 import train_transform_v2

def run(loader, target: int = 320) -> float:
    """统计从 loader 里读满 target 张图要多少秒"""
    t0 = time.time()
    seen = 0
    for images, _ in loader:
        seen += images.size(0)
        if seen >= target:
            break
    return time.time() - t0


def main() -> None:
    ds = datasets.ImageFolder(str(PROCESSED_DIR / "train"),
                              transform=train_transform_v2)
    print(f"训练集 {len(ds)} 张，每批 32，统计读满 320 张的耗时\n")

    configs = [
        ("主进程读（不开 worker）", dict(num_workers=0, pin_memory=False)),
        ("2 workers（v1 的配置）", dict(num_workers=2, pin_memory=False)),
        ("4 workers", dict(num_workers=4, pin_memory=False)),
        ("6 workers", dict(num_workers=6, pin_memory=False)),
        ("6 workers + pin_memory", dict(num_workers=6, pin_memory=True)),
        ("8 workers", dict(num_workers=8, pin_memory=False)),
    ]

    print(f"{'配置':<28}{'第1轮(s)':>10}{'第2轮(s)':>10}{'张/秒':>10}")
    for name, kw in configs:
        loader = DataLoader(ds, batch_size=32, shuffle=True, **kw)
        t1 = run(loader)      # 第 1 轮：worker 刚 spawn，含启动开销
        t2 = run(loader)      # 第 2 轮：worker 已就绪
        print(f"{name:<28}{t1:>10.1f}{t2:>10.1f}{320 / min(t1, t2):>10.0f}")

    print("\n注：'张/秒' 取两轮里较快的一次，数值越高越好。")


# Windows 上必须写这个保护！否则 DataLoader 每次 spawn 子进程时，子进程会重新
# 执行整个主模块，撞上 multiprocessing 的 bootstrapping phase 检查，直接抛
# RuntimeError。Linux 走 fork 不受影响 —— 这就是这个脚本第一次在 Windows 上
# 跑崩的原因（而 train_v2.py 因为有同样的保护，所以没事）。
if __name__ == "__main__":
    main()
