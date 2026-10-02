r"""对比不同优化器的训练曲线 —— 从 train.py 落盘的 history_*.json 重画，不用重训。

用法（在 D:\ML.2 下）：
    D:\ML\.venv_gpu\Scripts\python.exe docs\labs\compare_history.py

它会扫描 docs\logs\ 下所有 history_<优化器名>.json，画成一张对照图：
    左：train/val 准确率        右：train/val 损失
每组的 train 用实线、val 用虚线，颜色区分优化器；
准确率图里那条灰点线是「五分类瞎猜 = 20%」的基线，一眼就能看出学没学到。

输出：docs\figures\fig_compare_optimizers.png

想再加一组实验：换 optimizer 跑一次 train.py 即可（会自动落盘 history_XXX.json），
再运行本脚本，新曲线自动出现在图上。
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # 无显示环境也能出图
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]        # D:\ML.2
LOGS_DIR = ROOT / "docs" / "logs"                 # train.py 落盘的 history_*.json
FIGURES_DIR = ROOT / "docs" / "figures"           # 对照图输出


def load_histories() -> dict:
    """读回 logs 目录下所有 history_*.json，键 = 优化器名。"""
    runs = {}
    for p in sorted(LOGS_DIR.glob("history_*.json")):
        name = p.stem.replace("history_", "")
        runs[name] = json.loads(p.read_text(encoding="utf-8"))
    return runs


def main() -> None:
    runs = load_histories()
    if not runs:
        print("没找到任何 history_*.json —— 先跑一次 train.py（会落盘），再回来运行本脚本")
        return

    print(f"找到 {len(runs)} 组实验: {', '.join(runs)}\n")
    for name, h in runs.items():
        n = len(h["train_acc"])
        best = max(h["val_acc"])
        print(f"  {name:<6} {n:>2} 轮 | 末轮 train {h['train_acc'][-1]:.4f} / "
              f"val {h['val_acc'][-1]:.4f} | 最佳 val {best:.4f} "
              f"(epoch {h['val_acc'].index(best) + 1})")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

    for i, (name, h) in enumerate(runs.items()):
        c = colors[i % len(colors)]
        ep = range(1, len(h["train_acc"]) + 1)
        ax1.plot(ep, h["train_acc"],  color=c, linestyle="-",  label=f"{name} train")
        ax1.plot(ep, h["val_acc"],    color=c, linestyle="--", label=f"{name} val")
        ax2.plot(ep, h["train_loss"], color=c, linestyle="-",  label=f"{name} train")
        ax2.plot(ep, h["val_loss"],   color=c, linestyle="--", label=f"{name} val")

    # 五分类瞎猜的基线：低于它 = 模型啥也没学到
    ax1.axhline(1 / 5, color="gray", linestyle=":", linewidth=1.2,
                label="baseline (random 20%)")

    for ax, title, ylab in ((ax1, "Accuracy", "acc"), (ax2, "Loss", "loss")):
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylab)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "fig_compare_optimizers.png"
    fig.savefig(out, dpi=150)
    print(f"\n对照图已保存: {out}")


if __name__ == "__main__":
    main()
