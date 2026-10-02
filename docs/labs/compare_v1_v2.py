# 对比 v1（Adam，15 轮，基础增强）与 v2（AdamW + 强增强 + 余弦退火，35 轮）
#
# v1 成绩：val 最佳 0.6946 / test 0.7128   （报告里的提交成绩）
# v2 成绩：val 最佳 0.7865 / test 0.7910
#
# 输出：docs/figures/fig_compare_v1_v2.png
#       （新文件名，不覆盖已有的 fig_compare_optimizers.png）
#
# 运行：
#   D:\ML\.venv_gpu\Scripts\python.exe docs\labs\compare_v1_v2.py

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from config import FIGURES_DIR, HISTORY_DIR

# 两条曲线的配色固定：v1 红、v2 蓝
C1, C2 = "#c0392b", "#1f6feb"


def load(name: str) -> dict:
    return json.loads((HISTORY_DIR / name).read_text(encoding="utf-8"))


v1 = load("history_Adam.json")
v2 = load("history_AdamW_v2.json")

e1 = list(range(1, len(v1["val_acc"]) + 1))
e2 = list(range(1, len(v2["val_acc"]) + 1))
best1, best2 = max(v1["val_acc"]), max(v2["val_acc"])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

# ---- 左图：验证准确率（看泛化能力提升）----
ax1.plot(e1, v1["val_acc"], "o-", ms=3, lw=1.5, color=C1,
         label="v1  Adam, 15 ep")
ax1.plot(e2, v2["val_acc"], "s-", ms=3, lw=1.5, color=C2,
         label="v2  AdamW + aug + cosine, 35 ep")
ax1.axhline(best1, color=C1, ls="--", lw=0.8)
ax1.axhline(best2, color=C2, ls="--", lw=0.8)
ax1.annotate(f"best {best1:.4f}", (16, best1 - 0.045), color=C1, fontsize=8)
ax1.annotate(f"best {best2:.4f}", (18, best2 + 0.012), color=C2, fontsize=8)
ax1.set_title("Validation accuracy")
ax1.set_xlabel("epoch")
ax1.set_ylabel("accuracy")
ax1.legend(fontsize=8)
ax1.grid(alpha=0.3)

# ---- 右图：验证损失（看收敛是否更稳）----
ax2.plot(e1, v1["val_loss"], "o-", ms=3, lw=1.5, color=C1, label="v1  Adam")
ax2.plot(e2, v2["val_loss"], "s-", ms=3, lw=1.5, color=C2,
         label="v2  AdamW + aug + cosine")
ax2.set_title("Validation loss")
ax2.set_xlabel("epoch")
ax2.set_ylabel("loss")
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)

fig.suptitle("v1 vs v2   |   test accuracy  0.7128 -> 0.7910   (+7.82 pt)",
             fontsize=10)
fig.tight_layout()

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
out = FIGURES_DIR / "fig_compare_v1_v2.png"
fig.savefig(out, dpi=150)
print(f"对比图已保存: {out}")
print(f"v1 best val {best1:.4f}  |  v2 best val {best2:.4f}  |  delta {best2 - best1:+.4f}")
