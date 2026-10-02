r"""DataLoader 实验室 —— 用假数据把 DataLoader 的行为看明白。

只演示 torch.utils.data 的机制：造一个「假花数据集」，不碰你的 processed/，
不动项目任何文件。看完再回去填 dataset.py 就顺手了。

运行（在 D:\ML.2 目录下，可原样粘贴）：
    D:\ML\.venv_gpu\Scripts\python.exe docs\labs\dataloader_lab.py
"""

import torch
from torch.utils.data import DataLoader, Dataset


class FakeFlower(Dataset):
    """最小的 Dataset：只需要实现两个方法。

    __len__      : "仓库里一共有多少件货"
    __getitem__  : "给我编号 i 的那一件"  -> 返回 (图片张量, 标签)

    故意把假图的像素值设成编号、标签设成 编号%3，
    这样「有没有被打乱」一眼就能看出来。
    """

    def __init__(self, n: int):
        self.n = n
        self.classes = ["daisy", "rose", "tulip"]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        img = torch.full((3, 4, 4), float(idx))   # 假图：整张图像素都等于编号
        label = idx % 3                           # 标签 0,1,2,0,1,2... 循环
        return img, label


if __name__ == "__main__":          # Windows 下多进程必须放在这个守卫里
    print("=" * 62)
    print("实验 A：Dataset 是仓库，DataLoader 是套在外面的卡车")
    print("=" * 62)
    ds = FakeFlower(10)
    print(f"len(ds)          = {len(ds)}        <- 仓库里有几件货")
    img, label = ds[3]
    print(f"ds[3]            = 图片 {tuple(img.shape)}, 标签 {label}")
    print(f"ds[0][0][0,0,0]  = {ds[0][0][0, 0, 0].item():.0f}          <- 我造的假图：像素值 = 编号")

    loader = DataLoader(ds, batch_size=4)
    print(f"\nDataLoader(batch_size=4): 共 {len(loader)} 个 batch  <- 10 件货，4 件一车")
    imgs, labels = next(iter(loader))     # next(iter(...)) = 取出第一车
    print(f"第一车: images {tuple(imgs.shape)}, labels {labels.tolist()}")
    print("  注意形状：4 是第一维（一个 batch 里几张图）—— 训练时看到的永远是 B 张一起。")

    print()
    print("=" * 62)
    print("实验 B：shuffle 到底改变了什么")
    print("=" * 62)
    for shuffle in (False, True):
        ld = DataLoader(ds, batch_size=10, shuffle=shuffle)
        print(f"shuffle={str(shuffle):<5} -> labels = {next(iter(ld))[1].tolist()}")
    print("-> False 就是 0,1,2,0,1,2... 的原顺序；True 才被打乱。")
    print("-> 打乱发生在每个 epoch 开头，每个 epoch 的顺序都不一样。")

    print()
    print("=" * 62)
    print("实验 C：drop_last —— 最后那条凑不满的尾巴")
    print("=" * 62)
    for drop in (False, True):
        ld = DataLoader(ds, batch_size=4, drop_last=drop)
        shapes = [tuple(b[0].shape) for b in ld]
        print(f"drop_last={str(drop):<5} -> batch 数 {len(ld)}, 每车形状 {shapes}")
    print("-> 10 张、4 张一车：最后一车只有 2 张。drop_last=True 就把这 2 张扔掉。")

    print()
    print("=" * 62)
    print("实验 D：num_workers —— 搬货的工人数")
    print("=" * 62)
    for nw in (0, 2):
        ld = DataLoader(ds, batch_size=4, num_workers=nw)
        imgs, labels = next(iter(ld))
        print(f"num_workers={nw} -> 照样取出 {tuple(imgs.shape)}, labels {labels.tolist()}")
    print("-> 结果完全一样，区别只在「谁来搬」：0 = 主进程自己搬（Windows 上最稳），")
    print("   >0 = 另开子进程提前把下一车装好，读图快，但 Windows 下必须写")
    print("   `if __name__ == \"__main__\":` 守卫，否则子进程会再把整个脚本跑一遍。")
    print()
    print("=" * 62)
    print("回到 dataset.py：你要写的三行长得一样，只有两处不同 ——")
    print("  ① 第一个参数用哪个 dataset（train_ds / val_ds / test_ds）")
    print("  ② shuffle 填什么")
    print("=" * 62)
