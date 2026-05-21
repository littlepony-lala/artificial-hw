"""
==============================================================================
 Edges2Shoes 数据集加载模块
==============================================================================
 功能：
   1. 首次运行时自动解压数据集压缩包
   2. 将 512×256 的合成图切分为左半（边缘草图）和右半（真实鞋子）
   3. 训练集应用数据增强（随机裁剪 + 水平翻转），验证集仅做缩放归一化
   4. 像素值从 [0, 1] 线性映射到 [-1, 1]，以匹配 Generator 输出层 Tanh 的值域
"""

import zipfile
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from config import ZIP_PATH, EXTRACT_DIR, IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS


def _extract_once():
    """仅在首次调用时解压数据集，已存在则跳过。
    避免每次启动训练都重复解压。"""
    if EXTRACT_DIR.exists():
        return
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(EXTRACT_DIR)
    print(f"数据集已解压至 {EXTRACT_DIR}")


class Edges2ShoesDataset(Dataset):
    """Edges2Shoes 自定义数据集。

    每张原始图片为 512×256 像素：
      - 左半 256×256：边缘轮廓图（edge map）
      - 右半 256×256：真实鞋子照片（shoe photo）

    Args:
        split: "train" 使用数据增强；"val" 仅缩放，保持评估一致性。
    """

    def __init__(self, split: str = "train"):
        _extract_once()
        self.root = EXTRACT_DIR / split           # e.g. data/edges2shoes/train
        self.paths = sorted(self.root.glob("*.jpg"))  # 排序确保可复现
        self.split = split

        # 两个分支共用的基础变换：缩放到目标尺寸 → 转为张量
        base = [T.Resize((IMAGE_SIZE, IMAGE_SIZE)), T.ToTensor()]

        if split == "train":
            # 训练集数据增强流程（参考 pix2pix 论文）：
            #   Resize(286) → RandomCrop(256) → RandomHorizontalFlip → ToTensor
            # Resize 到 286 再随机裁回 256 增加尺度抖动，
            # 随机水平翻转提供左右镜像增强。
            self.transform = T.Compose([
                T.Resize((286, 286)),
                T.RandomCrop((IMAGE_SIZE, IMAGE_SIZE)),
                T.RandomHorizontalFlip(),
                *base,
            ])
        else:
            # 验证集不做随机增强，确保每次评估结果可比
            self.transform = T.Compose(base)

    def __len__(self):
        """返回数据集样本总数"""
        return len(self.paths)

    def __getitem__(self, idx):
        """返回第 idx 对样本： (边缘图, 真实鞋子图)，值域均为 [-1, 1]。

        关键步骤：
          1. 打开 512×256 合成图
          2. 左半 crop → edge，右半 crop → shoe
          3. 各自通过 transform（训练集有增强，验证集无）
          4. 从 [0,1] 线性映射到 [-1,1]：x * 2 - 1
        """
        img = Image.open(self.paths[idx]).convert("RGB")
        w = img.size[0]
        half = w // 2                                 # 一半宽度 = 256
        edge = img.crop((0, 0, half, img.size[1]))    # 左半：边缘图
        shoe = img.crop((half, 0, w, img.size[1]))    # 右半：真实鞋子

        edge = self.transform(edge)
        shoe = self.transform(shoe)

        # 值域映射：[0, 1] → [-1, 1]
        # 因为 Generator 输出层使用 Tanh，输出范围为 [-1, 1]，
        # 训练目标需要与之匹配
        edge = edge * 2 - 1
        shoe = shoe * 2 - 1

        return edge, shoe


def make_dataloaders():
    """创建训练集和验证集的 DataLoader。

    Returns:
        train_loader: batch_size=4, shuffle=True,  训练用（随机打乱）
        val_loader:   batch_size=4, shuffle=False, 验证用（固定顺序）
    """
    train_ds = Edges2ShoesDataset("train")
    val_ds = Edges2ShoesDataset("val")

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,        # 训练时打乱，防止模型记忆样本顺序
        num_workers=NUM_WORKERS,
        pin_memory=True,     # 锁页内存，加速 CPU→GPU 数据传输
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,       # 验证时保持固定顺序，方便跨 epoch 对比
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    return train_loader, val_loader
