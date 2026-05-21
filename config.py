"""
==============================================================================
 pix2pix 训练配置文件
==============================================================================
"""

import torch
from pathlib import Path

# ============================== 路径配置 ==============================

BASE_DIR = Path(__file__).parent                     # 项目根目录（config.py 所在目录）
DATA_DIR = BASE_DIR / "data"                         # 数据集压缩包存放目录
ZIP_PATH = DATA_DIR / "Edges2shoes Dataset.zip"      # 原始数据集压缩包路径
EXTRACT_DIR = DATA_DIR / "edges2shoes"               # 解压后的数据集目录
CHECKPOINT_DIR = BASE_DIR / "checkpoints"            # 模型检查点保存目录
OUTPUT_DIR = BASE_DIR / "outputs"                    # 验证生成图输出目录

# ============================== 数据参数 ==============================

IMAGE_SIZE = 256     # 输入/输出图像的宽高（像素）
BATCH_SIZE = 4       # 每批训练样本数。
NUM_WORKERS = 4      # DataLoader 的子进程数，负责 CPU 端的数据加载与预处理

# ============================== 优化器参数 ==============================

LEARNING_RATE = 2e-4  # Adam 初始学习率
BETA1 = 0.5           # Adam 的一阶动量衰减系数
BETA2 = 0.999         # Adam 的二阶动量衰减系数

# ============================== 损失权重 ==============================

L1_LAMBDA = 10.0      # L1 重建损失的权重 λ。
                      # 总损失: G_loss = GAN_loss + λ * L1_loss
                      # λ 越大，生成图越贴近目标但可能越模糊；
                      # λ 越小，纹理越丰富但结构约束越弱。

# ============================== 训练控制 ==============================

NUM_EPOCHS = 200      # 总训练轮数。2000 张图 × 200 轮 = 40 万次迭代
SAVE_INTERVAL = 10    # 每隔多少个 epoch 保存一次检查点（.pt 文件）
LOG_INTERVAL = 100    # 每隔多少个 batch 打印一次 loss 并写入 TensorBoard
VAL_INTERVAL = 5      # 每隔多少个 epoch 生成一次验证对比图（边缘→生成→真实）


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

