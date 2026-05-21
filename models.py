"""
==============================================================================
 pix2pix 模型定义：UNetGenerator + PatchGANDiscriminator
==============================================================================
 包含三个部分：
   1. _DownBlock  — 编码器基础模块（下采样）
   2. _UpBlock    — 解码器基础模块（上采样 + 跳跃连接拼接）
   3. UNetGenerator     — U-Net 256 生成器（54.4M 参数）
   4. PatchGANDiscriminator — 70×70 PatchGAN 判别器（2.8M 参数）

"""

import torch
import torch.nn as nn


# ============================================================================
# 基础模块
# ============================================================================

class _DownBlock(nn.Module):
    """编码器下采样模块。

    结构：Conv2d(k4, s2, p1) → InstanceNorm2d(可选) → LeakyReLU(0.2)

    步长 stride=2 每次将空间分辨率减半，通道数通常翻倍。
    use_norm=False 用于第一层（输入 3 通道，不做归一化）
    和瓶颈层（1×1 特征图无法做 InstanceNorm）。

    """

    def __init__(self, in_ch, out_ch, use_norm=True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1,
                            bias=not use_norm)]
        if use_norm:
            layers.append(nn.InstanceNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class _UpBlock(nn.Module):
    """解码器上采样模块"""

    def __init__(self, in_ch, out_ch, use_dropout=False):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1,
                               bias=False),
            nn.InstanceNorm2d(out_ch),
        ]
        if use_dropout:
            layers.insert(2, nn.Dropout(0.5))
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x, skip):
        """前向传播 + 跳跃连接拼接。

        Args:
            x:     来自上一层的特征图（经过上采样）
            skip:  编码器对应层的特征图（空间尺寸相同）

        Returns:
            沿通道维拼接后的特征图，通道数 = out_ch + skip_ch
        """
        x = self.block(x)
        return torch.cat([x, skip], dim=1)  # dim=1 是通道维 (N, C, H, W)


# ============================================================================
# Generator — U-Net 256
# ============================================================================

class UNetGenerator(nn.Module):
    """基于 U-Net 的图像生成器，用于 256×256 的图像翻译。

    架构概述：
      编码器 (8层): 256→128→64→32→16→8→4→2→1   （下采样×8）
      解码器 (7层): 1→2→4→8→16→32→64→128→256     （上采样×8，含输出层）
      跳跃连接: e1→d7, e2→d6, ..., e7→d1           （共 7 条）

    编码器输出特征图尺寸逐步减半、通道数逐步翻倍（到 512 后不再增加），
    在瓶颈处达到 1×1 的极大压缩比，强制 U-Net 学习紧凑的全局表示。
    解码器通过跳跃连接获取编码器的浅层空间信息，避免上采样时的空间模糊。

    """

    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        ngf = 64  # 基础通道数（Generator first-layer filters）

        # ---------------------- 编码器 ----------------------
        # 输入 256×256 → ... → 瓶颈 1×1
        self.e1 = _DownBlock(in_channels, ngf, use_norm=False)      # 3→64   128
        self.e2 = _DownBlock(ngf, ngf * 2)                          # 64→128 64
        self.e3 = _DownBlock(ngf * 2, ngf * 4)                      # 128→256 32
        self.e4 = _DownBlock(ngf * 4, ngf * 8)                      # 256→512 16
        self.e5 = _DownBlock(ngf * 8, ngf * 8)                      # 512→512 8
        self.e6 = _DownBlock(ngf * 8, ngf * 8)                      # 512→512 4
        self.e7 = _DownBlock(ngf * 8, ngf * 8)                      # 512→512 2
        self.e8 = _DownBlock(ngf * 8, ngf * 8, use_norm=False)      # 512→512 1
        # 注意：e8 不使用归一化，因为 1×1 特征图无空间方差/均值可统计

        # ---------------------- 解码器 ----------------------
        # 瓶颈 1×1 → ... → 输出 256×256
        # 每层接收来自相应编码器层的跳跃连接，通道数标注为 "cat 后 / 输出"
        self.d1 = _UpBlock(ngf * 8, ngf * 8, use_dropout=True)      # 512→512  2
        self.d2 = _UpBlock(ngf * 16, ngf * 8, use_dropout=True)     # 1024→512 4
        self.d3 = _UpBlock(ngf * 16, ngf * 8, use_dropout=True)     # 1024→512 8
        self.d4 = _UpBlock(ngf * 16, ngf * 8)                       # 1024→512 16
        self.d5 = _UpBlock(ngf * 16, ngf * 4)                       # 1024→256 32
        self.d6 = _UpBlock(ngf * 8, ngf * 2)                        # 512→128  64
        self.d7 = _UpBlock(ngf * 4, ngf)                            # 256→64   128

        # ---------------------- 输出层 ----------------------
        # 将 d7 拼接后的 128 通道特征映射回 3 通道 RGB 图像
        self.out_conv = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, out_channels, 4, stride=2, padding=1),
            nn.Tanh(),  # 输出值域 [-1, 1]，与输入归一化范围一致
        )

    def forward(self, x):
        """前向传播：编码 → 瓶颈 → 带跳跃连接的解码。

        Args:
            x: 输入边缘图，shape (N, 3, 256, 256)，值域 [-1, 1]

        Returns:
            生成的鞋子图像，shape (N, 3, 256, 256)，值域 [-1, 1]
        """
        # ———— 编码（特征图逐步压缩，通道数增加） ————
        e1 = self.e1(x)                     # (N, 64,  128, 128)
        e2 = self.e2(e1)                    # (N, 128, 64,  64)
        e3 = self.e3(e2)                    # (N, 256, 32,  32)
        e4 = self.e4(e3)                    # (N, 512, 16,  16)
        e5 = self.e5(e4)                    # (N, 512, 8,   8)
        e6 = self.e6(e5)                    # (N, 512, 4,   4)
        e7 = self.e7(e6)                    # (N, 512, 2,   2)
        e8 = self.e8(e7)                    # (N, 512, 1,   1) ← 瓶颈

        # ———— 解码（跳跃连接拼接 → 上采样） ————
        d1 = self.d1(e8, e7)                # (N, 1024, 2,  2)
        d2 = self.d2(d1, e6)                # (N, 1024, 4,  4)
        d3 = self.d3(d2, e5)                # (N, 1024, 8,  8)
        d4 = self.d4(d3, e4)                # (N, 1024, 16, 16)
        d5 = self.d5(d4, e3)                # (N, 512,  32, 32)
        d6 = self.d6(d5, e2)                # (N, 256,  64, 64)
        d7 = self.d7(d6, e1)                # (N, 128,  128,128)

        return self.out_conv(d7)            # (N, 3,   256,256)


# ============================================================================
# Discriminator — 70×70 PatchGAN
# ============================================================================

class PatchGANDiscriminator(nn.Module):
    """PatchGAN 判别器，对局部图像块进行真/伪分类。

    输入：6 通道（3ch 边缘图 + 3ch 鞋子图），沿通道维拼接。
    输出：30×30 的预测图，每个像素代表输入中对应 70×70 区域的判别结果。

    判别器只关注局部纹理的真实性（高频信息），不判断全局结构。
    全局结构的正确性由 L1 损失来保证，二者互补。

    架构：C64 → C128 → C256 → C512 → C1
    """

    def __init__(self, in_channels=6):
        super().__init__()
        ndf = 64  # 判别器基础通道数（Discriminator first-layer filters）

        self.layers = nn.Sequential(
            # ── C64: 256→128, 无 BN（第一层直接处理原始像素） ──
            nn.Conv2d(in_channels, ndf, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # ── C128: 128→64 ──
            nn.Conv2d(ndf, ndf * 2, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # ── C256: 64→32 ──
            nn.Conv2d(ndf * 2, ndf * 4, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # ── C512: 32→31, stride=1（不继续下采样，保持感受野 70×70） ──
            nn.Conv2d(ndf * 4, ndf * 8, 4, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # ── 输出层: 31→30, 单通道（每个像素一个 logit） ──
            nn.Conv2d(ndf * 8, 1, 4, stride=1, padding=1),
        )
        # 注意：判别器输出不经过 Sigmoid，因为训练中使用 BCEWithLogitsLoss
        # 将 Sigmoid 和 BCE Loss 合并为数值更稳定的单一操作

    def forward(self, edge, img):
        """前向传播。

        Args:
            edge: 边缘图，shape (N, 3, 256, 256)
            img:  鞋子图（真实或生成的），shape (N, 3, 256, 256)

        Returns:
            预测图，shape (N, 1, 30, 30)，每个值是该 70×70 patch 为"真"的 logit
        """
        x = torch.cat([edge, img], dim=1)  # 沿通道拼接: 3+3=6 通道
        return self.layers(x)
