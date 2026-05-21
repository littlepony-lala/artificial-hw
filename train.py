"""
==============================================================================
 pix2pix 训练脚本
==============================================================================
 实现完整的 pix2pix 训练流程，包括：
   1. 模型构建（U-Net Generator + PatchGAN Discriminator）
   2. 训练循环（D 和 G 交替更新）
   3. 训练稳定性增强（标签平滑、实例噪声）
   4. 检查点保存/恢复（支持断点续训）
   5. 验证样本生成（固定验证集对比图）
   6. TensorBoard 日志记录
"""

import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image

from config import (
    DEVICE, LEARNING_RATE, BETA1, BETA2, L1_LAMBDA,
    NUM_EPOCHS, SAVE_INTERVAL, LOG_INTERVAL, VAL_INTERVAL,
    CHECKPOINT_DIR, OUTPUT_DIR,
)
from dataset import make_dataloaders
from models import UNetGenerator, PatchGANDiscriminator


# ============================================================================
# 辅助函数
# ============================================================================

def _init_weights(m):
    """权重初始化
    """
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def save_checkpoint(epoch, G, D, g_opt, d_opt):
    """保存训练检查点。

    保存内容包括：
      - epoch:        当前 epoch 编号（0-indexed），恢复时从此轮 +1 继续
      - G_state:      Generator 的所有权重（19 层）
      - D_state:      Discriminator 的所有权重（7 层）
      - G_opt / D_opt: Adam 优化器的完整状态
                       包括 exp_avg（一阶动量）、exp_avg_sq（二阶动量）、
                       param_groups 等，确保恢复训练后优化器行为无缝衔接

    文件命名: epoch_009.pt 表示完成第 10 个 epoch（1-indexed）后保存。
    """
    checkpoint = {
        "epoch": epoch,
        "G_state": G.state_dict(),
        "D_state": D.state_dict(),
        "G_opt": g_opt.state_dict(),
        "D_opt": d_opt.state_dict(),
    }
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    torch.save(checkpoint, CHECKPOINT_DIR / f"epoch_{epoch:03d}.pt")


def load_checkpoint(path, G, D, g_opt, d_opt):
    """从文件恢复训练状态
    """
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    G.load_state_dict(ckpt["G_state"])
    D.load_state_dict(ckpt["D_state"])
    g_opt.load_state_dict(ckpt["G_opt"])
    d_opt.load_state_dict(ckpt["D_opt"])
    return ckpt["epoch"]


@torch.no_grad()
def generate_samples(G, val_loader, epoch, writer):
    """生成验证对比图并保存。
    """
    G.eval()  # 切换到评估模式
    OUTPUT_DIR.mkdir(exist_ok=True)

    edges, shoes = next(iter(val_loader))
    edges = edges.to(DEVICE)
    shoes = shoes.to(DEVICE)

    fakes = G(edges)  # 生成图像，此时不追踪梯度

    # ———— 反归一化：[-1, 1] → [0, 1] ————
    edges = (edges + 1) / 2
    shoes = (shoes + 1) / 2
    fakes = (fakes + 1) / 2

    # ———— 构建对比网格：每行 = [边缘 | 生成 | 真实] ————
    n = edges.size(0)
    rows = []
    for i in range(n):
        rows.extend([edges[i], fakes[i], shoes[i]])
    grid = torch.stack(rows)  # shape: (3*N, C, H, W)

    save_image(grid, OUTPUT_DIR / f"epoch_{epoch:03d}.png", nrow=3)

    if writer:
        writer.add_images("val/samples", grid, epoch)

    G.train()  # 切换回训练模式


# ============================================================================
# 主训练流程
# ============================================================================

def train():
    """pix2pix 主训练函数。
    """
    # —————————————— 准备工作 ——————————————
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_loader, val_loader = make_dataloaders()
    writer = SummaryWriter(log_dir="runs")  # TensorBoard 日志写入 runs/ 目录

    print(f"Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}")
    print(f"Device: {DEVICE}")

    # —————————————— 模型 & 优化器 & 损失函数 ——————————————
    G = UNetGenerator().to(DEVICE)
    D = PatchGANDiscriminator().to(DEVICE)
    G.apply(_init_weights)  # 递归初始化所有权重
    D.apply(_init_weights)

    g_opt = optim.Adam(G.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))
    d_opt = optim.Adam(D.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))

    # BCEWithLogitsLoss = Sigmoid + BCE，合并为数值稳定的单一操作
    gan_loss = nn.BCEWithLogitsLoss()
    # L1Loss = 逐像素绝对值误差 |pred - target|，产生比 L2 更锐利的图像
    l1_loss = nn.L1Loss()

    # —————————————— 断点续训 ——————————————
    start_epoch = 0
    ckpt_files = sorted(CHECKPOINT_DIR.glob("epoch_*.pt"))
    if ckpt_files:
        start_epoch = load_checkpoint(ckpt_files[-1], G, D, g_opt, d_opt) + 1
        print(f"Resumed from {ckpt_files[-1]}, starting epoch {start_epoch + 1}")

    # 全局步数计数器：用于 TensorBoard x 轴和实例噪声衰减
    global_step = start_epoch * len(train_loader)

    # —————————————— 训练循环 ——————————————
    for epoch in range(start_epoch, NUM_EPOCHS):
        g_epoch_loss = 0.0  # 累积当前 epoch 的 G 总损失
        d_epoch_loss = 0.0  # 累积当前 epoch 的 D 总损失

        for i, (edges, shoes) in enumerate(train_loader):
            edges = edges.to(DEVICE)
            shoes = shoes.to(DEVICE)
            b = edges.size(0)  # batch_size

            # ========== 训练稳定性增强 ==========

            # 单侧标签平滑（one-sided label smoothing）：
            #   真实标签用 0.9 而非 1.0，防止 D 过度自信进入 sigmoid 饱和区；
            #   虚假标签保持 0.0（实践证明单侧平滑已足够）
            real = torch.full((b, 1, 30, 30), 0.9, device=DEVICE)
            fake = torch.zeros(b, 1, 30, 30, device=DEVICE)

            # 实例噪声：对 D 的输入加入递减高斯噪声
            #   初始幅度 0.05，在 10000 步内线性衰减至 0。
            #   噪声迫使 D 学习更鲁棒的特征，防止其记住特定样本的像素模式。
            noise_scale = 0.05 * (max(0, 1 - global_step / 10000))

            # ★ 关键 ★ 生成 fakes 必须在 torch.no_grad() 外部！
            #   fakes 需要携带 G 的计算图，后续 G 优化时梯度才能从 D 回传到 G。
            #   如果放在 no_grad 内，fakes 没有 grad_fn，G 收不到任何梯度。
            fakes = G(edges)

            # ========== 第一步：训练判别器 D ==========
            # 目标：提高 D 区分真假图片对的能力
            D.zero_grad()

            # 真实样本对：edge + real shoe → 期望 D 输出 0.9（接近"真"）
            pred_real = D(edges, shoes + torch.randn_like(shoes) * noise_scale)
            d_real_loss = gan_loss(pred_real, real)

            # 虚假样本对：edge + generated shoe → 期望 D 输出 0.0（"假"）
            # ★ .detach() 切断 fakes 与 G 的梯度连接 ★
            #   训练 D 时 G 的参数不能动，detach 后的张量数据相同但无 grad_fn
            pred_fake = D(edges, fakes.detach() + torch.randn_like(fakes) * noise_scale)
            d_fake_loss = gan_loss(pred_fake, fake)

            d_loss = (d_real_loss + d_fake_loss) * 0.5  # 取平均
            d_loss.backward()
            d_opt.step()  # 只更新 D 的参数（D_opt 只注册了 D.parameters()）

            # ========== 第二步：训练生成器 G ==========
            # 目标：(1) 骗过 D，使 D 认为生成的图片是为真
            #       (2) 像素级逼近真实图片（L1 损失）
            G.zero_grad()

            # GAN 对抗损失：将同一批 fakes（不带 detach！）喂给 D
            # 梯度路径：g_loss → D(fakes) → fakes → G
            pred_fake = D(edges, fakes)
            g_gan_loss = gan_loss(pred_fake, torch.ones_like(real))
            # 注意：G 的目标是让 D 输出 1.0（硬标签），与 D 的软标签 0.9 不同。
            # 因为 G 不知道标签被平滑了，它只管全力生成"看起来绝对真"的图。

            # L1 像素重建损失：|generated - real| × λ
            # 负责保证生成图像的全局结构和内容与输入边缘图对应，
            # 防止 G 仅仅通过改变纹理就骗过 D（模式坍塌）
            g_l1 = l1_loss(fakes, shoes) * L1_LAMBDA

            g_loss = g_gan_loss + g_l1
            g_loss.backward()
            g_opt.step()  # 只更新 G 的参数

            # ———— 累积 loss 和步数 ————
            g_epoch_loss += g_loss.item()
            d_epoch_loss += d_loss.item()
            global_step += 1

            # ———— 控制台打印 + TensorBoard 记录 ————
            if (i + 1) % LOG_INTERVAL == 0:
                writer.add_scalar("loss/D", d_loss.item(), global_step)
                writer.add_scalar("loss/G_GAN", g_gan_loss.item(), global_step)
                writer.add_scalar("loss/G_L1", g_l1.item(), global_step)
                writer.add_scalar("loss/G_total", g_loss.item(), global_step)
                print(
                    f"Epoch {epoch+1:03d}/{NUM_EPOCHS}  "
                    f"Batch {i+1:04d}/{len(train_loader)}  "
                    f"D: {d_loss.item():.4f}  "
                    f"G: {g_loss.item():.4f}  "
                    f"(GAN: {g_gan_loss.item():.4f}  L1: {g_l1.item():.4f})"
                )

        # —————————————— Epoch 结束处理 ——————————————
        avg_g = g_epoch_loss / len(train_loader)
        avg_d = d_epoch_loss / len(train_loader)
        print(f"=== Epoch {epoch+1:03d} done  |  D avg: {avg_d:.4f}  |  G avg: {avg_g:.4f} ===\n")

        # 每 VAL_INTERVAL 个 epoch 生成验证对比图
        if (epoch + 1) % VAL_INTERVAL == 0:
            generate_samples(G, val_loader, epoch + 1, writer)

        # 每 SAVE_INTERVAL 个 epoch 保存检查点
        if (epoch + 1) % SAVE_INTERVAL == 0:
            save_checkpoint(epoch, G, D, g_opt, d_opt)

    # —————————————— 训练结束 ——————————————
    save_checkpoint(NUM_EPOCHS, G, D, g_opt, d_opt)
    writer.close()
    print("训练完成。")


if __name__ == "__main__":
    train()
