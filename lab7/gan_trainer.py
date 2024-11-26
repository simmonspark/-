import torch
import torch.nn as nn
import torch.optim as optim
from babel.dates import date_
from loss import ModelLoss
from Gan import Generator, Discriminator


# GAN 학습 루프
def train_gan(generator, discriminator, dataloader, latent_dim=100, epochs=100, lr=1e-4, device='cuda'):

    generator = generator.to(device)
    discriminator = discriminator.to(device)

    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    loss_fn = ModelLoss(model_type="gan")

    for epoch in range(epochs):
        generator.train()
        discriminator.train()
        total_d_loss = 0
        total_g_loss = 0

        for real_imgs in dataloader:
            real_imgs = real_imgs.to(device)
            batch_size = real_imgs.size(0)

            # 진짜와 가짜 레이블 생성
            real_labels = torch.ones((batch_size, 1), device=device)
            fake_labels = torch.zeros((batch_size, 1), device=device)

            # ----------- Discriminator 학습 -----------
            optimizer_D.zero_grad()

            # 진짜 이미지 손실
            real_preds = discriminator(real_imgs)
            z = torch.randn(batch_size, latent_dim, device=device)
            fake_imgs = generator(z)
            fake_preds = discriminator(fake_imgs.detach())

            d_loss, _ = loss_fn(real_preds, fake_preds, real_labels, fake_labels)
            d_loss.backward()
            optimizer_D.step()

            # ----------- Generator 학습 -----------
            optimizer_G.zero_grad()

            fake_preds = discriminator(fake_imgs)
            _, g_loss = loss_fn(real_preds, fake_preds, real_labels, fake_labels)
            g_loss.backward()
            optimizer_G.step()

            total_d_loss += d_loss.item()
            total_g_loss += g_loss.item()

        avg_d_loss = total_d_loss / len(dataloader)
        avg_g_loss = total_g_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{epochs}], D Loss: {avg_d_loss:.4f}, G Loss: {avg_g_loss:.4f}")
