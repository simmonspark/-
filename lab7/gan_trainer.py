import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from loss import ModelLoss
from Gan import Generator, Discriminator
from utils import *
from dataloader import Dataset
import matplotlib.pyplot as plt
import numpy as np

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

def visualize_images(generator, test_loader, latent_dim, device):
    """Validation 과정에서 생성된 이미지를 시각화"""
    generator.eval()
    z = torch.randn(4, latent_dim, device=device)  # 4개의 이미지를 생성
    with torch.no_grad():
        fake_imgs = generator(z).cpu().numpy()

    # Normalize to [0, 255]
    fake_imgs = (fake_imgs * 255).clip(0, 255).astype(np.uint8)
    fake_imgs = fake_imgs.transpose(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]

    plt.figure(figsize=(8, 4))
    for i in range(4):
        plt.subplot(1, 4, i + 1)
        plt.imshow(fake_imgs[i])
        plt.axis("off")
    plt.tight_layout()
    plt.show()


def validate_gan(generator, discriminator, test_loader, latent_dim, device):
    """Validation 단계에서 생성 이미지와 손실 계산"""
    generator.eval()
    discriminator.eval()
    total_d_loss = 0
    total_g_loss = 0
    loss_fn = ModelLoss(model_type="gan")

    with torch.no_grad():
        for real_imgs in tqdm(test_loader):
            real_imgs = real_imgs.to(device)
            batch_size = real_imgs.size(0)

            real_labels = torch.ones((batch_size, 1), device=device)
            fake_labels = torch.zeros((batch_size, 1), device=device)

            # Discriminator validation
            real_preds = discriminator(real_imgs)
            z = torch.randn(batch_size, latent_dim, device=device)
            fake_imgs = generator(z)
            fake_preds = discriminator(fake_imgs)

            d_loss, g_loss = loss_fn(real_preds, fake_preds, real_labels, fake_labels)

            total_d_loss += d_loss.item()
            total_g_loss += g_loss.item()

    avg_d_loss = total_d_loss / len(test_loader)
    avg_g_loss = total_g_loss / len(test_loader)
    return avg_d_loss, avg_g_loss


def train_gan(generator, discriminator, dataloader, test_loader, latent_dim=512, epochs=100, lr=1e-4, device='cuda', save_path="./model_checkpoint"):
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

        for real_imgs in tqdm(dataloader):
            real_imgs = real_imgs.to(device)
            batch_size = real_imgs.size(0)

            real_labels = torch.ones((batch_size, 1), device=device)
            fake_labels = torch.zeros((batch_size, 1), device=device)

            optimizer_D.zero_grad()
            real_preds = discriminator(real_imgs)
            z = torch.randn(batch_size, latent_dim, device=device)
            fake_imgs = generator(z)
            fake_preds = discriminator(fake_imgs.detach())

            d_loss, _ = loss_fn(real_preds, fake_preds, real_labels, fake_labels)
            d_loss.backward()
            optimizer_D.step()

            optimizer_G.zero_grad()
            fake_preds = discriminator(fake_imgs)
            _, g_loss = loss_fn(real_preds, fake_preds, real_labels, fake_labels)
            g_loss.backward()
            optimizer_G.step()

            total_d_loss += d_loss.item()
            total_g_loss += g_loss.item()

        avg_d_loss = total_d_loss / len(dataloader)
        avg_g_loss = total_g_loss / len(dataloader)
        tqdm.write(f"Epoch [{epoch + 1}/{epochs}], D Loss: {avg_d_loss:.4f}, G Loss: {avg_g_loss:.4f}")

        if (epoch + 1) % 5 == 0:
            # Validation 단계
            val_d_loss, val_g_loss = validate_gan(generator, discriminator, test_loader, latent_dim, device)
            tqdm.write(f"Validation - Epoch [{epoch + 1}/{epochs}], D Loss: {val_d_loss:.4f}, G Loss: {val_g_loss:.4f}")

            # 생성 이미지 시각화
            visualize_images(generator, test_loader, latent_dim, device)

            # 모델 저장
            torch.save(generator.state_dict(), f"{save_path}_generator_epoch.pth")
            torch.save(discriminator.state_dict(), f"{save_path}_discriminator_epoch.pth")


if __name__ == '__main__':
    train, test = get_img_path()
    train_loader = Dataset('./train.json')
    test_loader = Dataset('./test.json')
    train_gan(generator=Generator(latent_dim=512), discriminator=Discriminator(), dataloader=train_loader, test_loader=test_loader, latent_dim=512, epochs=100, lr=1e-4)
