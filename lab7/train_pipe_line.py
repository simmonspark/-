import torch
from torch.optim import Adam

from lab7.AutoEncoder import Autoencoder
from loss import ModelLoss
from tqdm import tqdm
from VAE import VAE
from utils import *
from dataloader import Dataset
import matplotlib.pyplot as plt
import torch
from torch.optim import Adam
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch

# Random seed 설정 제거
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

def train_autoencoder_or_vae(
    model, train_loader, test_loader, model_type, epochs=10, lr=1e-3,
    beta=1.0, device='cuda', save_path="./model_checkpoint", visualize=False
):
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = ModelLoss(model_type=model_type, beta=beta)

    best_val_loss = float('inf')
    early_stop_counter = 0

    for epoch in range(epochs):
        # Training phase
        model.train()
        total_loss = 0

        for batch in tqdm(iter(train_loader), desc=f"Epoch {epoch + 1}/{epochs} - Training"):
            imgs = batch.to(device)  # Move batch to device
            optimizer.zero_grad()

            # Forward pass
            if model_type == "autoencoder":
                recon_imgs = model(imgs)
                loss = loss_fn(recon_imgs, imgs)
            elif model_type == "vae":
                recon_imgs, mu, logvar = model(imgs)
                loss = loss_fn(recon_imgs, imgs, mu, logvar)
            else:
                raise ValueError("model_type must be 'autoencoder' or 'vae'")

            # Backpropagation
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Validation phase
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in tqdm(iter(test_loader), desc=f"Epoch {epoch + 1}/{epochs} - Validation"):
                imgs = batch.to(device)

                if model_type == "autoencoder":
                    recon_imgs = model(imgs)
                    loss = loss_fn(recon_imgs, imgs)
                elif model_type == "vae":
                    recon_imgs, mu, logvar = model(imgs)
                    loss = loss_fn(recon_imgs, imgs, mu, logvar)

                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(test_loader)

        # Logging progress
        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

        # Best Model Save
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = f"{save_path}_best_{model_type}.pth"
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved at {best_model_path} with Validation Loss: {best_val_loss:.4f}")
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        # Early Stopping Check
        if early_stop_counter >= 2:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break


if __name__ == '__main__':
    train, test = get_img_path()
    train_loader = Dataset('./train.json')
    test_loader = Dataset('./test.json')
    vae = VAE()
    autoencoder = Autoencoder()
    train_autoencoder_or_vae(
        model=vae,
        train_loader=train_loader,
        test_loader=test_loader,
        visualize=True,
        model_type="vae",
        epochs=10000,
        lr=1e-4,
        device='cuda'
    )
