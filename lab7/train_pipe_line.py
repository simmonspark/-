import torch
from torch.optim import Adam

from lab7.AutoEncoder import Autoencoder
from loss import ModelLoss
from tqdm import tqdm
from VAE import VAE
from utils import *
from dataloader import Dataset

def train_autoencoder_or_vae(model, train_loader, test_loader, model_type, epochs=10, lr=1e-3, beta=1.0, device='cuda', save_path="./model_checkpoint"):
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = ModelLoss(model_type=model_type, beta=beta)

    for epoch in tqdm(range(epochs)):
        model.train()
        total_loss = 0

        for batch in train_loader:
            imgs = batch.to(device)
            optimizer.zero_grad()

            if model_type == "autoencoder":
                recon_imgs = model(imgs)
                loss = loss_fn(recon_imgs, imgs)
            elif model_type == "vae":
                recon_imgs, mu, logvar = model(imgs)
                loss = loss_fn(recon_imgs, imgs, mu, logvar)
            else:
                raise ValueError("model_type must be 'autoencoder' or 'vae'")

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                imgs = batch.to(device)
                if model_type == "autoencoder":
                    recon_imgs = model(imgs)
                    loss = loss_fn(recon_imgs, imgs)
                elif model_type == "vae":
                    recon_imgs, mu, logvar = model(imgs)
                    loss = loss_fn(recon_imgs, imgs, mu, logvar)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(test_loader)

        tqdm.write(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"{save_path}_{model_type}.pth")

if __name__ == '__main__':
    train, test = get_img_path()
    train_loader = Dataset('./train.json')
    test_loader = Dataset('./test.json')
    vae = VAE(latent_dim=128)
    train_autoencoder_or_vae(
        model=vae,
        train_loader=train_loader,
        test_loader=test_loader,
        model_type="vae",
        epochs=100,
        lr=1e-4,
        beta=1.5,
        device='cuda'
    )
