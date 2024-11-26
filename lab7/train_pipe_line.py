import torch
from torch.optim import Adam
from loss import ModelLoss
from tqdm import tqdm

def train_autoencoder_or_vae(model, dataloader, model_type, epochs=10, lr=1e-3, beta=1.0, device='cuda', save_path="./model_checkpoint"):
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = ModelLoss(model_type=model_type, beta=beta)

    for epoch in tqdm(range(epochs)):
        model.train()
        total_loss = 0

        for batch in dataloader:
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

        avg_loss = total_loss / len(dataloader)
        tqdm.write(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")

        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"{save_path}_{model_type}_epoch_{epoch + 1}.pth")
