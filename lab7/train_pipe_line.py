import torch
from torch.optim import Adam


# Autoencoder와 VAE 통합 트레이닝 루프
def train_autoencoder_or_vae(model, dataloader, model_type, epochs=10, lr=1e-3, beta=1.0, device='cuda'):
    """
    Autoencoder 또는 VAE 학습 루프.

    Args:
        model: Autoencoder 또는 VAE 모델.
        dataloader: 데이터로더.
        model_type: 'autoencoder' 또는 'vae'.
        epochs: 학습 에포크 수.
        lr: 학습률.
        beta: VAE의 KL 다이버전스 가중치 (VAE에서만 사용).
        device: 'cuda' 또는 'cpu'.
    """
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = ModelLoss(model_type=model_type, beta=beta)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in dataloader:
            imgs = batch.to(device)  # 데이터 로드
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
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")
