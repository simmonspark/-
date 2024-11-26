import torch


def test_model(model, dataloader, model_type, latent_dim=None, device='cuda'):
    """
    모델 유형에 따라 테스트 실행.

    Args:
        model: 테스트할 모델 (Autoencoder, VAE, Generator, 또는 Discriminator).
        dataloader: 데이터로더.
        model_type: 'autoencoder', 'vae', 또는 'gan' 중 하나.
        latent_dim: GAN의 Generator에서 사용되는 잠재 공간 크기 (GAN일 경우 필수).
        device: 테스트에 사용할 장치 ('cuda' 또는 'cpu').

    Returns:
        None. 테스트 결과를 출력.
    """
    model = model.to(device)
    model.eval()  # 평가 모드로 전환

    with torch.no_grad():  # 테스트 시 그래디언트 계산 비활성화
        if model_type in ['autoencoder', 'vae']:
            total_loss = 0
            for i, imgs in enumerate(dataloader):
                imgs = imgs.to(device)

                if model_type == 'autoencoder':
                    recon_imgs = model(imgs)  # Autoencoder는 재구성 이미지만 반환
                    print(f"[{i}] Original Image Shape: {imgs.shape}, Reconstructed Shape: {recon_imgs.shape}")

                elif model_type == 'vae':
                    recon_imgs, mu, logvar = model(imgs)  # VAE는 재구성 이미지와 잠재 변수 반환
                    print(f"[{i}] Original Shape: {imgs.shape}, Reconstructed Shape: {recon_imgs.shape}")
                    print(f"[{i}] Latent Mean Shape: {mu.shape}, Log Variance Shape: {logvar.shape}")

        elif model_type == 'gan':
            if latent_dim is None:
                raise ValueError("latent_dim must be provided for GAN testing.")

            # Generator 테스트
            z = torch.randn(1, latent_dim, device=device)  # 샘플 잠재 벡터 생성
            generated_img = model(z)  # Generator에서 이미지 생성
            print(f"Generated Image Shape: {generated_img.shape}")

        else:
            raise ValueError(f"Invalid model_type: {model_type}. Choose from 'autoencoder', 'vae', or 'gan'.")
