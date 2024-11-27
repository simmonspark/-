import torch
import matplotlib.pyplot as plt
from VAE import VAE
from Gan import Generator as GANGenerator
from AutoEncoder import Autoencoder
import numpy as np
import json
from dataloader import Dataset


# 이미지 생성 및 시각화
def test_models(autoencoder, vae, gan_generator, dataset, device='cuda'):
    autoencoder.to(device)
    vae.to(device)
    gan_generator.to(device)

    # 테스트용 latent 생성
    autoencoder_latent = torch.randn(1, 512, 16, 16).to(device)  # Autoencoder latent space
    vae_latent = torch.randn(1, 512).to(device)  # VAE latent space
    gan_latent = torch.randn(1, 512).to(device)  # GAN latent space

    # Autoencoder로 생성
    autoencoder.eval()
    with torch.no_grad():
        ae_generated = autoencoder.decoder(autoencoder_latent)

    # VAE로 생성
    vae.eval()
    with torch.no_grad():
        tmp = vae.fc_decode(vae_latent)
        tmp = tmp.view(-1, 512, 16, 16)
        vae_generated = vae.decoder(tmp)

    # GAN Generator로 생성
    gan_generator.eval()
    with torch.no_grad():
        gan_generated = gan_generator(gan_latent)

    # Dataset에서 첫 번째 이미지 가져오기
    original_image = dataset[0][0].cpu().numpy().transpose(1, 2, 0)  # [C, H, W] -> [H, W, C]

    # 생성된 이미지를 시각화
    generated_images = {
        "Original": original_image,
        "Autoencoder": ae_generated.cpu().numpy().squeeze().transpose(1, 2, 0),
        "VAE": vae_generated.cpu().numpy().squeeze().transpose(1, 2, 0),
        "GAN": gan_generated.cpu().numpy().squeeze().transpose(1, 2, 0)
    }

    plt.figure(figsize=(12, 8))
    for i, (title, img) in enumerate(generated_images.items(), start=1):
        plt.subplot(2, 2, i)
        plt.imshow((img * 255).clip(0, 255).astype(np.uint8))  # [0, 1] -> [0, 255]
        plt.title(title)
        plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Dataset 초기화
    dataset = Dataset('./test.json')

    # 모델 초기화
    autoencoder = Autoencoder()
    vae = VAE(latent_dim=512)
    gan_generator = GANGenerator(latent_dim=512)
    # 모델 체크포인트 로드
    autoencoder.load_state_dict(torch.load('model_checkpoint_best_autoencoder.pth'))
    vae.load_state_dict(torch.load('model_checkpoint_best_vae.pth'))
    gan_generator.load_state_dict(torch.load('model_checkpoint_generator.pth'))

    # 테스트 실행
    test_models(autoencoder, vae, gan_generator, dataset, device='cuda' if torch.cuda.is_available() else 'cpu')
