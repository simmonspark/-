import torch
import torch.nn as nn
import torch.nn.functional as F

import torch

# Random seed 설정 제거
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
class ModelLoss(nn.Module):
    def __init__(self, model_type, beta=1.0, latent_dim=128):
        """
        손실 계산 클래스

        Args:
            model_type (str): 'autoencoder', 'vae', 또는 'gan' 중 하나를 선택.
            beta (float): VAE에서 KL 다이버전스 가중치 (기본값: 1.0).
            latent_dim (int): VAE에서 잠재 공간 크기.
        """
        super(ModelLoss, self).__init__()
        self.model_type = model_type
        self.beta = beta
        self.latent_dim = latent_dim
        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()

    def forward(self, *inputs):
        """
        손실 계산.

        Args:
            inputs: 모델 유형에 따라 다름:
                - Autoencoder: (recon_x, x)
                - VAE: (recon_x, x, mu, logvar)
                - GAN: (real_preds, fake_preds, real_labels, fake_labels)

        Returns:
            Tensor: 계산된 손실.
        """
        if self.model_type == "autoencoder":
            return self._autoencoder_loss(*inputs)
        elif self.model_type == "vae":
            return self._vae_loss(*inputs)
        elif self.model_type == "gan":
            return self._gan_loss(*inputs)
        else:
            raise ValueError(f"Invalid model_type: {self.model_type}. Choose from 'autoencoder', 'vae', or 'gan'.")

    def _autoencoder_loss(self, recon_x, x):
        """
        Autoencoder 손실 (MSE 사용).
        Args:
            recon_x: 재구성된 이미지.
            x: 원본 이미지.
        Returns:
            Tensor: MSE 손실.
        """
        return self.mse_loss(recon_x, x)

    '''def _vae_loss(self, recon_x, x, mu, logvar):
        # MSE 손실
        #recon_loss_mae = nn.L1Loss()(recon_x,x)
        recon_loss = nn.MSELoss()(recon_x, x)
        # KL 다이버전스 손실
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_divergence /= x.size(0)

        # 총 손실
        return recon_loss + self.beta * kl_divergence'''

    def _vae_loss(self,recon_x, x, mu, logvar):
        recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kl_div

    def _gan_loss(self, real_preds, fake_preds, real_labels, fake_labels):
        """
        GAN 손실 (BCE 사용).
        Args:
            real_preds: Discriminator의 진짜 이미지에 대한 예측.
            fake_preds: Discriminator의 가짜 이미지에 대한 예측.
            real_labels: 진짜 이미지에 대한 레이블 (1).
            fake_labels: 가짜 이미지에 대한 레이블 (0).
        Returns:
            Tensor: 총 손실 (Discriminator와 Generator 손실 모두 반환).
        """
        # Discriminator 손실
        real_loss = self.bce_loss(real_preds, real_labels)
        fake_loss = self.bce_loss(fake_preds, fake_labels)
        d_loss = real_loss + fake_loss

        # Generator 손실
        g_loss = self.bce_loss(fake_preds, real_labels)  # Generator는 fake_preds를 real로 속이려 함

        return d_loss, g_loss
