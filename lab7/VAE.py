import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self, latent_dim=512):  # 잠재 공간 크기
        super(VAE, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # 256 -> 128
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 128 -> 64
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 64 -> 32
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # 32 -> 16
            nn.ReLU(),
        )
        self.flatten = nn.Flatten()  # Latent space로 변환 전 flatten

        # Encoder 출력 크기 확인 (512채널, 16x16 -> 131072)
        encoder_output_dim = 512 * 16 * 16
        self.fc_mu = nn.Linear(encoder_output_dim, latent_dim)  # 평균 (μ)
        self.fc_logvar = nn.Linear(encoder_output_dim, latent_dim)  # 로그 분산 (log(σ^2))

        # Decoder
        self.fc_decode = nn.Linear(latent_dim, encoder_output_dim)  # 잠재 공간을 원래 크기로 복원
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 16 -> 32
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 32 -> 64
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 64 -> 128
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),  # 128 -> 256
            nn.Sigmoid(),  # 픽셀 값을 [0,1]로 제한
        )

    def reparameterize(self, mu, logvar):
        """
        Reparameterization Trick
        z = μ + ε * σ, where ε ~ N(0, I)
        """
        std = torch.exp(0.5 * logvar)  # σ = exp(0.5 * log(σ^2))
        eps = torch.randn_like(std)  # ε ~ N(0, I)
        return mu + eps * std

    def forward(self, x):
        # Encoder
        x = self.encoder(x)
        x = self.flatten(x)  # Flatten to [batch_size, encoder_output_dim]
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        # Latent space sampling
        z = self.reparameterize(mu, logvar)

        # Decoder
        x = self.fc_decode(z)
        x = x.view(-1, 512, 16, 16)  # Reshape to match decoder input
        x = self.decoder(x)
        return x, mu, logvar


if __name__ == '__main__':
    vae = VAE(latent_dim=512)  # 잠재 공간 크기 설정
    tmp = torch.randn(size=(1, 3, 256, 256))  # 입력 샘플
    output, mu, logvar = vae(tmp)
    print("Output shape:", output.shape)  # 기대 출력: (1, 3, 256, 256)
    print("Mu shape:", mu.shape)  # 기대 출력: (1, 512)
    print("Logvar shape:", logvar.shape)  # 기대 출력: (1, 512)
