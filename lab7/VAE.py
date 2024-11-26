import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self, latent_dim=128):
        super(VAE, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # 256 -> 128
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 128 -> 64
            nn.BatchNorm2d(128),
            nn.Dropout2d(0.3),  # Dropout 추가
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 64 -> 32
            nn.BatchNorm2d(256),
            nn.Dropout2d(0.3),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # 32 -> 16
            nn.BatchNorm2d(512),
            nn.Dropout2d(0.3),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(512 * 16 * 16, latent_dim)
        self.fc_logvar = nn.Linear(512 * 16 * 16, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 512 * 16 * 16)

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 16 -> 32
            nn.BatchNorm2d(256),
            nn.Dropout2d(0.3),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 32 -> 64
            nn.BatchNorm2d(128),
            nn.Dropout2d(0.3),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 64 -> 128
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.3),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),  # 128 -> 256
            nn.Sigmoid(),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        z = self.reparameterize(mu, logvar)
        z = self.fc_decode(z).view(-1, 512, 16, 16)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar


if __name__ == '__main__':
    # 모델 초기화
    vae = VAE(latent_dim=128)
    tmp = torch.randn(size=(1, 3, 256, 256))
    x_recon, mu, logvar = vae(tmp)
    print(x_recon.shape, mu.shape, logvar.shape)
