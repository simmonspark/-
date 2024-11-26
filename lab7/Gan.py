import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super(Generator, self).__init__()
        self.init_size = 16  # 초기 이미지 크기: 16x16
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512 * self.init_size * self.init_size)
        )

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 16 -> 32
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 32 -> 64
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 64 -> 128
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),  # 128 -> 256
            nn.Tanh()  # 출력 값을 [-1, 1] 범위로 제한
        )

    def forward(self, z):
        out = self.fc(z)
        out = out.view(out.size(0), 512, self.init_size, self.init_size)  # reshape
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # 256 -> 128
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 128 -> 64
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 64 -> 32
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # 32 -> 16
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0),  # 16 -> 1 (Real/Fake Score)
            nn.Flatten(),
            nn.Linear(13 * 13, 1),
            nn.Sigmoid()  # 출력 범위 [0, 1]
        )

    def forward(self, img):
        validity = self.model(img)
        return validity.view(-1, 1)  # Flatten for binary classification


if __name__ == '__main__':
    # Discriminator 생성
    discriminator = Discriminator()
    generator = Generator()
    latent_z = torch.randn(1, 100)
    tmp = generator(latent_z)
    print(tmp.shape)
    tmp = discriminator(tmp)
    print(tmp.shape)
