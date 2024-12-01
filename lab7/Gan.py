import torch
import torch.nn as nn

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 채널 평균 풀링
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()  # (batch_size, channels, height, width)
        y = self.avg_pool(x).view(b, c)  # 채널 압축
        y = self.fc(y).view(b, c, 1, 1)  # FC 레이어를 통해 중요도 계산
        return x * y  # 원래 feature map에 중요도 반영


class Generator(nn.Module):
    def __init__(self, latent_dim=512):
        super(Generator, self).__init__()
        self.init_size = 16  # 초기 이미지 크기: 16x16
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512 * self.init_size * self.init_size)  # Latent vector를 초기 이미지로 매핑
        )

        # ConvTranspose2d 블록마다 Channel Attention 추가
        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 16 -> 32
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.ca1 = ChannelAttention(256)  # Channel Attention

        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 32 -> 64
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.ca2 = ChannelAttention(128)  # Channel Attention

        self.conv3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 64 -> 128
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.ca3 = ChannelAttention(64)  # Channel Attention

        self.conv4 = nn.Sequential(
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),  # 128 -> 256
            nn.Tanh()  # 출력 값을 [-1, 1] 범위로 제한
        )

    def forward(self, z):
        out = self.fc(z)
        out = out.view(out.size(0), 512, self.init_size, self.init_size)  # reshape

        out = self.conv1(out)  # ConvTranspose2d + BatchNorm + ReLU
        out = self.ca1(out)  # Channel Attention

        out = self.conv2(out)  # ConvTranspose2d + BatchNorm + ReLU
        out = self.ca2(out)  # Channel Attention

        out = self.conv3(out)  # ConvTranspose2d + BatchNorm + ReLU
        out = self.ca3(out)  # Channel Attention

        img = self.conv4(out)  # ConvTranspose2d + Tanh
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

            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0),  # 16 -> 13
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(13 * 13, 1),
            nn.Sigmoid()  # 출력 범위 [0, 1]
        )

    def forward(self, img):
        features = self.model(img)
        validity = self.fc(features)
        return validity


if __name__ == '__main__':
    # Generator와 Discriminator 초기화
    generator = Generator(latent_dim=512)
    discriminator = Discriminator()

    # Generator 테스트
    latent_z = torch.randn(1, 512)  # Latent vector 크기 512
    generated_img = generator(latent_z)
    print(f"Generated Image Shape: {generated_img.shape}")  # 예상 출력: (1, 3, 256, 256)

    # Discriminator 테스트
    validity = discriminator(generated_img)
    print(f"Discriminator Output Shape: {validity.shape}")  # 예상 출력: (1, 1)
