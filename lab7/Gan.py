import torch
import torch.nn as nn

import torch
import torch.nn as nn




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
 # Channel Attention

        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 32 -> 64
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )


        self.conv3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 64 -> 128
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )


        self.conv4 = nn.Sequential(
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),  # 128 -> 256
            nn.Tanh()  # 출력 값을 [-1, 1] 범위로 제한
        )

    def forward(self, z):
        out = self.fc(z)
        out = out.view(out.size(0), 512, self.init_size, self.init_size)  # reshape

        out = self.conv1(out)  # ConvTranspose2d + BatchNorm + ReLU


        out = self.conv2(out)  # ConvTranspose2d + BatchNorm + ReLU


        out = self.conv3(out)  # ConvTranspose2d + BatchNorm + ReLU


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
