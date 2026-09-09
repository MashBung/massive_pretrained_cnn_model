import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

        if stride != 1 or in_channels != out_channels:
            self.pipe = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.pipe = nn.Identity()

    def forward(self, x):
        pipe_x = self.pipe(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(pipe_x + out)


class cnn(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # 320 > 160
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(),
        )  # (256,32,160,160) dtype=float16

        # 160
        self.stage1 = nn.Sequential(
            ResidualBlock(32, 32, stride=1),
            ResidualBlock(32, 32, stride=1),
        )  # (256,32,160,160)

        # 160 > 80
        self.stage2 = nn.Sequential(
            ResidualBlock(32, 64, stride=2),
            ResidualBlock(64, 64, stride=1),
        )  # (256,64,80,80)

        # 80 > 40
        self.stage3 = nn.Sequential(
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128, stride=1),
        )  # (256,128,40,40)

        # 40 > 20
        self.stage4 = nn.Sequential(
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 256, stride=1),
        )  # (256,256,20,20)

        # 20 > 10
        self.stage5 = nn.Sequential(
            ResidualBlock(256, 512, stride=2),
            ResidualBlock(512, 512, stride=1),
        )  # (256,512,10,10)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # (256,512,1,1)
            nn.Flatten(),  # (256,512)
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes),  # (256,500)
        )

    def forward(self, x):
        x = self.stem(x)
        for stage in [
            self.stage1,
            self.stage2,
            self.stage3,
            self.stage4,
            self.stage5,
            self.classifier,
        ]:
            x = stage(x)
        return x  # (256,500)
