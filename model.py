from typing import Any

import torch
import torch.nn as nn
from torch import Tensor
from torchvision import models
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        ## Encoder

        # 572x572 -> 568x568
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
            # to stabilize and accelerate the training process by normalizing the inputs of each layer
            # Batch normalization normalizes the input to each layer by subtracting the batch mean and dividing by the batch standard deviation.
            nn.BatchNorm2d(64)
        )
        # 570x570 -> 568x568
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )
        # 568x568 -> 284x284
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # 284x284 -> 282x282
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128)
        )
        # 282x282 -> 280x280
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128)
        )
        # 280x280 -> 140x140
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # 140x140 -> 138x138
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256)
        )
        # 138x138 -> 136x136
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256)
        )
        # 136x136 -> 68x68
        self.max_pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # 68x68 -> 66x66
        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512)
        )
        # 66x66 -> 64x64
        self.conv8 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512)
        )
        # 64x64 -> 32x32
        self.max_pool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)


        ## Bottleneck

        # 32x32 -> 30x30
        self.conv9 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024)
        )
        # 30x30 -> 28x28
        self.conv10 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024)
        )


        ## Decoder

        # 28x28 -> 56x56
        self.upconv1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2, padding=0)
        # 56x56 -> 54x54
        self.conv11 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512)
        )
        # 54x54 -> 52x52
        self.conv12 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512)
        )

        # 52x52 -> 104x104
        self.upconv2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2, padding=0)
        # 104x104 -> 102x102
        self.conv13 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256)
        )
        # 102x102 -> 100x100
        self.conv14 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256)
        )

        # 100x100 -> 200x200
        self.upconv3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2, padding=0)
        # 200x200 -> 198x198
        self.conv15 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128)
        )
        # 198x198 -> 196x196
        self.conv16 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128)
        )

        # 196x196 -> 392x392
        self.upconv4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2, padding=0)
        # 392x392 -> 390x390
        self.conv17 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )
        # 390x390 -> 388x388
        self.conv18 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )

        ## Output
        # 388x388 -> 388x388
        self.conv19 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0)

    def forward(self, x: Any) -> Tensor:
        ## Encoder

        x = F.relu(self.conv1(x))
        x1 = F.relu(self.conv2(x))
        x = self.max_pool1(x1)

        x = F.relu(self.conv3(x))
        x2 = F.relu(self.conv4(x))
        x = self.max_pool2(x2)

        x = F.relu(self.conv5(x))
        x3 = F.relu(self.conv6(x))
        x = self.max_pool3(x3)

        x = F.relu(self.conv7(x))
        x4 = F.relu(self.conv8(x))
        x = self.max_pool4(x4)


        ## Bottleneck

        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))


        ## Decoder

        x11 = self.upconv1(x)
        # skip connection
        # print(f"x11 shape: {x11.shape}")
        # print(f"x4 shape: {x4.shape}")
        x = torch.cat([x11, x4], dim=1)
        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))

        x22 = self.upconv2(x)
        x = torch.cat([x22, x3], dim=1)
        x = F.relu(self.conv13(x))
        x = F.relu(self.conv14(x))

        x33 = self.upconv3(x)
        x = torch.cat([x33, x2], dim=1)
        x = F.relu(self.conv15(x))
        x = F.relu(self.conv16(x))

        x44 = self.upconv4(x)
        x = torch.cat([x44, x1], dim=1)
        x = F.relu(self.conv17(x))
        x = F.relu(self.conv18(x))

        ## Output
        x = self.conv19(x)

        return x

