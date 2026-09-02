import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 5,
        num_classes: int = 20,
    ):
        super().__init__()

        self.encoder1 = DoubleConv(
            in_channels,
            32,
        )

        self.pool1 = nn.MaxPool2d(2)

        self.encoder2 = DoubleConv(
            32,
            64,
        )

        self.pool2 = nn.MaxPool2d(2)

        self.encoder3 = DoubleConv(
            64,
            128,
        )

        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(
            128,
            256,
        )

        self.up3 = nn.ConvTranspose2d(
            256,
            128,
            kernel_size=2,
            stride=2,
        )

        self.decoder3 = DoubleConv(
            256,
            128,
        )

        self.up2 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=2,
            stride=2,
        )

        self.decoder2 = DoubleConv(
            128,
            64,
        )

        self.up1 = nn.ConvTranspose2d(
            64,
            32,
            kernel_size=2,
            stride=2,
        )

        self.decoder1 = DoubleConv(
            64,
            32,
        )

        self.output = nn.Conv2d(
            32,
            num_classes,
            kernel_size=1,
        )

    def forward(
            self,
            x: torch.Tensor,
    ) -> torch.Tensor:
        e1 = self.encoder1(x)

        e2 = self.encoder2(
            self.pool1(e1)
        )

        e3 = self.encoder3(
            self.pool2(e2)
        )

        b = self.bottleneck(
            self.pool3(e3)
        )

        d3 = self.up3(b)
        d3 = F.interpolate(
            d3,
            size=e3.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        d3 = torch.cat(
            [d3, e3],
            dim=1,
        )
        d3 = self.decoder3(d3)

        d2 = self.up2(d3)
        d2 = F.interpolate(
            d2,
            size=e2.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        d2 = torch.cat(
            [d2, e2],
            dim=1,
        )
        d2 = self.decoder2(d2)

        d1 = self.up1(d2)
        d1 = F.interpolate(
            d1,
            size=e1.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        d1 = torch.cat(
            [d1, e1],
            dim=1,
        )
        d1 = self.decoder1(d1)

        output = self.output(d1)

        output = F.interpolate(
            output,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        return output