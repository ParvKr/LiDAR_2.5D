import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout_rate: float = 0.0,
    ):
        super().__init__()

        layers = [
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
        ]

        if dropout_rate > 0.0:
            layers.append(nn.Dropout2d(p=dropout_rate))

        self.block = nn.Sequential(*layers)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.block(x)


class ASPPModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout_rate: float = 0.0):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            # Use GroupNorm instead of BatchNorm2d because batch size 1 will crash BatchNorm on 1x1 spatial dims
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True)
        )
        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate) if dropout_rate > 0 else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = []
        res.append(self.conv1(x))
        res.append(self.conv2(x))
        res.append(self.conv3(x))
        res.append(self.conv4(x))
        pool = self.image_pool(x)
        pool = F.interpolate(pool, size=x.shape[2:], mode='bilinear', align_corners=False)
        res.append(pool)
        out = torch.cat(res, dim=1)
        return self.project(out)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 6,
        num_classes: int = 20,
        dropout_rate: float = 0.2,
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

        self.bottleneck = ASPPModule(
            128,
            256,
            dropout_rate=dropout_rate,
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
            dropout_rate=dropout_rate,
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
            dropout_rate=dropout_rate,
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
            dropout_rate=dropout_rate,
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