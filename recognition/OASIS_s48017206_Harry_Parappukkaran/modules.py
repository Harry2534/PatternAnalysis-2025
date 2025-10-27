import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv3d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down3d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv3d(in_ch, out_ch),
        )

    def forward(self, x):
        return self.net(x)


class Up3d(nn.Module):
    def __init__(self, in_ch, out_ch, trilinear: bool = True):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True) if trilinear \
                  else nn.ConvTranspose3d(in_ch // 2, in_ch // 2, 2, stride=2)
        self.conv = DoubleConv3d(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # pad to match D,H,W
        diffD = x2.size(2) - x1.size(2)
        diffH = x2.size(3) - x1.size(3)
        diffW = x2.size(4) - x1.size(4)
        x1 = F.pad(x1, [diffW // 2, diffW - diffW // 2,
                        diffH // 2, diffH - diffH // 2,
                        diffD // 2, diffD - diffD // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv3d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, 1)

    def forward(self, x):
        return self.conv(x)


class ImprovedUNet3D(nn.Module):
    def __init__(self, n_channels=1, n_classes=4, base_ch=16, trilinear=True):
        super().__init__()
        # Encoder
        self.inc   = DoubleConv3d(n_channels, base_ch)       # 16
        self.down1 = Down3d(base_ch, base_ch * 2)            # 32
        self.down2 = Down3d(base_ch * 2, base_ch * 4)        # 64
        self.down3 = Down3d(base_ch * 4, base_ch * 8)        # 128
        self.down4 = Down3d(base_ch * 8, base_ch * 16)       # 256
        # Decoder
        self.up1   = Up3d(base_ch * 16 + base_ch * 8, base_ch * 8, trilinear)   # 256+128 -> 128
        self.up2   = Up3d(base_ch * 8 + base_ch * 4, base_ch * 4, trilinear)    # 128+64  -> 64
        self.up3   = Up3d(base_ch * 4 + base_ch * 2, base_ch * 2, trilinear)    # 64+32   -> 32
        self.up4   = Up3d(base_ch * 2 + base_ch, base_ch, trilinear)            # 32+16   -> 16
        self.outc  = OutConv3d(base_ch, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x  = self.up1(x5, x4)
        x  = self.up2(x,  x3)
        x  = self.up3(x,  x2)
        x  = self.up4(x,  x1)
        return self.outc(x)   # [B, C, D, H, W]
