import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm3d(out_channels, affine=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm3d(out_channels, affine=True)
        self.act = nn.LeakyReLU(0.01, inplace=True)
        if in_channels == out_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        identity = self.skip(x)
        out = self.act(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.act(out + identity)


class ResidualBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm2d(out_channels, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm2d(out_channels, affine=True)
        self.drop = nn.Dropout2d(float(dropout)) if float(dropout) > 0 else nn.Identity()
        self.act = nn.LeakyReLU(0.01, inplace=True)
        if in_channels == out_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        identity = self.skip(x)
        out = self.act(self.norm1(self.conv1(x)))
        out = self.drop(out)
        out = self.norm2(self.conv2(out))
        return self.act(out + identity)


class DownBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.block = ResidualBlock3D(out_channels, out_channels)

    def forward(self, x):
        return self.block(self.down(x))


class DownBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.block = ResidualBlock2D(out_channels, out_channels, dropout=dropout)

    def forward(self, x):
        return self.block(self.down(x))


class BottleneckTransformer(nn.Module):
    def __init__(self, channels, patch_size, num_layers=2, num_heads=8, dropout=0.1):
        super().__init__()
        token_shape = [max(1, int(x) // 16) for x in patch_size]
        max_tokens = token_shape[0] * token_shape[1] * token_shape[2]
        self.pos_embed = nn.Parameter(torch.zeros(1, max_tokens, channels))
        layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=int(num_heads),
            dim_feedforward=channels * 4,
            dropout=float(dropout),
            batch_first=True,
            activation="gelu",
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(num_layers))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        b, c, d, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        if tokens.shape[1] <= self.pos_embed.shape[1]:
            tokens = tokens + self.pos_embed[:, : tokens.shape[1]]
        tokens = self.encoder(tokens)
        return tokens.transpose(1, 2).reshape(b, c, d, h, w)


class BottleneckTransformer2D(nn.Module):
    def __init__(self, channels, patch_size, num_layers=2, num_heads=8, dropout=0.1):
        super().__init__()
        token_shape = [max(1, int(x) // 16) for x in patch_size]
        max_tokens = token_shape[0] * token_shape[1]
        self.pos_embed = nn.Parameter(torch.zeros(1, max_tokens, channels))
        layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=int(num_heads),
            dim_feedforward=channels * 4,
            dropout=float(dropout),
            batch_first=True,
            activation="gelu",
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(num_layers))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        if tokens.shape[1] <= self.pos_embed.shape[1]:
            tokens = tokens + self.pos_embed[:, : tokens.shape[1]]
        tokens = self.encoder(tokens)
        return tokens.transpose(1, 2).reshape(b, c, h, w)


class UpBlock3D(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.block = ResidualBlock3D(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class UpBlock2D(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, dropout=0.0):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.block = ResidualBlock2D(out_channels + skip_channels, out_channels, dropout=dropout)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class ResidualTransformerUNet3D(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        base_channels=24,
        patch_size=(64, 128, 128),
        transformer_layers=2,
        transformer_heads=8,
        transformer_dropout=0.1,
    ):
        super().__init__()
        c1 = int(base_channels)
        c2, c3, c4, c5 = c1 * 2, c1 * 4, c1 * 8, c1 * 16
        self.stem = nn.Conv3d(in_channels, c1, kernel_size=3, padding=1, bias=False)
        self.enc1 = ResidualBlock3D(c1, c1)
        self.enc2 = DownBlock3D(c1, c2)
        self.enc3 = DownBlock3D(c2, c3)
        self.enc4 = DownBlock3D(c3, c4)
        self.down4 = DownBlock3D(c4, c5)
        self.bottleneck = nn.Sequential(
            ResidualBlock3D(c5, c5),
            BottleneckTransformer(
                c5,
                patch_size=patch_size,
                num_layers=transformer_layers,
                num_heads=transformer_heads,
                dropout=transformer_dropout,
            ),
        )
        self.up4 = UpBlock3D(c5, c4, c4)
        self.up3 = UpBlock3D(c4, c3, c3)
        self.up2 = UpBlock3D(c3, c2, c2)
        self.up1 = UpBlock3D(c2, c1, c1)
        self.out = nn.Conv3d(c1, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.enc1(self.stem(x))
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.bottleneck(self.down4(x4))
        y = self.up4(x5, x4)
        y = self.up3(y, x3)
        y = self.up2(y, x2)
        y = self.up1(y, x1)
        return self.out(y)


class ResidualTransformerUNet2D(nn.Module):
    def __init__(
        self,
        in_channels=7,
        out_channels=1,
        base_channels=64,
        patch_size=(160, 160),
        transformer_layers=2,
        transformer_heads=8,
        transformer_dropout=0.1,
        residual_dropout=0.0,
    ):
        super().__init__()
        c1 = int(base_channels)
        c2, c3, c4, c5 = c1 * 2, c1 * 4, c1 * 8, c1 * 16
        self.stem = nn.Conv2d(in_channels, c1, kernel_size=3, padding=1, bias=False)
        self.enc1 = ResidualBlock2D(c1, c1, dropout=residual_dropout)
        self.enc2 = DownBlock2D(c1, c2, dropout=residual_dropout)
        self.enc3 = DownBlock2D(c2, c3, dropout=residual_dropout)
        self.enc4 = DownBlock2D(c3, c4, dropout=residual_dropout)
        self.down4 = DownBlock2D(c4, c5, dropout=residual_dropout)
        self.bottleneck = nn.Sequential(
            ResidualBlock2D(c5, c5, dropout=residual_dropout),
            BottleneckTransformer2D(
                c5,
                patch_size=patch_size,
                num_layers=transformer_layers,
                num_heads=transformer_heads,
                dropout=transformer_dropout,
            ),
        )
        self.up4 = UpBlock2D(c5, c4, c4, dropout=residual_dropout)
        self.up3 = UpBlock2D(c4, c3, c3, dropout=residual_dropout)
        self.up2 = UpBlock2D(c3, c2, c2, dropout=residual_dropout)
        self.up1 = UpBlock2D(c2, c1, c1, dropout=residual_dropout)
        self.out = nn.Conv2d(c1, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.enc1(self.stem(x))
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.bottleneck(self.down4(x4))
        y = self.up4(x5, x4)
        y = self.up3(y, x3)
        y = self.up2(y, x2)
        y = self.up1(y, x1)
        return self.out(y)


class Shallow3DEncoderUNet2D(nn.Module):
    def __init__(
        self,
        in_channels=7,
        out_channels=1,
        base_channels=48,
        patch_size=(160, 160),
        transformer_layers=1,
        transformer_heads=8,
        transformer_dropout=0.2,
        residual_dropout=0.15,
    ):
        super().__init__()
        depth = int(in_channels)
        c1 = int(base_channels)
        c2, c3, c4, c5 = c1 * 2, c1 * 4, c1 * 8, c1 * 16
        stem_mid = max(8, c1 // 2)
        self.depth = depth
        self.depth_stem = nn.Sequential(
            nn.Conv3d(1, stem_mid, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(stem_mid, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(stem_mid, c1, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(c1, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.depth_collapse = nn.Sequential(
            nn.Conv3d(c1, c1, kernel_size=(depth, 1, 1), bias=False),
            nn.InstanceNorm3d(c1, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.enc1 = ResidualBlock2D(c1, c1, dropout=residual_dropout)
        self.enc2 = DownBlock2D(c1, c2, dropout=residual_dropout)
        self.enc3 = DownBlock2D(c2, c3, dropout=residual_dropout)
        self.enc4 = DownBlock2D(c3, c4, dropout=residual_dropout)
        self.down4 = DownBlock2D(c4, c5, dropout=residual_dropout)
        self.bottleneck = nn.Sequential(
            ResidualBlock2D(c5, c5, dropout=residual_dropout),
            BottleneckTransformer2D(
                c5,
                patch_size=patch_size,
                num_layers=transformer_layers,
                num_heads=transformer_heads,
                dropout=transformer_dropout,
            ),
        )
        self.up4 = UpBlock2D(c5, c4, c4, dropout=residual_dropout)
        self.up3 = UpBlock2D(c4, c3, c3, dropout=residual_dropout)
        self.up2 = UpBlock2D(c3, c2, c2, dropout=residual_dropout)
        self.up1 = UpBlock2D(c2, c1, c1, dropout=residual_dropout)
        self.out = nn.Conv2d(c1, out_channels, kernel_size=1)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.depth:
            raise ValueError(f"Expected input shape [B, {self.depth}, H, W], got {tuple(x.shape)}")
        x = self.depth_stem(x.unsqueeze(1))
        x1 = self.enc1(self.depth_collapse(x).squeeze(2))
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.bottleneck(self.down4(x4))
        y = self.up4(x5, x4)
        y = self.up3(y, x3)
        y = self.up2(y, x2)
        y = self.up1(y, x1)
        return self.out(y)


def build_model(cfg, patch_size):
    model_cfg = cfg.get("model", {})
    family = str(model_cfg.get("family", "3dunet")).lower()
    if family == "2dunet":
        name = str(model_cfg.get("name", "ResidualTransformerUNet2D")).lower()
        if "shallow3d" in name or "2.5d" in name:
            return Shallow3DEncoderUNet2D(
                in_channels=int(model_cfg.get("in_channels", 7)),
                out_channels=int(model_cfg.get("out_channels", 1)),
                base_channels=int(model_cfg.get("base_channels", 48)),
                patch_size=patch_size,
                transformer_layers=int(model_cfg.get("transformer_layers", 1)),
                transformer_heads=int(model_cfg.get("transformer_heads", 8)),
                transformer_dropout=float(model_cfg.get("transformer_dropout", 0.2)),
                residual_dropout=float(model_cfg.get("residual_dropout", 0.15)),
            )
        return ResidualTransformerUNet2D(
            in_channels=int(model_cfg.get("in_channels", 7)),
            out_channels=int(model_cfg.get("out_channels", 1)),
            base_channels=int(model_cfg.get("base_channels", 64)),
            patch_size=patch_size,
            transformer_layers=int(model_cfg.get("transformer_layers", 2)),
            transformer_heads=int(model_cfg.get("transformer_heads", 8)),
            transformer_dropout=float(model_cfg.get("transformer_dropout", 0.1)),
            residual_dropout=float(model_cfg.get("residual_dropout", 0.0)),
        )
    return ResidualTransformerUNet3D(
        in_channels=int(model_cfg.get("in_channels", 1)),
        out_channels=int(model_cfg.get("out_channels", 1)),
        base_channels=int(model_cfg.get("base_channels", 24)),
        patch_size=patch_size,
        transformer_layers=int(model_cfg.get("transformer_layers", 2)),
        transformer_heads=int(model_cfg.get("transformer_heads", 8)),
        transformer_dropout=float(model_cfg.get("transformer_dropout", 0.1)),
    )
