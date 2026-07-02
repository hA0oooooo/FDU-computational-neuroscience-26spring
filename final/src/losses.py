import torch
import torch.nn as nn
import torch.nn.functional as F


class SSIM3DLoss(nn.Module):
    def __init__(self, window_size=7, data_range=4.0, eps=1e-8):
        super().__init__()
        self.window_size = int(window_size)
        self.data_range = float(data_range)
        self.eps = float(eps)

    def forward(self, pred, target):
        pad = self.window_size // 2
        mu_x = F.avg_pool3d(pred, self.window_size, stride=1, padding=pad)
        mu_y = F.avg_pool3d(target, self.window_size, stride=1, padding=pad)
        sigma_x = F.avg_pool3d(pred * pred, self.window_size, stride=1, padding=pad) - mu_x * mu_x
        sigma_y = F.avg_pool3d(target * target, self.window_size, stride=1, padding=pad) - mu_y * mu_y
        sigma_xy = F.avg_pool3d(pred * target, self.window_size, stride=1, padding=pad) - mu_x * mu_y
        c1 = (0.01 * self.data_range) ** 2
        c2 = (0.03 * self.data_range) ** 2
        numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
        denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
        ssim = numerator / (denominator + self.eps)
        return 1.0 - ssim.mean()


class SSIM2DLoss(nn.Module):
    def __init__(self, window_size=7, data_range=4.0, eps=1e-8):
        super().__init__()
        self.window_size = int(window_size)
        self.data_range = float(data_range)
        self.eps = float(eps)

    def forward(self, pred, target):
        pad = self.window_size // 2
        mu_x = F.avg_pool2d(pred, self.window_size, stride=1, padding=pad)
        mu_y = F.avg_pool2d(target, self.window_size, stride=1, padding=pad)
        sigma_x = F.avg_pool2d(pred * pred, self.window_size, stride=1, padding=pad) - mu_x * mu_x
        sigma_y = F.avg_pool2d(target * target, self.window_size, stride=1, padding=pad) - mu_y * mu_y
        sigma_xy = F.avg_pool2d(pred * target, self.window_size, stride=1, padding=pad) - mu_x * mu_y
        c1 = (0.01 * self.data_range) ** 2
        c2 = (0.03 * self.data_range) ** 2
        numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
        denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
        ssim = numerator / (denominator + self.eps)
        return 1.0 - ssim.mean()


class Gradient3DLoss(nn.Module):
    def forward(self, pred, target):
        loss_d = torch.mean(torch.abs(torch.diff(pred, dim=2) - torch.diff(target, dim=2)))
        loss_h = torch.mean(torch.abs(torch.diff(pred, dim=3) - torch.diff(target, dim=3)))
        loss_w = torch.mean(torch.abs(torch.diff(pred, dim=4) - torch.diff(target, dim=4)))
        return (loss_d + loss_h + loss_w) / 3.0


class Gradient2DLoss(nn.Module):
    def forward(self, pred, target):
        loss_h = torch.mean(torch.abs(torch.diff(pred, dim=2) - torch.diff(target, dim=2)))
        loss_w = torch.mean(torch.abs(torch.diff(pred, dim=3) - torch.diff(target, dim=3)))
        return (loss_h + loss_w) / 2.0


class CompositeTranslationLoss(nn.Module):
    def __init__(
        self,
        l1_weight=1.0,
        ssim_weight=0.5,
        grad_weight=0.1,
        ssim_window=7,
        ssim_data_range=4.0,
        spatial_dims=3,
        high_intensity_enabled=False,
        high_intensity_top_fraction=0.15,
        high_intensity_weight=2.0,
    ):
        super().__init__()
        self.l1_weight = float(l1_weight)
        self.ssim_weight = float(ssim_weight)
        self.grad_weight = float(grad_weight)
        self.high_intensity_enabled = bool(high_intensity_enabled)
        self.high_intensity_top_fraction = float(high_intensity_top_fraction)
        self.high_intensity_weight = float(high_intensity_weight)
        self.l1 = nn.L1Loss()
        if int(spatial_dims) == 2:
            self.ssim = SSIM2DLoss(window_size=ssim_window, data_range=ssim_data_range)
            self.grad = Gradient2DLoss()
        else:
            self.ssim = SSIM3DLoss(window_size=ssim_window, data_range=ssim_data_range)
            self.grad = Gradient3DLoss()

    def weighted_l1(self, pred, target):
        if not self.high_intensity_enabled:
            return self.l1(pred, target)
        fraction = min(max(self.high_intensity_top_fraction, 0.0), 1.0)
        if fraction <= 0.0 or self.high_intensity_weight <= 1.0:
            return self.l1(pred, target)
        flat = target.detach().flatten(1)
        threshold = torch.quantile(flat, 1.0 - fraction, dim=1)
        view_shape = [target.shape[0]] + [1] * (target.ndim - 1)
        high_mask = target.detach() >= threshold.reshape(view_shape)
        weights = torch.ones_like(target)
        weights = torch.where(high_mask, torch.full_like(weights, self.high_intensity_weight), weights)
        weighted_abs = weights * torch.abs(pred - target)
        return weighted_abs.sum() / weights.sum().clamp_min(1.0)

    def forward(self, pred, target):
        parts = {"l1": self.weighted_l1(pred, target)}
        loss = self.l1_weight * parts["l1"]
        if self.ssim_weight > 0:
            parts["ssim"] = self.ssim(pred, target)
            loss = loss + self.ssim_weight * parts["ssim"]
        if self.grad_weight > 0:
            parts["grad"] = self.grad(pred, target)
            loss = loss + self.grad_weight * parts["grad"]
        parts["loss"] = loss
        return parts


def build_loss(cfg):
    loss_cfg = cfg.get("loss", {})
    family = str(cfg.get("model", {}).get("family", "3dunet")).lower()
    high_cfg = loss_cfg.get("high_intensity", {})
    return CompositeTranslationLoss(
        l1_weight=loss_cfg.get("l1_weight", 1.0),
        ssim_weight=loss_cfg.get("ssim_weight", 0.5),
        grad_weight=loss_cfg.get("grad_weight", 0.1),
        ssim_window=loss_cfg.get("ssim_window", 7),
        ssim_data_range=loss_cfg.get("ssim_data_range", 4.0),
        spatial_dims=2 if family == "2dunet" else 3,
        high_intensity_enabled=high_cfg.get("enabled", False),
        high_intensity_top_fraction=high_cfg.get("top_fraction", 0.15),
        high_intensity_weight=high_cfg.get("weight", 2.0),
    )
