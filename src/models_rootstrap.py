from pathlib import Path

import torch
import torch.nn as nn
from monai.networks.nets import DenseNet121


ROOTSTRAP_LABELS = ["AD", "MCI", "CN"]


def rootstrap_checkpoint_missing_message(path):
    return (
        f"Rootstrap checkpoint not found: {path}\n"
        "Download the valid Hugging Face weight file and place it at:\n"
        "  models/Alzheimer-Classifier-Demo/86_acc_model.pth"
    )


class RootstrapDenseNet(nn.Module):
    def __init__(self, checkpoint_path=None, load_pretrained=True):
        super().__init__()
        self.model = DenseNet121(spatial_dims=3, in_channels=1, out_channels=3)
        if load_pretrained:
            self.load_checkpoint(checkpoint_path)

    def load_checkpoint(self, checkpoint_path):
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists() or checkpoint_path.stat().st_size <= 1024:
            raise FileNotFoundError(rootstrap_checkpoint_missing_message(checkpoint_path))
        state = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if not isinstance(state, dict):
            raise RuntimeError(f"Unsupported Rootstrap checkpoint format: {type(state).__name__}")
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        print(
            "Rootstrap DenseNet121 weights loaded: "
            f"keys={len(state)}, missing={len(missing)}, unexpected={len(unexpected)}"
        )
        if missing:
            print("Rootstrap missing keys sample:", list(missing)[:20])
        if unexpected:
            print("Rootstrap unexpected keys sample:", list(unexpected)[:20])

    def forward(self, image):
        return self.model(image)
