import importlib.util
from pathlib import Path

import torch
import torch.nn as nn


def brainmvp_missing_message(repo):
    return (
        f"BrainMVP repo not found: {repo}\n"
        "Clone it manually:\n"
        "  git clone --depth=1 https://github.com/shaohao011/BrainMVP.git"
    )


def brainmvp_checkpoint_missing_message(path):
    return (
        f"BrainMVP checkpoint not found: {path}\n"
        "Download UniFormer weights from BrainMVP README and place them at:\n"
        "  models/brainmvp_uniformer.pth\n"
        "Suggested manual command if gdown is available:\n"
        "  gdown 1DTmz5WACESD0wfkZ2r0x-zjTwOgd9ov3 -O models/brainmvp_uniformer.pth"
    )


def load_uniformer_small(brainmvp_repo):
    repo = Path(brainmvp_repo)
    if not repo.exists():
        raise FileNotFoundError(brainmvp_missing_message(repo))
    uniformer_py = repo / "Downstream" / "model" / "uniformer.py"
    if not uniformer_py.exists():
        raise FileNotFoundError(f"BrainMVP downstream uniformer.py not found: {uniformer_py}")
    spec = importlib.util.spec_from_file_location("_brainmvp_downstream_uniformer", uniformer_py)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.uniformer_small


def checkpoint_state_dict(checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(brainmvp_checkpoint_missing_message(checkpoint_path))
    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        return state["state_dict"]
    return state


def strip_prefix(key, prefix):
    return key[len(prefix) :] if key.startswith(prefix) else key


def load_brainmvp_encoder_weights(encoder, checkpoint_path):
    raw_state = checkpoint_state_dict(checkpoint_path)
    target_state = encoder.state_dict()
    filtered = {}
    skipped = []

    for key, value in raw_state.items():
        mapped = strip_prefix(str(key), "module.")
        mapped = strip_prefix(mapped, "encoder.uniformer.")
        mapped = strip_prefix(mapped, "uniformer.")
        mapped = strip_prefix(mapped, "encoder.")
        if mapped in target_state and tuple(target_state[mapped].shape) == tuple(value.shape):
            filtered[mapped] = value
        elif mapped in target_state:
            skipped.append((mapped, tuple(value.shape), tuple(target_state[mapped].shape)))

    missing, unexpected = encoder.load_state_dict(filtered, strict=False)
    print(
        "BrainMVP encoder weights loaded: "
        f"matched={len(filtered)}, shape_skipped={len(skipped)}, missing={len(missing)}, unexpected={len(unexpected)}"
    )
    if skipped:
        print("BrainMVP shape-skipped keys:", [item[0] for item in skipped[:6]])
    if missing:
        print("BrainMVP missing keys sample:", list(missing)[:20])
    if unexpected:
        print("BrainMVP unexpected keys sample:", list(unexpected)[:20])


class BrainMVPLabelModel(nn.Module):
    def __init__(
        self,
        brainmvp_repo,
        checkpoint_path=None,
        num_classes=3,
        in_channels=1,
        dropout=0.0,
        load_pretrained=True,
    ):
        super().__init__()
        uniformer_small = load_uniformer_small(brainmvp_repo)
        self.encoder = uniformer_small(img_size=96, in_chans=in_channels)
        if load_pretrained:
            load_brainmvp_encoder_weights(self.encoder, checkpoint_path)
        self.dropout = nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity()
        self.label_head = nn.Linear(512, int(num_classes))

    def freeze_backbone(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward_features(self, image):
        features = self.encoder(image)
        x4 = features[-1]
        return x4.flatten(2).mean(dim=-1)

    def forward(self, image):
        return self.label_head(self.dropout(self.forward_features(image)))
