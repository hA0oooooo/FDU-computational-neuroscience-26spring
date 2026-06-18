from pathlib import Path

import torch
import torch.nn as nn


def compute_output_size(i, kernel, padding, stride):
    return int(((i - kernel + 2 * padding) / stride) + 1)


class CNN8CLBConfig:
    def __init__(self, num_classes=2):
        self.input_dim = [73, 96, 96]
        self.out_channels = [8, 8, 16, 16, 32, 32, 64, 64]
        self.in_channels = [1] + [channels for channels in self.out_channels[:-1]]
        self.n_conv = len(self.out_channels)
        self.kernels = [(3, 3, 3)] * self.n_conv
        self.pooling = [
            (4, 4, 4),
            (0, 0, 0),
            (3, 3, 3),
            (0, 0, 0),
            (2, 2, 2),
            (0, 0, 0),
            (2, 2, 2),
            (0, 0, 0),
        ]
        for i in range(self.n_conv):
            for dim in range(3):
                if self.pooling[i][dim] != 0:
                    self.input_dim[dim] = compute_output_size(
                        self.input_dim[dim],
                        self.pooling[i][dim],
                        0,
                        self.pooling[i][dim],
                    )
        flattened = self.input_dim[0] * self.input_dim[1] * self.input_dim[2]
        self.fweights = [self.out_channels[-1] * flattened, int(num_classes)]
        self.dropout = 0.0


class CNN8CLB(nn.Module):
    def __init__(self, num_classes=2, dropout=0.0):
        super().__init__()
        param = CNN8CLBConfig(num_classes=num_classes)
        param.dropout = float(dropout)
        self.embedding = nn.ModuleList()
        for i in range(param.n_conv):
            padding = tuple(int((kernel - 1) / 2) for kernel in param.kernels[i])
            layers = [
                nn.Conv3d(
                    in_channels=param.in_channels[i],
                    out_channels=param.out_channels[i],
                    kernel_size=param.kernels[i],
                    stride=(1, 1, 1),
                    padding=padding,
                    bias=False,
                ),
                nn.BatchNorm3d(param.out_channels[i]),
                nn.ReLU(inplace=True),
            ]
            if param.pooling[i] != (0, 0, 0):
                layers.append(nn.MaxPool3d(param.pooling[i], stride=param.pooling[i]))
            self.embedding.append(nn.Sequential(*layers))
        self.ReLU = nn.ReLU(inplace=True)
        self.Dropout = nn.Dropout(p=param.dropout)
        self.f = nn.ModuleList()
        for i in range(len(param.fweights) - 1):
            self.f.append(nn.Linear(param.fweights[i], param.fweights[i + 1]))

    def forward_features(self, image, embedding_index=7):
        out = image
        for i, block in enumerate(self.embedding):
            out = block(out)
            if i == int(embedding_index):
                return out.view(out.size(0), -1)
        return out.view(out.size(0), -1)

    def forward(self, image):
        out = self.forward_features(image, embedding_index=len(self.embedding) - 1)
        for fc in self.f[:-1]:
            out = fc(out)
            out = self.ReLU(out)
            out = self.Dropout(out)
        return self.f[-1](out)


def load_3dcnn_pretrained(model, checkpoint_path, strict=False):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"3D-CNN checkpoint not found: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if strict:
        model.load_state_dict(state_dict)
        return
    model_state = model.state_dict()
    filtered = {
        key: value
        for key, value in state_dict.items()
        if key in model_state and tuple(model_state[key].shape) == tuple(value.shape)
    }
    model_state.update(filtered)
    model.load_state_dict(model_state)


def build_3dcnn_model(checkpoint_path, num_classes=3, dropout=0.0, load_pretrained=True):
    model = CNN8CLB(num_classes=num_classes, dropout=dropout)
    if load_pretrained:
        load_3dcnn_pretrained(model, checkpoint_path, strict=(int(num_classes) == 2))
    return model
