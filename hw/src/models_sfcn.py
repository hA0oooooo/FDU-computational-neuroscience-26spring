import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


AGE_BIN_RANGE = (42, 82)
AGE_BIN_STEP = 1
AGE_BIN_CENTERS = torch.arange(AGE_BIN_RANGE[0] + 0.5, AGE_BIN_RANGE[1], AGE_BIN_STEP, dtype=torch.float32)


def sfcn_missing_message(sfcn_repo):
    return (
        f"SFCN repo not found: {sfcn_repo}\n"
        "Clone it manually with:\n"
        "  git clone --depth=1 https://github.com/ha-ha-ha-han/UKBiobank_deep_pretrain.git"
    )


def load_sfcn_class(sfcn_repo):
    sfcn_repo = Path(sfcn_repo)
    if not sfcn_repo.exists():
        raise FileNotFoundError(sfcn_missing_message(sfcn_repo))
    sys.path.insert(0, str(sfcn_repo))
    from dp_model.model_files.sfcn import SFCN

    return SFCN


def strip_dataparallel_prefix(state_dict):
    if not any(key.startswith("module.") for key in state_dict.keys()):
        return state_dict
    return {key.removeprefix("module."): value for key, value in state_dict.items()}


def default_checkpoint(sfcn_repo, task):
    if task == "age":
        return Path(sfcn_repo) / "brain_age" / "run_20190719_00_epoch_best_mae.p"
    if task == "sex":
        return Path(sfcn_repo) / "sex_prediction" / "run_20191008_00_epoch_last.p"
    raise ValueError("SFCN official checkpoints are single-task; use task age or sex.")


def sfcn_channel_number(task):
    if task == "sex":
        return [28, 58, 128, 256, 256, 64]
    return [32, 64, 128, 256, 256, 64]


def sfcn_output_dim(task):
    return 2 if task == "sex" else 40


class SFCNWrapper(nn.Module):
    def __init__(self, sfcn_repo, task, checkpoint_path=None, dropout=True, load_pretrained=True):
        super().__init__()
        if task not in {"age", "sex"}:
            raise ValueError("SFCNWrapper supports task age or sex. Use separate configs for SFCN tasks.")
        self.task = task
        self.sfcn_repo = Path(sfcn_repo)
        sfcn_cls = load_sfcn_class(self.sfcn_repo)
        self.model = sfcn_cls(
            output_dim=sfcn_output_dim(task),
            channel_number=sfcn_channel_number(task),
            dropout=bool(dropout),
        )
        if load_pretrained:
            checkpoint_path = Path(checkpoint_path) if checkpoint_path else default_checkpoint(self.sfcn_repo, task)
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"SFCN checkpoint not found: {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            self.model.load_state_dict(strip_dataparallel_prefix(state_dict), strict=True)

    def forward_log_prob(self, image):
        return self.model(image)[0].flatten(1)

    def forward_features(self, image):
        features = self.model.feature_extractor(image)
        pooled = self.model.classifier.average_pool(features)
        return pooled.flatten(1)

    def forward_age(self, image):
        log_prob = self.forward_log_prob(image)
        centers = AGE_BIN_CENTERS.to(log_prob.device)
        return torch.exp(log_prob).matmul(centers)

    def forward_sex_logits(self, image):
        return self.forward_log_prob(image)

    def forward(self, image):
        if self.task == "age":
            return self.forward_age(image), None
        return None, self.forward_sex_logits(image)


def age_soft_labels(age_values, sigma=1.0):
    from scipy.stats import norm

    values = np.asarray(age_values, dtype=float)
    centers = np.arange(AGE_BIN_RANGE[0] + 0.5, AGE_BIN_RANGE[1], AGE_BIN_STEP, dtype=float)
    labels = np.zeros((len(values), len(centers)), dtype=np.float32)
    for row, age in enumerate(values):
        for col, center in enumerate(centers):
            left = center - AGE_BIN_STEP / 2
            right = center + AGE_BIN_STEP / 2
            labels[row, col] = norm.cdf(right, loc=age, scale=sigma) - norm.cdf(left, loc=age, scale=sigma)
    labels = labels / np.maximum(labels.sum(axis=1, keepdims=True), 1e-8)
    return torch.tensor(labels, dtype=torch.float32)


class SFCNLabelModel(nn.Module):
    def __init__(
        self,
        sfcn_repo,
        pretrained_task="age",
        checkpoint_path=None,
        num_classes=3,
        dropout=True,
        load_pretrained=True,
    ):
        super().__init__()
        if pretrained_task not in {"age", "sex"}:
            raise ValueError("SFCNLabelModel pretrained_task must be age or sex.")
        self.pretrained_task = pretrained_task
        self.sfcn_repo = Path(sfcn_repo)
        sfcn_cls = load_sfcn_class(self.sfcn_repo)
        self.model = sfcn_cls(
            output_dim=sfcn_output_dim(pretrained_task),
            channel_number=sfcn_channel_number(pretrained_task),
            dropout=bool(dropout),
        )
        if load_pretrained:
            checkpoint_path = Path(checkpoint_path) if checkpoint_path else default_checkpoint(self.sfcn_repo, pretrained_task)
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"SFCN checkpoint not found: {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            self.model.load_state_dict(strip_dataparallel_prefix(state_dict), strict=True)
        in_channel = sfcn_channel_number(pretrained_task)[-1]
        classifier = nn.Sequential()
        classifier.add_module("average_pool", nn.AvgPool3d([5, 6, 5]))
        if bool(dropout):
            classifier.add_module("dropout", nn.Dropout(0.5))
        classifier.add_module("conv_6", nn.Conv3d(in_channel, int(num_classes), padding=0, kernel_size=1))
        self.model.classifier = classifier

    def forward_features(self, image):
        features = self.model.feature_extractor(image)
        pooled = self.model.classifier.average_pool(features)
        return pooled.flatten(1)

    def forward(self, image):
        features = self.model.feature_extractor(image)
        logits = self.model.classifier(features).flatten(1)
        return logits
