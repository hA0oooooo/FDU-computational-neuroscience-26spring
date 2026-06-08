import csv
import importlib.util
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


def load_brainiac_backbone_class(brainiac_repo):
    model_py = Path(brainiac_repo) / "src" / "model.py"
    if not model_py.exists():
        raise FileNotFoundError(f"BrainIAC model.py not found: {model_py}")

    spec = importlib.util.spec_from_file_location("_brainiac_model", model_py)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ViTBackboneNet


def checkpoint_missing_message(checkpoint_path):
    return (
        f"BrainIAC checkpoint not found: {checkpoint_path}\n"
        "Download the BrainIAC weights from the checkpoint link in BrainIAC/README.md "
        "and place the backbone checkpoint at models/BrainIAC.ckpt, or update "
        "checkpoint_path in the YAML config."
    )


class BrainIACEncoder(nn.Module):
    def __init__(self, brainiac_repo, checkpoint_path):
        super().__init__()
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_missing_message(checkpoint_path))
        backbone_cls = load_brainiac_backbone_class(brainiac_repo)
        self.backbone = backbone_cls(simclr_ckpt_path=str(checkpoint_path))

    def forward(self, image):
        return self.backbone(image)


class BrainIACTaskModel(nn.Module):
    def __init__(self, brainiac_repo, checkpoint_path, num_sex_classes, dropout=0.0):
        super().__init__()
        self.encoder = BrainIACEncoder(brainiac_repo, checkpoint_path)
        self.backbone = self.encoder.backbone
        self.dropout = nn.Dropout(p=float(dropout))
        self.age_head = nn.Linear(768, 1)
        self.sex_head = nn.Linear(768, num_sex_classes)

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward_features(self, image):
        return self.encoder(image)

    def forward(self, image):
        features = self.dropout(self.forward_features(image))
        age = self.age_head(features).squeeze(1)
        sex_logits = self.sex_head(features)
        return age, sex_logits


class BrainIACLabelModel(nn.Module):
    def __init__(self, brainiac_repo, checkpoint_path, num_classes, dropout=0.0):
        super().__init__()
        self.encoder = BrainIACEncoder(brainiac_repo, checkpoint_path)
        self.backbone = self.encoder.backbone
        self.dropout = nn.Dropout(p=float(dropout))
        self.label_head = nn.Linear(768, int(num_classes))

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward_features(self, image):
        return self.encoder(image)

    def forward(self, image):
        features = self.dropout(self.forward_features(image))
        return self.label_head(features)


def extract_embeddings(model_or_encoder, loader, device, desc="Extract embeddings"):
    model_or_encoder.eval()
    ids = []
    features = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, unit="batch"):
            image = batch["image"].to(device)
            if hasattr(model_or_encoder, "forward_features"):
                embedding = model_or_encoder.forward_features(image)
            else:
                embedding = model_or_encoder(image)
            ids.extend([str(case_id) for case_id in batch["id"]])
            features.append(embedding.detach().cpu().numpy())
    return ids, np.concatenate(features, axis=0)


def save_embeddings(features_dir, name, ids, embeddings):
    features_dir = Path(features_dir)
    features_dir.mkdir(parents=True, exist_ok=True)
    np.save(features_dir / f"{name}_embeddings.npy", embeddings)
    ids_path = features_dir / f"{name}_ids.csv"
    with ids_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["ID"])
        writer.writeheader()
        writer.writerows({"ID": case_id} for case_id in ids)
    return features_dir / f"{name}_embeddings.npy", ids_path


def load_embedding_ids(path):
    with Path(path).open("r", newline="") as f:
        return [row["ID"] for row in csv.DictReader(f)]
