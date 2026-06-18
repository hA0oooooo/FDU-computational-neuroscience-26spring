import csv
import json
import os
import random
from pathlib import Path


def load_yaml(path):
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing Python dependency: yaml. Install the BrainIAC environment first:\n"
            "  pip install -r BrainIAC/requirements.txt"
        ) from exc

    with Path(path).open("r") as f:
        return yaml.safe_load(f)


def configure_gpu_from_config(config):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
    gpu_id = config.get("gpu_id")
    if gpu_id is not None and str(gpu_id).lower() not in {"", "none", "null", "cpu"}:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)


def set_seed(seed):
    import numpy as np

    try:
        import torch
    except ModuleNotFoundError:
        torch = None

    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def resolve_device(device_name):
    import torch

    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Config requested {device_name}, but CUDA is not available.")
    return torch.device(device_name)


def output_dir_from_config(config_path, output_root):
    output_dir = Path(output_root) / Path(config_path).stem
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def k_fold_indices(n_items, num_folds=5, seed=42):
    if num_folds < 2:
        raise ValueError("num_folds must be at least 2 for k-fold validation.")
    if num_folds > n_items:
        raise ValueError("num_folds cannot be larger than the number of samples.")

    indices = list(range(n_items))
    rng = random.Random(seed)
    rng.shuffle(indices)

    fold_sizes = [n_items // num_folds] * num_folds
    for i in range(n_items % num_folds):
        fold_sizes[i] += 1

    folds = []
    start = 0
    for fold_size in fold_sizes:
        val_indices = sorted(indices[start : start + fold_size])
        val_set = set(val_indices)
        train_indices = sorted(index for index in indices if index not in val_set)
        folds.append((train_indices, val_indices))
        start += fold_size
    return folds


def stratified_k_fold_indices(labels, num_folds=5, seed=42):
    if num_folds < 2:
        raise ValueError("num_folds must be at least 2 for k-fold validation.")
    if num_folds > len(labels):
        raise ValueError("num_folds cannot be larger than the number of samples.")

    from sklearn.model_selection import StratifiedKFold

    splitter = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    indices = list(range(len(labels)))
    return [
        (sorted(train_idx.tolist()), sorted(val_idx.tolist()))
        for train_idx, val_idx in splitter.split(indices, labels)
    ]


def write_csv(path, rows, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2)
