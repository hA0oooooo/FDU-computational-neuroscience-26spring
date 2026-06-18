from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    RandAdjustContrastd,
    RandAffined,
    RandFlipd,
    Resized,
    ScaleIntensityd,
    ToTensord,
)
from torch.utils.data import DataLoader, Subset

from src.dataset import ADNILabelDataset
from src.metrics import classification_metrics, majority_label
from src.models_rootstrap import ROOTSTRAP_LABELS, RootstrapDenseNet
from src.utils import output_dir_from_config, set_seed, stratified_k_fold_indices, write_csv, write_json


def label_mapping(label_classes):
    return {label: idx for idx, label in enumerate(label_classes)}


def adni_classification_metrics(y_true, y_pred, prefix):
    metrics = classification_metrics(y_true, y_pred)
    return {
        f"{prefix}_acc": metrics["accuracy"],
        f"{prefix}_balanced_acc": metrics["balanced_accuracy"],
        f"{prefix}_macro_f1": metrics["macro_f1"],
    }


def majority_baseline_metrics(train_labels, val_labels):
    pred_label = majority_label(train_labels)
    pred = [pred_label for _ in val_labels]
    return adni_classification_metrics(val_labels, pred, "majority_baseline")


def confusion_matrix_rows(y_true, y_pred, labels):
    index = {label: i for i, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for true, pred in zip(y_true, y_pred):
        matrix[index[str(true)]][index[str(pred)]] += 1
    return matrix


def mean_label_metrics(fold_rows):
    keys = [
        "majority_baseline_acc",
        "majority_baseline_balanced_acc",
        "majority_baseline_macro_f1",
        "model_acc",
        "model_balanced_acc",
        "model_macro_f1",
    ]
    mean = {}
    for key in keys:
        values = [float(row[key]) for row in fold_rows if key in row]
        if values:
            mean[key] = float(np.mean(values))
    return mean


def augment_enabled(config):
    settings = config.get("data_augment", False)
    if isinstance(settings, bool):
        return settings
    return bool(settings.get("enabled", False))


def rootstrap_transform(image_size=(96, 96, 96), train=False, config=None):
    settings = config.get("data_augment", {}) if config is not None else {}
    if not isinstance(settings, dict):
        settings = {}
    transforms = [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        ScaleIntensityd(keys=["image"]),
        Resized(keys=["image"], spatial_size=image_size, mode="trilinear"),
    ]
    if train:
        import math

        rotate_degrees = float(settings.get("rotate_degrees", 8))
        rotate_radians = math.radians(rotate_degrees)
        shift_voxels = int(settings.get("shift_voxels", 5))
        scale_range = settings.get("scale_range", [0.95, 1.05])
        gamma_range = settings.get("gamma_range", [0.85, 1.15])
        scale_delta = max(abs(float(scale_range[0]) - 1.0), abs(float(scale_range[1]) - 1.0))
        transforms.extend(
            [
                RandFlipd(keys=["image"], spatial_axis=0, prob=float(settings.get("flip_prob", 0.5))),
                RandAffined(
                    keys=["image"],
                    prob=float(settings.get("affine_prob", 0.5)),
                    rotate_range=(rotate_radians, rotate_radians, rotate_radians),
                    translate_range=(shift_voxels, shift_voxels, shift_voxels),
                    scale_range=(scale_delta, scale_delta, scale_delta),
                    mode="bilinear",
                    padding_mode="zeros",
                ),
                RandAdjustContrastd(
                    keys=["image"],
                    prob=float(settings.get("contrast_prob", 0.2)),
                    gamma=(float(gamma_range[0]), float(gamma_range[1])),
                ),
            ]
        )
    transforms.append(ToTensord(keys=["image"]))
    return Compose(transforms)


def make_loader(dataset, indices, config, device, shuffle=False):
    subset = Subset(dataset, indices) if indices is not None else dataset
    return DataLoader(
        subset,
        batch_size=int(config.get("batch_size", 2)),
        shuffle=shuffle,
        num_workers=int(config.get("num_workers", 2)),
        pin_memory=device.type == "cuda",
    )


def print_fold_distribution(fold, train_labels, val_labels):
    print(f"fold {fold + 1}: train={dict(Counter(train_labels))} val={dict(Counter(val_labels))}")


def predict(model, loader, device):
    model.eval()
    rows = []
    true_labels = []
    pred_labels = []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["image"].to(device))
            pred = torch.argmax(logits, dim=1).detach().cpu().numpy().tolist()
            labels = [ROOTSTRAP_LABELS[int(idx)] for idx in pred]
            for case_id, label, true_name in zip(batch["id"], labels, batch["label_name"]):
                rows.append({"ID": str(case_id), "Pre": label})
                true_labels.append(str(true_name))
                pred_labels.append(label)
    return rows, true_labels, pred_labels


def run_rootstrap_baseline(config, config_path, device):
    output_dir = output_dir_from_config(config_path, config["output_dir"])
    experiment_name = str(config.get("experiment_name", Path(config_path).stem))
    transform = rootstrap_transform(tuple(config.get("rootstrap_input_shape", [96, 96, 96])), train=False)
    dataset = ADNILabelDataset(config["metadata_csv"], transform=transform, label_classes=ROOTSTRAP_LABELS)
    loader = make_loader(dataset, None, config, device, shuffle=False)
    model = RootstrapDenseNet(config["checkpoint_path"], load_pretrained=True, dropout=float(config.get("dropout", 0))).to(device)
    pred_rows, y_true, y_pred = predict(model, loader, device)
    metrics = adni_classification_metrics(y_true, y_pred, "model")
    baseline = majority_baseline_metrics(y_true, y_true)
    rows = [
        {
            "fold": "all",
            "train_size": len(dataset),
            "val_size": len(dataset),
            **baseline,
            **metrics,
            "confusion_matrix": confusion_matrix_rows(y_true, y_pred, ROOTSTRAP_LABELS),
        }
    ]
    write_csv(output_dir / "pred.csv", sorted(pred_rows, key=lambda row: row["ID"]), fieldnames=["ID", "Pre"])
    write_json(output_dir / "label_mapping.json", label_mapping(ROOTSTRAP_LABELS))
    write_json(
        output_dir / "metrics.json",
        {
            "experiment_name": experiment_name,
            "dataset": "adni",
            "model": "rootstrap",
            "trainer": "baseline",
            "task": "label",
            "label_classes": ROOTSTRAP_LABELS,
            "folds": rows,
            "mean": mean_label_metrics(rows),
        },
    )
    print(f"Rootstrap baseline acc={metrics['model_acc']} balanced_acc={metrics['model_balanced_acc']}")
    print(f"Saved outputs to {output_dir}")


def set_batchnorm_eval(model):
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
            module.eval()


def run_epoch(model, loader, device, criterion, optimizer=None, freeze_batchnorm=False):
    train = optimizer is not None
    model.train(train)
    if train and freeze_batchnorm:
        set_batchnorm_eval(model)
    total_loss = 0.0
    n_samples = 0
    y_true = []
    y_pred = []
    for batch in loader:
        image = batch["image"].to(device)
        label = batch["label"].to(device)
        with torch.set_grad_enabled(train):
            logits = model(image)
            loss = criterion(logits, label)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        batch_size = image.shape[0]
        total_loss += float(loss.detach().cpu()) * batch_size
        n_samples += batch_size
        y_true.extend([ROOTSTRAP_LABELS[int(idx)] for idx in label.detach().cpu().numpy().tolist()])
        y_pred.extend([ROOTSTRAP_LABELS[int(idx)] for idx in torch.argmax(logits, dim=1).detach().cpu().numpy().tolist()])
    values = {"loss": total_loss / max(1, n_samples)}
    values.update(adni_classification_metrics(y_true, y_pred, "label"))
    return values


def build_optimizer(model, config):
    return torch.optim.Adam(
        model.parameters(),
        lr=float(config.get("lr", 1e-4)),
        weight_decay=float(config.get("weight_decay", 0)),
    )


def run_rootstrap_finetune(config, config_path, device):
    output_dir = output_dir_from_config(config_path, config["output_dir"])
    experiment_name = str(config.get("experiment_name", Path(config_path).stem))
    num_folds = int(config.get("num_folds", 5))
    seeds = config.get("seeds") or [int(config["seed"])]
    seeds = [int(seed) for seed in seeds]
    multi_seed = len(seeds) > 1
    image_size = tuple(config.get("rootstrap_input_shape", [96, 96, 96]))
    dataset = ADNILabelDataset(config["metadata_csv"], transform=rootstrap_transform(image_size, train=False), label_classes=ROOTSTRAP_LABELS)
    train_dataset = ADNILabelDataset(
        config["metadata_csv"],
        transform=rootstrap_transform(image_size, train=augment_enabled(config), config=config),
        label_classes=ROOTSTRAP_LABELS,
    )
    labels = dataset.labels()
    if min(Counter(labels).values()) < num_folds:
        raise ValueError(f"ADNI label count is too small for {num_folds}-fold: {dict(Counter(labels))}")

    criterion = nn.CrossEntropyLoss()
    fold_rows = []
    pred_by_id = {}
    log_rows = []
    freeze_batchnorm = bool(config.get("freeze_batchnorm", True))

    for seed in seeds:
        folds = stratified_k_fold_indices(labels, num_folds=num_folds, seed=seed)
        for fold, (train_idx, val_idx) in enumerate(folds):
            set_seed(seed + fold)
            train_labels = dataset.labels(train_idx)
            val_labels = dataset.labels(val_idx)
            print_fold_distribution(fold, train_labels, val_labels)
            train_loader = make_loader(train_dataset, train_idx, config, device, shuffle=True)
            val_loader = make_loader(dataset, val_idx, config, device, shuffle=False)
            model = RootstrapDenseNet(
                config["checkpoint_path"],
                load_pretrained=True,
                dropout=float(config.get("dropout", 0)),
            ).to(device)
            optimizer = build_optimizer(model, config)
            best_score = -1.0
            best_epoch = 0
            best_path = output_dir / (f"seed_{seed}_fold_{fold}.pt" if multi_seed else f"fold_{fold}.pt")

            for epoch in range(1, int(config.get("epochs", 20)) + 1):
                train_metrics = run_epoch(model, train_loader, device, criterion, optimizer, freeze_batchnorm=freeze_batchnorm)
                val_metrics = run_epoch(model, val_loader, device, criterion)
                is_best = val_metrics["label_balanced_acc"] > best_score
                if is_best:
                    best_score = val_metrics["label_balanced_acc"]
                    best_epoch = epoch
                    torch.save(
                        {
                            "config": config,
                            "seed": seed,
                            "fold": fold,
                            "epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "label_classes": ROOTSTRAP_LABELS,
                            "metrics": {"train": train_metrics, "val": val_metrics},
                        },
                        best_path,
                    )
                log_row = {
                    "seed": seed,
                    "fold": fold,
                    "epoch": epoch,
                    "train_loss": train_metrics["loss"],
                    "train_acc": train_metrics["label_acc"],
                    "train_balanced_acc": train_metrics["label_balanced_acc"],
                    "val_loss": val_metrics["loss"],
                    "val_acc": val_metrics["label_acc"],
                    "val_balanced_acc": val_metrics["label_balanced_acc"],
                    "val_macro_f1": val_metrics["label_macro_f1"],
                    "is_best": int(is_best),
                }
                log_rows.append(log_row)
                write_csv(output_dir / "train_log.csv", log_rows)
                print(
                    f"seed {seed} fold {fold + 1:02d}/{num_folds} epoch {epoch:03d} "
                    f"train_loss={train_metrics['loss']} train_acc={train_metrics['label_acc']} "
                    f"val_loss={val_metrics['loss']} val_acc={val_metrics['label_acc']} "
                    f"val_balanced_acc={val_metrics['label_balanced_acc']}"
                )

            checkpoint = torch.load(best_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            pred_rows, val_true, val_pred = predict(model, val_loader, device)
            for row in pred_rows:
                pred_by_id.setdefault(str(row["ID"]), []).append(row["Pre"])
            baseline = majority_baseline_metrics(train_labels, val_true)
            model_metrics = adni_classification_metrics(val_true, val_pred, "model")
            fold_rows.append(
                {
                    "seed": seed,
                    "fold": fold,
                    "best_epoch": best_epoch,
                    "train_size": len(train_idx),
                    "val_size": len(val_idx),
                    **baseline,
                    **model_metrics,
                    "confusion_matrix": confusion_matrix_rows(val_true, val_pred, ROOTSTRAP_LABELS),
                }
            )

    final_pred_rows = []
    for case_id, values in pred_by_id.items():
        counts = Counter(values)
        label = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
        final_pred_rows.append({"ID": case_id, "Pre": label})
    write_csv(output_dir / "pred.csv", sorted(final_pred_rows, key=lambda row: row["ID"]), fieldnames=["ID", "Pre"])
    write_json(output_dir / "label_mapping.json", label_mapping(ROOTSTRAP_LABELS))
    write_json(
        output_dir / "metrics.json",
        {
            "experiment_name": experiment_name,
            "dataset": "adni",
            "model": "rootstrap",
            "trainer": "dl",
            "mode": "finetune",
            "task": "label",
            "label_classes": ROOTSTRAP_LABELS,
            "folds": fold_rows,
            "mean": mean_label_metrics(fold_rows),
        },
    )
    print(f"Saved outputs to {output_dir}")
