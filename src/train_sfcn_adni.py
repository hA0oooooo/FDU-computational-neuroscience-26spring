from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, Subset

from src.data_augment import (
    augment_enabled,
    augment_multiplier,
    expanded_indices,
    get_sfcn_separate_train_transforms,
    get_sfcn_train_transform,
    separate_augment_enabled,
)
from src.dataset import ADNILabelDataset, get_sfcn_transform
from src.metrics import classification_metrics, majority_label
from src.models_sfcn import SFCNLabelModel
from src.utils import output_dir_from_config, set_seed, stratified_k_fold_indices, write_csv, write_json


LABEL_CLASSES = ["CN", "MCI", "AD"]


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


def print_fold_distribution(fold, train_labels, val_labels):
    print(f"fold {fold + 1}: train={dict(Counter(train_labels))} val={dict(Counter(val_labels))}")


def make_loader(dataset, indices, config, device, shuffle=False):
    subset = Subset(dataset, indices) if indices is not None else dataset
    return DataLoader(
        subset,
        batch_size=int(config.get("batch_size", 1)),
        shuffle=shuffle,
        num_workers=int(config.get("num_workers", 2)),
        pin_memory=device.type == "cuda",
    )


def build_sfcn_label_model(config, num_classes):
    return SFCNLabelModel(
        sfcn_repo=config["sfcn_repo"],
        pretrained_task=str(config.get("pretrained_task", "age")).lower(),
        checkpoint_path=config.get("checkpoint_path"),
        num_classes=num_classes,
        dropout=bool(config.get("dropout", True)),
        load_pretrained=bool(config.get("load_pretrained", True)),
    )


def extract_sfcn_features(model, loader, device):
    model.eval()
    ids = []
    features = []
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device)
            feature = model.forward_features(image)
            ids.extend([str(case_id) for case_id in batch["id"]])
            features.append(feature.detach().cpu().numpy())
    return ids, np.concatenate(features, axis=0)


def align_features(features, feature_ids, dataset):
    by_id = {case_id: i for i, case_id in enumerate(feature_ids)}
    missing = [row["ID"] for row in dataset.rows if row["ID"] not in by_id]
    if missing:
        raise ValueError(f"Feature extraction is missing metadata IDs: {missing[:5]}")
    return features[[by_id[row["ID"]] for row in dataset.rows]]


class ADNISeparateAugmentDataset(Dataset):
    def __init__(self, base_dataset, indices, transforms):
        self.base_dataset = base_dataset
        self.items = [(index, transform) for index in indices for transform in transforms]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        row_index, transform = self.items[idx]
        row = self.base_dataset.rows[row_index]
        sample = transform({"image": row["image_path"]})
        label = str(row["label"])
        return {
            "id": row["ID"],
            "image": sample["image"],
            "label": torch.tensor(self.base_dataset.label_to_index[label], dtype=torch.long),
            "label_name": label,
        }


def make_adni_separate_augment_dataset(base_dataset, indices, config):
    return ADNISeparateAugmentDataset(base_dataset, indices, get_sfcn_separate_train_transforms(config))


def run_sfcn_adni_sklearn_training(config, config_path, device):
    output_dir = output_dir_from_config(config_path, config["output_dir"])
    experiment_name = str(config.get("experiment_name", Path(config_path).stem))
    num_folds = int(config.get("num_folds", 5))
    transform = get_sfcn_transform(tuple(config.get("sfcn_input_shape", [160, 192, 160])))
    dataset = ADNILabelDataset(config["metadata_csv"], transform=transform, label_classes=LABEL_CLASSES)
    labels = dataset.labels()
    if min(Counter(labels).values()) < num_folds:
        raise ValueError(f"ADNI label count is too small for {num_folds}-fold: {dict(Counter(labels))}")

    loader = make_loader(dataset, None, config, device, shuffle=False)
    model = build_sfcn_label_model(config, num_classes=len(dataset.label_classes)).to(device)
    ids, embeddings = extract_sfcn_features(model, loader, device)
    features = align_features(embeddings, ids, dataset)
    print(f"SFCN ADNI feature matrix: {features.shape}")

    folds = stratified_k_fold_indices(labels, num_folds=num_folds, seed=int(config["seed"]))
    sk_cfg = config.get("sklearn", {})
    fold_rows = []
    all_pred_rows = []

    for fold, (train_idx, val_idx) in enumerate(folds):
        train_labels = dataset.labels(train_idx)
        val_labels = dataset.labels(val_idx)
        print_fold_distribution(fold, train_labels, val_labels)
        train_x = features[train_idx]
        val_x = features[val_idx]
        if bool(sk_cfg.get("standardize_features", True)):
            scaler = StandardScaler()
            train_x = scaler.fit_transform(train_x)
            val_x = scaler.transform(val_x)
            joblib.dump(scaler, output_dir / f"scaler_fold_{fold}.joblib")
        classifier = LogisticRegression(
            C=float(sk_cfg.get("C", 1.0)),
            penalty=str(sk_cfg.get("penalty", "l2")),
            max_iter=int(sk_cfg.get("max_iter", 2000)),
            solver="lbfgs",
            class_weight=sk_cfg.get("class_weight", "balanced"),
            random_state=int(config["seed"]) + fold,
        )
        classifier.fit(train_x, train_labels)
        pred = classifier.predict(val_x)
        joblib.dump(classifier, output_dir / f"classifier_fold_{fold}.joblib")

        metrics = {}
        metrics.update(majority_baseline_metrics(train_labels, val_labels))
        metrics.update(adni_classification_metrics(val_labels, pred, "model"))
        fold_row = {
            "fold": fold,
            "train_size": len(train_idx),
            "val_size": len(val_idx),
            **metrics,
            "confusion_matrix": confusion_matrix_rows(val_labels, pred, dataset.label_classes),
        }
        fold_rows.append(fold_row)
        for row_index, value in zip(val_idx, pred):
            all_pred_rows.append({"ID": dataset.rows[row_index]["ID"], "Pre": str(value)})
        print(
            f"fold {fold + 1:02d}/{num_folds} "
            f"val_acc={metrics['model_acc']} val_balanced_acc={metrics['model_balanced_acc']} "
            f"baseline_acc={metrics['majority_baseline_acc']}"
        )

    write_csv(output_dir / "pred.csv", sorted(all_pred_rows, key=lambda row: row["ID"]), fieldnames=["ID", "Pre"])
    write_json(output_dir / "label_mapping.json", label_mapping(dataset.label_classes))
    write_json(
        output_dir / "metrics.json",
        {
            "experiment_name": experiment_name,
            "dataset": "adni",
            "model": "sfcn",
            "trainer": "sklearn",
            "task": "label",
            "label_classes": dataset.label_classes,
            "folds": fold_rows,
            "mean": mean_label_metrics(fold_rows),
        },
    )
    print(f"Saved outputs to {output_dir}")


def set_sfcn_label_trainable(model, mode):
    if mode == "frozen":
        for param in model.model.feature_extractor.parameters():
            param.requires_grad = False
        for param in model.model.classifier.parameters():
            param.requires_grad = True
    elif mode == "finetune":
        for param in model.parameters():
            param.requires_grad = True
    else:
        raise ValueError(f"Unknown SFCN ADNI train mode: {mode}")


def build_optimizer(model, config):
    lr = float(config.get("lr", 1e-4))
    lr_backbone = float(config.get("lr_backbone", lr))
    weight_decay = float(config.get("weight_decay", 0))
    groups = []
    backbone = [param for param in model.model.feature_extractor.parameters() if param.requires_grad]
    head = [param for param in model.model.classifier.parameters() if param.requires_grad]
    if backbone:
        groups.append({"params": backbone, "lr": lr_backbone})
    if head:
        groups.append({"params": head, "lr": lr})
    if not groups:
        raise ValueError("No trainable SFCN ADNI parameters found.")
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


def run_epoch(model, loader, device, criterion, optimizer=None):
    train = optimizer is not None
    model.train(train)
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
        y_true.extend(label.detach().cpu().numpy().tolist())
        y_pred.extend(torch.argmax(logits, dim=1).detach().cpu().numpy().tolist())
    values = {"loss": total_loss / max(1, n_samples)}
    values.update(adni_classification_metrics(y_true, y_pred, "label"))
    return values


def predict(model, loader, device, label_classes):
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["image"].to(device))
            pred = torch.argmax(logits, dim=1).detach().cpu().numpy().tolist()
            for case_id, pred_idx in zip(batch["id"], pred):
                rows.append({"ID": str(case_id), "Pre": label_classes[int(pred_idx)]})
    return rows


def run_sfcn_adni_dl_training(config, config_path, device):
    output_dir = output_dir_from_config(config_path, config["output_dir"])
    experiment_name = str(config.get("experiment_name", Path(config_path).stem))
    num_folds = int(config.get("num_folds", 5))
    mode = str(config.get("mode", "finetune")).lower()
    transform = get_sfcn_transform(tuple(config.get("sfcn_input_shape", [160, 192, 160])))
    train_transform = get_sfcn_train_transform(config) if augment_enabled(config) and not separate_augment_enabled(config) else transform
    dataset = ADNILabelDataset(config["metadata_csv"], transform=transform, label_classes=LABEL_CLASSES)
    train_dataset = ADNILabelDataset(config["metadata_csv"], transform=train_transform, label_classes=dataset.label_classes)
    labels = dataset.labels()
    if min(Counter(labels).values()) < num_folds:
        raise ValueError(f"ADNI label count is too small for {num_folds}-fold: {dict(Counter(labels))}")

    folds = stratified_k_fold_indices(labels, num_folds=num_folds, seed=int(config["seed"]))
    criterion = nn.CrossEntropyLoss()
    fold_rows = []
    all_pred_rows = []
    log_rows = []

    for fold, (train_idx, val_idx) in enumerate(folds):
        set_seed(int(config["seed"]) + fold)
        train_labels = dataset.labels(train_idx)
        val_labels = dataset.labels(val_idx)
        print_fold_distribution(fold, train_labels, val_labels)
        if separate_augment_enabled(config):
            train_loader = make_loader(make_adni_separate_augment_dataset(train_dataset, train_idx, config), None, config, device, shuffle=True)
        else:
            train_loader = make_loader(train_dataset, expanded_indices(train_idx, augment_multiplier(config)), config, device, shuffle=True)
        val_loader = make_loader(dataset, val_idx, config, device, shuffle=False)

        model = build_sfcn_label_model(config, num_classes=len(dataset.label_classes)).to(device)
        set_sfcn_label_trainable(model, mode)
        optimizer = build_optimizer(model, config)
        best_score = -1.0
        best_loss = float("inf")
        best_epoch = 0
        best_path = output_dir / f"fold_{fold}.pt"

        for epoch in range(1, int(config.get("epochs", 20)) + 1):
            train_metrics = run_epoch(model, train_loader, device, criterion, optimizer)
            val_metrics = run_epoch(model, val_loader, device, criterion)
            is_best = val_metrics["label_balanced_acc"] > best_score
            if is_best:
                best_score = val_metrics["label_balanced_acc"]
                best_loss = val_metrics["loss"]
                best_epoch = epoch
                torch.save(
                    {
                        "config": config,
                        "fold": fold,
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "label_classes": dataset.label_classes,
                        "metrics": {"train": train_metrics, "val": val_metrics},
                    },
                    best_path,
                )
            log_row = {
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
                f"fold {fold + 1:02d}/{num_folds} epoch {epoch:03d} "
                f"train_loss={train_metrics['loss']} train_acc={train_metrics['label_acc']} "
                f"val_loss={val_metrics['loss']} val_acc={val_metrics['label_acc']} "
                f"val_balanced_acc={val_metrics['label_balanced_acc']}"
            )

        checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        pred_rows = predict(model, val_loader, device, dataset.label_classes)
        all_pred_rows.extend(pred_rows)
        val_pred = [row["Pre"] for row in pred_rows]
        baseline = majority_baseline_metrics(train_labels, val_labels)
        model_metrics = adni_classification_metrics(val_labels, val_pred, "model")
        fold_rows.append(
            {
                "fold": fold,
                "best_epoch": best_epoch,
                "train_size": len(train_idx),
                "val_size": len(val_idx),
                **baseline,
                **model_metrics,
                "confusion_matrix": confusion_matrix_rows(val_labels, val_pred, dataset.label_classes),
            }
        )

    write_csv(output_dir / "pred.csv", sorted(all_pred_rows, key=lambda row: row["ID"]), fieldnames=["ID", "Pre"])
    write_json(output_dir / "label_mapping.json", label_mapping(dataset.label_classes))
    write_json(
        output_dir / "metrics.json",
        {
            "experiment_name": experiment_name,
            "dataset": "adni",
            "model": "sfcn",
            "trainer": "dl",
            "mode": mode,
            "task": "label",
            "label_classes": dataset.label_classes,
            "folds": fold_rows,
            "mean": mean_label_metrics(fold_rows),
        },
    )
    print(f"Saved outputs to {output_dir}")
