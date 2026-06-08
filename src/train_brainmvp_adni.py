from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Subset

from src.dataset import ADNILabelDataset, get_brainmvp_transform
from src.metrics import classification_metrics, majority_label
from src.models_brainmvp import BrainMVPLabelModel
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


def build_model(config, num_classes):
    return BrainMVPLabelModel(
        brainmvp_repo=config["brainmvp_repo"],
        checkpoint_path=config.get("checkpoint_path"),
        num_classes=num_classes,
        in_channels=int(config.get("in_channels", 1)),
        dropout=float(config.get("dropout", 0.0)),
        load_pretrained=bool(config.get("load_pretrained", True)),
    )


def extract_features(model, loader, device):
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


def run_brainmvp_adni_sklearn_training(config, config_path, device):
    output_dir = output_dir_from_config(config_path, config["output_dir"])
    experiment_name = str(config.get("experiment_name", Path(config_path).stem))
    num_folds = int(config.get("num_folds", 5))
    transform = get_brainmvp_transform(tuple(config.get("brainmvp_input_shape", [96, 96, 64])))
    dataset = ADNILabelDataset(config["metadata_csv"], transform=transform, label_classes=LABEL_CLASSES)
    labels = dataset.labels()
    if min(Counter(labels).values()) < num_folds:
        raise ValueError(f"ADNI label count is too small for {num_folds}-fold: {dict(Counter(labels))}")

    loader = make_loader(dataset, None, config, device, shuffle=False)
    model = build_model(config, num_classes=len(dataset.label_classes)).to(device)
    ids, embeddings = extract_features(model, loader, device)
    features = align_features(embeddings, ids, dataset)
    print(f"BrainMVP ADNI feature matrix: {features.shape}")

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
            "model": "brainmvp",
            "trainer": "sklearn",
            "task": "label",
            "label_classes": dataset.label_classes,
            "folds": fold_rows,
            "mean": mean_label_metrics(fold_rows),
        },
    )
    print(f"Saved outputs to {output_dir}")


def set_trainable(model, mode):
    if mode == "frozen":
        model.freeze_backbone()
        for param in model.label_head.parameters():
            param.requires_grad = True
    elif mode == "finetune":
        for param in model.parameters():
            param.requires_grad = True
    else:
        raise ValueError(f"Unknown BrainMVP ADNI train mode: {mode}")


def build_optimizer(model, config):
    lr_head = float(config.get("lr_head", config.get("lr", 1e-3)))
    lr_backbone = float(config.get("lr_backbone", lr_head))
    weight_decay = float(config.get("weight_decay", 0))
    groups = []
    backbone = [param for param in model.encoder.parameters() if param.requires_grad]
    head = [param for param in model.label_head.parameters() if param.requires_grad]
    if backbone:
        groups.append({"params": backbone, "lr": lr_backbone})
    if head:
        groups.append({"params": head, "lr": lr_head})
    if not groups:
        raise ValueError("No trainable BrainMVP parameters found.")
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


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


def run_brainmvp_adni_dl_training(config, config_path, device):
    output_dir = output_dir_from_config(config_path, config["output_dir"])
    experiment_name = str(config.get("experiment_name", Path(config_path).stem))
    num_folds = int(config.get("num_folds", 5))
    mode = str(config.get("mode", "finetune")).lower()
    transform = get_brainmvp_transform(tuple(config.get("brainmvp_input_shape", [96, 96, 64])))
    dataset = ADNILabelDataset(config["metadata_csv"], transform=transform, label_classes=LABEL_CLASSES)
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
        train_loader = make_loader(dataset, train_idx, config, device, shuffle=True)
        val_loader = make_loader(dataset, val_idx, config, device, shuffle=False)

        model = build_model(config, num_classes=len(dataset.label_classes)).to(device)
        set_trainable(model, mode)
        optimizer = build_optimizer(model, config)
        best_score = -1.0
        best_loss = float("inf")
        best_epoch = 0
        best_path = output_dir / f"fold_{fold}.pt"
        freeze_batchnorm = bool(config.get("freeze_batchnorm", True))

        for epoch in range(1, int(config.get("epochs", 20)) + 1):
            train_metrics = run_epoch(
                model,
                train_loader,
                device,
                criterion,
                optimizer,
                freeze_batchnorm=freeze_batchnorm,
            )
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
            "model": "brainmvp",
            "trainer": "dl",
            "mode": mode,
            "task": "label",
            "label_classes": dataset.label_classes,
            "folds": fold_rows,
            "mean": mean_label_metrics(fold_rows),
        },
    )
    print(f"Saved outputs to {output_dir}")
