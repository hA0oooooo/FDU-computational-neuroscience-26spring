from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Subset

from src.dataset import UKBAgeSexDataset, get_brainiac_transform
from src.metrics import (
    baseline_age_metrics,
    baseline_sex_metrics,
    classification_metrics,
    fold_summary,
    mean_metrics,
    regression_metrics,
)
from src.models_brainiac import (
    BrainIACEncoder,
    BrainIACTaskModel,
    extract_embeddings,
    load_embedding_ids,
    save_embeddings,
)
from src.train_dl import build_optimizer, run_epoch, uses_age, uses_sex
from src.utils import k_fold_indices, output_dir_from_config, set_seed, write_csv, write_json


def ensure_frozen_embeddings(config, dataset, transform, device):
    features_dir = Path(config.get("features_dir", Path(config["processed_dir"]) / "features"))
    emb_path = features_dir / "brainiac_frozen_embeddings.npy"
    ids_path = features_dir / "brainiac_frozen_ids.csv"
    if emb_path.exists() and ids_path.exists():
        return np.load(emb_path), load_embedding_ids(ids_path)

    loader = DataLoader(
        dataset,
        batch_size=int(config.get("batch_size", 2)),
        shuffle=False,
        num_workers=int(config.get("num_workers", 2)),
        pin_memory=device.type == "cuda",
    )
    encoder = BrainIACEncoder(config["brainiac_repo"], config["checkpoint_path"]).to(device)
    ids, embeddings = extract_embeddings(encoder, loader, device, desc="BrainIAC frozen embeddings")
    save_embeddings(features_dir, "brainiac_frozen", ids, embeddings)
    return embeddings, ids


def align_embeddings_to_dataset(embeddings, feature_ids, dataset):
    by_id = {case_id: i for i, case_id in enumerate(feature_ids)}
    missing = [row["ID"] for row in dataset.rows if row["ID"] not in by_id]
    if missing:
        raise ValueError(f"Feature file is missing metadata IDs: {missing[:5]}")
    order = [by_id[row["ID"]] for row in dataset.rows]
    return embeddings[order]


def fold_baseline_metrics(dataset, train_idx, val_idx, task):
    metrics = {}
    if uses_age(task):
        base = baseline_age_metrics(dataset.ages(train_idx), dataset.ages(val_idx))
        metrics["age_baseline_mae"] = base["mae"]
        metrics["age_baseline_mse"] = base["mse"]
    if uses_sex(task):
        train_sex = [dataset.rows[i]["Sex"] for i in train_idx]
        val_sex = [dataset.rows[i]["Sex"] for i in val_idx]
        base = baseline_sex_metrics(train_sex, val_sex)
        metrics["sex_baseline_acc"] = base["accuracy"]
        metrics["sex_baseline_balanced_acc"] = base["balanced_accuracy"]
    return metrics


def train_finetuned_backbone_for_fold(config, dataset, transform, train_idx, val_idx, output_dir, fold, device, task):
    age_mean, age_std = dataset.age_stats(train_idx)
    fold_dataset = UKBAgeSexDataset(
        config["metadata_csv"],
        transform=transform,
        sex_classes=dataset.sex_classes,
        age_standardize=bool(config.get("age_standardize", True)),
        age_mean=age_mean,
        age_std=age_std,
    )
    train_loader = DataLoader(
        Subset(fold_dataset, train_idx),
        batch_size=int(config.get("batch_size", 1)),
        shuffle=True,
        num_workers=int(config.get("num_workers", 2)),
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        Subset(fold_dataset, val_idx),
        batch_size=int(config.get("batch_size", 1)),
        shuffle=False,
        num_workers=int(config.get("num_workers", 2)),
        pin_memory=device.type == "cuda",
    )
    model = BrainIACTaskModel(
        brainiac_repo=config["brainiac_repo"],
        checkpoint_path=config["checkpoint_path"],
        num_sex_classes=len(dataset.sex_classes),
        dropout=float(config.get("dropout", 0)),
    ).to(device)
    optimizer = build_optimizer(model, config, task)
    age_criterion = nn.MSELoss()
    sex_criterion = nn.CrossEntropyLoss()
    best_loss = float("inf")
    best_path = output_dir / f"fold_{fold}.pt"

    for epoch in range(1, int(config.get("epochs", 20)) + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            optimizer,
            age_criterion,
            sex_criterion,
            config,
            task,
            train=True,
            age_mean=age_mean,
            age_std=age_std,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            device,
            optimizer,
            age_criterion,
            sex_criterion,
            config,
            task,
            train=False,
            age_mean=age_mean,
            age_std=age_std,
        )
        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            torch.save(
                {
                    "config": config,
                    "epoch": epoch,
                    "fold": fold,
                    "backbone_state_dict": model.backbone.state_dict(),
                    "model_state_dict": model.state_dict(),
                    "age_mean": age_mean,
                    "age_std": age_std,
                    "metrics": {f"train_{k}": v for k, v in train_metrics.items()}
                    | {f"val_{k}": v for k, v in val_metrics.items()},
                },
                best_path,
            )
        parts = [
            f"fold_{fold} finetune epoch {epoch:03d}",
            f"train_loss={train_metrics['loss']}",
            f"val_loss={val_metrics['loss']}",
        ]
        if uses_age(task):
            parts.append(f"train_age_mae={train_metrics['age_mae']}")
            parts.append(f"val_age_mae={val_metrics['age_mae']}")
        if uses_sex(task):
            parts.append(f"train_sex_acc={train_metrics['sex_accuracy']}")
            parts.append(f"val_sex_acc={val_metrics['sex_accuracy']}")
        print(" ".join(parts))

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def extract_fold_embeddings(model, dataset, train_idx, val_idx, fold, device, config):
    loader_args = {
        "batch_size": int(config.get("batch_size", 2)),
        "shuffle": False,
        "num_workers": int(config.get("num_workers", 2)),
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(Subset(dataset, train_idx), **loader_args)
    val_loader = DataLoader(Subset(dataset, val_idx), **loader_args)
    _, train_features = extract_embeddings(model, train_loader, device, desc=f"fold_{fold} train embeddings")
    _, val_features = extract_embeddings(model, val_loader, device, desc=f"fold_{fold} val embeddings")
    return train_features, val_features


def fit_and_eval_sklearn(config, output_dir, fold, train_x, val_x, train_idx, val_idx, dataset, task):
    sk_cfg = config.get("sklearn", {})
    if bool(sk_cfg.get("standardize_features", True)):
        scaler = StandardScaler()
        train_x = scaler.fit_transform(train_x)
        val_x = scaler.transform(val_x)
        joblib.dump(scaler, output_dir / f"scaler_fold_{fold}.joblib")

    metrics = fold_baseline_metrics(dataset, train_idx, val_idx, task)
    pred_rows = []

    if uses_age(task):
        train_age = np.asarray(dataset.ages(train_idx), dtype=float)
        val_age = np.asarray(dataset.ages(val_idx), dtype=float)
        age_model = Ridge(alpha=float(sk_cfg.get("ridge_alpha", 1.0)))
        age_model.fit(train_x, train_age)
        age_pred = age_model.predict(val_x)
        age_metrics = regression_metrics(val_age, age_pred)
        metrics["val_age_mae"] = age_metrics["mae"]
        metrics["val_age_mse"] = age_metrics["mse"]
        joblib.dump(age_model, output_dir / f"age_ridge_fold_{fold}.joblib")
    else:
        age_pred = [None for _ in val_idx]

    if uses_sex(task):
        train_sex = np.asarray([dataset.rows[i]["Sex"] for i in train_idx])
        val_sex = np.asarray([dataset.rows[i]["Sex"] for i in val_idx])
        sex_model = LogisticRegression(
            C=float(sk_cfg.get("logistic_C", 1.0)),
            penalty=str(sk_cfg.get("logistic_penalty", "l2")),
            max_iter=int(sk_cfg.get("max_iter", 2000)),
            solver="lbfgs",
        )
        sex_model.fit(train_x, train_sex)
        sex_pred = sex_model.predict(val_x)
        sex_metrics = classification_metrics(val_sex, sex_pred)
        metrics["val_sex_acc"] = sex_metrics["accuracy"]
        metrics["val_sex_balanced_acc"] = sex_metrics["balanced_accuracy"]
        joblib.dump(sex_model, output_dir / f"sex_logreg_fold_{fold}.joblib")
    else:
        sex_pred = [None for _ in val_idx]

    for row_index, age_value, sex_value in zip(val_idx, age_pred, sex_pred):
        row = {"ID": dataset.rows[row_index]["ID"]}
        if uses_age(task):
            row["Age"] = float(age_value)
        if uses_sex(task):
            row["Sex"] = str(sex_value)
        pred_rows.append(row)

    return metrics, pred_rows


def run_sklearn_training(config, config_path, device):
    task = str(config.get("task", "joint")).lower()
    if task not in {"joint", "age", "sex"}:
        raise ValueError(f"Unknown task: {task}")
    feature_source = str(config.get("feature_source", "frozen")).lower()
    if feature_source not in {"frozen", "finetuned"}:
        raise ValueError(f"Unknown feature_source: {feature_source}")

    output_dir = output_dir_from_config(config_path, config["output_dir"])
    experiment_name = str(config.get("experiment_name", Path(config_path).stem))
    num_folds = int(config.get("num_folds", 5))
    transform = get_brainiac_transform(tuple(config.get("image_size", [96, 96, 96])))
    dataset = UKBAgeSexDataset(config["metadata_csv"], transform=transform)

    frozen_features = None
    if feature_source == "frozen":
        embeddings, feature_ids = ensure_frozen_embeddings(config, dataset, transform, device)
        frozen_features = align_embeddings_to_dataset(embeddings, feature_ids, dataset)

    folds = k_fold_indices(len(dataset), num_folds=num_folds, seed=int(config["seed"]))
    all_pred_rows = []
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(folds):
        set_seed(int(config["seed"]) + fold)

        if feature_source == "frozen":
            train_x = frozen_features[train_idx]
            val_x = frozen_features[val_idx]
        else:
            model = train_finetuned_backbone_for_fold(
                config, dataset, transform, train_idx, val_idx, output_dir, fold, device, task
            )
            train_x, val_x = extract_fold_embeddings(model, dataset, train_idx, val_idx, fold, device, config)

        metrics, pred_rows = fit_and_eval_sklearn(config, output_dir, fold, train_x, val_x, train_idx, val_idx, dataset, task)
        metrics.update(
            {
                "experiment_name": experiment_name,
                "trainer": "sklearn",
                "task": task,
                "feature_source": feature_source,
                "fold": fold,
                "train_size": len(train_idx),
                "val_size": len(val_idx),
            }
        )
        fold_metrics.append(fold_summary(metrics))
        all_pred_rows.extend(pred_rows)

        parts = [f"fold {fold + 1:02d}/{num_folds}"]
        if uses_age(task):
            parts.append(f"val_age_mae={metrics['val_age_mae']}")
            parts.append(f"age_baseline_mae={metrics['age_baseline_mae']}")
        if uses_sex(task):
            parts.append(f"val_sex_acc={metrics['val_sex_acc']}")
            parts.append(f"sex_baseline_acc={metrics['sex_baseline_acc']}")
        print(" ".join(parts))

    write_csv(output_dir / "pred.csv", sorted(all_pred_rows, key=lambda row: row["ID"]))
    write_json(
        output_dir / "metrics.json",
        {
            "experiment_name": experiment_name,
            "trainer": "sklearn",
            "task": task,
            "feature_source": feature_source,
            "folds": fold_metrics,
            "mean": mean_metrics(fold_metrics),
        },
    )
    print(f"Saved outputs to {output_dir}")
