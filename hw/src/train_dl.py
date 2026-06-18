from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
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
from src.models_brainiac import BrainIACTaskModel
from src.utils import k_fold_indices, output_dir_from_config, set_seed, write_csv, write_json


def uses_age(task):
    return task in {"joint", "age"}


def uses_sex(task):
    return task in {"joint", "sex"}


def build_optimizer(model, config, task):
    mode = str(config.get("mode", "finetune" if config.get("feature_source") == "finetuned" else "")).lower()
    head_params = []
    if uses_age(task):
        head_params.extend(model.age_head.parameters())
    if uses_sex(task):
        head_params.extend(model.sex_head.parameters())

    if mode == "frozen":
        model.freeze_backbone()
        return torch.optim.AdamW(head_params, lr=config["lr_head"], weight_decay=float(config.get("weight_decay", 0)))
    if mode == "finetune":
        return torch.optim.AdamW(
            [
                {"params": model.backbone.parameters(), "lr": config["lr_backbone"]},
                {"params": head_params, "lr": config["lr_head"]},
            ],
            weight_decay=float(config.get("weight_decay", 0)),
        )
    raise ValueError(f"Unknown training mode: {config['mode']}")


def denormalize_age(age, age_mean, age_std, enabled):
    if not enabled:
        return age
    return age * age_std + age_mean


def run_epoch(model, loader, device, optimizer, age_criterion, sex_criterion, config, task, train, age_mean, age_std):
    model.train(train)
    totals = {"loss": 0.0}
    if uses_age(task):
        totals["age_loss"] = 0.0
    if uses_sex(task):
        totals["sex_loss"] = 0.0

    n_samples = 0
    ages_true, ages_pred, sex_true, sex_pred = [], [], [], []

    for batch in loader:
        images = batch["image"].to(device)
        ages = batch["age"].to(device)
        ages_raw = batch["age_raw"].to(device)
        sex = batch["sex"].to(device)

        with torch.set_grad_enabled(train):
            pred_age, sex_logits = model(images)
            loss = torch.tensor(0.0, device=device)

            if uses_age(task):
                age_loss = age_criterion(pred_age, ages)
                loss = loss + float(config.get("age_loss_weight", 1.0)) * age_loss
            else:
                age_loss = None

            if uses_sex(task):
                sex_loss = sex_criterion(sex_logits, sex)
                loss = loss + float(config.get("sex_loss_weight", 1.0)) * sex_loss
            else:
                sex_loss = None

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        batch_size = images.shape[0]
        n_samples += batch_size
        totals["loss"] += float(loss.detach().cpu()) * batch_size
        if age_loss is not None:
            totals["age_loss"] += float(age_loss.detach().cpu()) * batch_size
            pred_age_raw = denormalize_age(
                pred_age.detach(),
                age_mean,
                age_std,
                bool(config.get("age_standardize", True)),
            )
            ages_true.extend(ages_raw.detach().cpu().numpy().tolist())
            ages_pred.extend(pred_age_raw.cpu().numpy().tolist())
        if sex_loss is not None:
            sex_true.extend(sex.detach().cpu().numpy().tolist())
            sex_pred.extend(torch.argmax(sex_logits, dim=1).detach().cpu().numpy().tolist())

    values = {key: value / max(1, n_samples) for key, value in totals.items()}
    if uses_age(task):
        values.update({f"age_{key}": value for key, value in regression_metrics(ages_true, ages_pred).items()})
    if uses_sex(task):
        values.update({f"sex_{key}": value for key, value in classification_metrics(sex_true, sex_pred).items()})
    return values


def predict_for_ensemble(model, loader, device, task, age_mean, age_std, age_standardize):
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            pred_age, sex_logits = model(images)
            row_batch = {"ids": [str(case_id) for case_id in batch["id"]]}
            if uses_age(task):
                pred_age = denormalize_age(pred_age, age_mean, age_std, age_standardize)
                row_batch["age"] = pred_age.cpu().numpy().tolist()
            if uses_sex(task):
                row_batch["sex_logits"] = sex_logits.cpu().numpy().tolist()
            rows.append(row_batch)
    return rows


def ensemble_predictions(config, checkpoint_paths, dataset, loader, device, task):
    totals = {}
    model = BrainIACTaskModel(
        brainiac_repo=config["brainiac_repo"],
        checkpoint_path=config["checkpoint_path"],
        num_sex_classes=len(dataset.sex_classes),
        dropout=float(config.get("dropout", 0)),
    ).to(device)

    for checkpoint_path in checkpoint_paths:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        for batch in predict_for_ensemble(
            model,
            loader,
            device,
            task,
            float(checkpoint.get("age_mean", 0.0)),
            float(checkpoint.get("age_std", 1.0)),
            bool(checkpoint["config"].get("age_standardize", True)),
        ):
            for i, case_id in enumerate(batch["ids"]):
                totals.setdefault(case_id, {"count": 0, "age_sum": 0.0, "logit_sum": None})
                totals[case_id]["count"] += 1
                if uses_age(task):
                    totals[case_id]["age_sum"] += float(batch["age"][i])
                if uses_sex(task):
                    logits = batch["sex_logits"][i]
                    if totals[case_id]["logit_sum"] is None:
                        totals[case_id]["logit_sum"] = [0.0 for _ in logits]
                    totals[case_id]["logit_sum"] = [
                        old + float(new) for old, new in zip(totals[case_id]["logit_sum"], logits)
                    ]

    rows = []
    for case_id, values in totals.items():
        row = {"ID": case_id}
        count = max(1, values["count"])
        if uses_age(task):
            row["Age"] = values["age_sum"] / count
        if uses_sex(task):
            logits = [value / count for value in values["logit_sum"]]
            sex_idx = max(range(len(logits)), key=lambda idx: logits[idx])
            row["Sex"] = dataset.sex_classes[int(sex_idx)]
        rows.append(row)
    return sorted(rows, key=lambda row: row["ID"])


def make_checkpoint_payload(config, model, sex_classes, fold, epoch, metrics, age_mean, age_std):
    return {
        "config": config,
        "fold": fold,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "sex_classes": sex_classes,
        "age_mean": age_mean,
        "age_std": age_std,
        "metrics": metrics,
    }


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


def add_model_metric_aliases(metrics, task, source):
    values = {}
    if uses_age(task):
        values[f"{source}_age_mae_origin"] = metrics[f"{source}_age_mae"]
        values[f"{source}_age_mse_origin"] = metrics[f"{source}_age_mse"]
        values[f"{source}_age_mae"] = metrics[f"{source}_age_mae"]
        values[f"{source}_age_mse"] = metrics[f"{source}_age_mse"]
    if uses_sex(task):
        values[f"{source}_sex_acc"] = metrics[f"{source}_sex_accuracy"]
        values[f"{source}_sex_balanced_acc"] = metrics[f"{source}_sex_balanced_accuracy"]
    return values


def compact_train_log_row(row, task):
    values = {
        "fold": row["fold"],
        "epoch": row["epoch"],
        "train_loss": row["train_loss"],
        "val_loss": row["val_loss"],
        "is_best": row["is_best"],
    }
    if uses_age(task):
        values.update(
            {
                "train_age_loss": row["train_age_loss"],
                "val_age_loss": row["val_age_loss"],
                "train_age_mae_origin": row["train_age_mae_origin"],
                "val_age_mae_origin": row["val_age_mae_origin"],
            }
        )
    if uses_sex(task):
        values.update(
            {
                "train_sex_loss": row["train_sex_loss"],
                "val_sex_loss": row["val_sex_loss"],
                "train_sex_acc": row["train_sex_acc"],
                "val_sex_acc": row["val_sex_acc"],
            }
        )
    return values


def run_dl_training(config, config_path, device):
    task = str(config.get("task", "joint")).lower()
    if task not in {"joint", "age", "sex"}:
        raise ValueError(f"Unknown task: {task}")

    output_dir = output_dir_from_config(config_path, config["output_dir"])
    experiment_name = str(config.get("experiment_name", Path(config_path).stem))
    num_folds = int(config.get("num_folds", 5))
    age_standardize = bool(config.get("age_standardize", True))

    transform = get_brainiac_transform(tuple(config.get("image_size", [96, 96, 96])))
    dataset = UKBAgeSexDataset(config["metadata_csv"], transform=transform)
    all_loader = DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=int(config["num_workers"]),
        pin_memory=device.type == "cuda",
    )

    age_criterion = nn.MSELoss()
    sex_criterion = nn.CrossEntropyLoss()
    log_rows = []
    best_checkpoint_paths = []
    fold_metric_rows = []

    folds = k_fold_indices(len(dataset), num_folds=num_folds, seed=int(config["seed"]))
    for fold, (train_idx, val_idx) in enumerate(folds):
        set_seed(int(config["seed"]) + fold)
        age_mean, age_std = dataset.age_stats(train_idx)
        fold_dataset = UKBAgeSexDataset(
            config["metadata_csv"],
            transform=transform,
            sex_classes=dataset.sex_classes,
            age_standardize=age_standardize,
            age_mean=age_mean,
            age_std=age_std,
        )
        train_loader = DataLoader(
            Subset(fold_dataset, train_idx),
            batch_size=int(config["batch_size"]),
            shuffle=True,
            num_workers=int(config["num_workers"]),
            pin_memory=device.type == "cuda",
        )
        val_loader = DataLoader(
            Subset(fold_dataset, val_idx),
            batch_size=int(config["batch_size"]),
            shuffle=False,
            num_workers=int(config["num_workers"]),
            pin_memory=device.type == "cuda",
        )

        model = BrainIACTaskModel(
            brainiac_repo=config["brainiac_repo"],
            checkpoint_path=config["checkpoint_path"],
            num_sex_classes=len(dataset.sex_classes),
            dropout=float(config.get("dropout", 0)),
        ).to(device)
        optimizer = build_optimizer(model, config, task)

        best_val_loss = float("inf")
        best_metrics = {}
        best_checkpoint_path = output_dir / f"fold_{fold}.pt"
        baseline = fold_baseline_metrics(dataset, train_idx, val_idx, task)

        for epoch in range(1, int(config["epochs"]) + 1):
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

            row = {
                "experiment_name": experiment_name,
                "trainer": "dl",
                "task": task,
                "mode": config["mode"],
                "fold": fold,
                "epoch": epoch,
                "train_size": len(train_idx),
                "val_size": len(val_idx),
                "age_standardize": int(age_standardize),
                "age_mean": age_mean,
                "age_std": age_std,
            }
            row.update(baseline)
            row.update({f"train_{key}": value for key, value in train_metrics.items()})
            row.update({f"val_{key}": value for key, value in val_metrics.items()})
            row.update(add_model_metric_aliases(row, task, "train"))
            row.update(add_model_metric_aliases(row, task, "val"))

            is_best = val_metrics["loss"] < best_val_loss
            row["is_best"] = int(is_best)
            if is_best:
                best_val_loss = val_metrics["loss"]
                best_metrics = row.copy()
                torch.save(
                    make_checkpoint_payload(
                        config,
                        model,
                        dataset.sex_classes,
                        fold,
                        epoch,
                        best_metrics,
                        age_mean,
                        age_std,
                    ),
                    best_checkpoint_path,
                )

            log_rows.append(compact_train_log_row(row, task))
            write_csv(output_dir / "train_log.csv", log_rows)

            parts = [
                f"fold {fold + 1:02d}/{num_folds}",
                f"epoch {epoch:03d}",
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

        best_checkpoint_paths.append(best_checkpoint_path)
        fold_metric_rows.append(fold_summary(best_metrics))

    pred_rows = ensemble_predictions(config, best_checkpoint_paths, dataset, all_loader, device, task)
    write_csv(output_dir / "pred.csv", pred_rows)
    metrics = {
        "experiment_name": experiment_name,
        "trainer": "dl",
        "task": task,
        "folds": fold_metric_rows,
        "mean": mean_metrics(fold_metric_rows),
    }
    write_json(output_dir / "metrics.json", metrics)
    print(f"Saved outputs to {output_dir}")
