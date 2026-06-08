from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.data_augment import (
    augment_enabled,
    augment_multiplier,
    expanded_indices,
    get_sfcn_train_transform,
    make_sfcn_separate_augment_dataset,
    separate_augment_enabled,
)
from src.dataset import UKBAgeSexDataset, get_sfcn_transform
from src.metrics import (
    baseline_age_metrics,
    baseline_sex_metrics,
    classification_metrics,
    fold_summary,
    mean_metrics,
    regression_metrics,
)
from src.models_sfcn import AGE_BIN_CENTERS, SFCNWrapper, age_soft_labels
from src.utils import output_dir_from_config, set_seed, stratified_k_fold_indices, write_csv, write_json


def fold_baseline_metrics(dataset, train_idx, val_idx, task):
    if task == "age":
        base = baseline_age_metrics(dataset.ages(train_idx), dataset.ages(val_idx))
        return {"age_baseline_mae": base["mae"], "age_baseline_mse": base["mse"]}
    train_sex = [dataset.rows[i]["Sex"] for i in train_idx]
    val_sex = [dataset.rows[i]["Sex"] for i in val_idx]
    base = baseline_sex_metrics(train_sex, val_sex)
    return {"sex_baseline_acc": base["accuracy"], "sex_baseline_balanced_acc": base["balanced_accuracy"]}


def make_loader(dataset, indices, config, device, shuffle=False):
    subset = Subset(dataset, indices) if indices is not None else dataset
    return DataLoader(
        subset,
        batch_size=int(config.get("batch_size", 1)),
        shuffle=shuffle,
        num_workers=int(config.get("num_workers", 2)),
        pin_memory=device.type == "cuda",
    )


def age_sex_stratification_labels(dataset, num_folds):
    ages = np.asarray(dataset.ages(), dtype=float)
    sexes = [str(row["Sex"]) for row in dataset.rows]
    order = np.argsort(ages)
    for num_age_bins in (4, 3, 2, 1):
        age_bins = np.empty(len(ages), dtype=int)
        for rank, index in enumerate(order):
            age_bins[index] = min(num_age_bins - 1, rank * num_age_bins // len(ages))
        labels = [f"{sex}_{age_bin}" for sex, age_bin in zip(sexes, age_bins)]
        if min(Counter(labels).values()) >= num_folds:
            return labels
    return sexes


def evaluate_model(model, loader, device, task, sex_classes):
    model.eval()
    rows = []
    age_true, age_pred = [], []
    sex_true, sex_pred = [], []
    total_loss = 0.0
    n_samples = 0
    criterion = nn.KLDivLoss(reduction="batchmean") if task == "age" else nn.NLLLoss()
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device)
            log_prob = model.forward_log_prob(image)
            if task == "age":
                target = age_soft_labels(batch["age_raw"].detach().cpu().numpy()).to(device)
                loss = criterion(log_prob, target)
                pred_tensor = torch.exp(log_prob).matmul(AGE_BIN_CENTERS.to(log_prob.device))
                pred = pred_tensor.detach().cpu().numpy().tolist()
                truth = batch["age_raw"].detach().cpu().numpy().tolist()
                age_true.extend(truth)
                age_pred.extend(pred)
                for case_id, age in zip(batch["id"], pred):
                    rows.append({"ID": str(case_id), "Age": float(age), "Sex": ""})
            else:
                loss = criterion(log_prob, batch["sex"].to(device))
                pred_idx = torch.argmax(log_prob, dim=1).detach().cpu().numpy().tolist()
                true_idx = batch["sex"].detach().cpu().numpy().tolist()
                sex_true.extend(true_idx)
                sex_pred.extend(pred_idx)
                for case_id, idx in zip(batch["id"], pred_idx):
                    rows.append({"ID": str(case_id), "Age": "", "Sex": sex_classes[int(idx)]})
            batch_size = image.shape[0]
            total_loss += float(loss.detach().cpu()) * batch_size
            n_samples += batch_size

    if task == "age":
        metrics = regression_metrics(age_true, age_pred)
        return {"loss": total_loss / max(1, n_samples), "val_age_mae": metrics["mae"], "val_age_mse": metrics["mse"]}, rows
    metrics = classification_metrics(sex_true, sex_pred)
    return {
        "loss": total_loss / max(1, n_samples),
        "val_sex_acc": metrics["accuracy"],
        "val_sex_balanced_acc": metrics["balanced_accuracy"],
    }, rows


def train_one_epoch(model, loader, device, optimizer, task):
    model.train()
    if hasattr(model.model, "feature_extractor") and not any(param.requires_grad for param in model.model.feature_extractor.parameters()):
        model.model.feature_extractor.eval()
    total = 0.0
    n_samples = 0
    age_true, age_pred = [], []
    sex_true, sex_pred = [], []
    if task == "age":
        criterion = nn.KLDivLoss(reduction="batchmean")
    else:
        criterion = nn.NLLLoss()

    for batch in loader:
        image = batch["image"].to(device)
        optimizer.zero_grad(set_to_none=True)
        log_prob = model.forward_log_prob(image)
        if task == "age":
            target = age_soft_labels(batch["age_raw"].detach().cpu().numpy()).to(device)
            loss = criterion(log_prob, target)
            pred = torch.exp(log_prob.detach()).matmul(AGE_BIN_CENTERS.to(log_prob.device))
            age_true.extend(batch["age_raw"].detach().cpu().numpy().tolist())
            age_pred.extend(pred.cpu().numpy().tolist())
        else:
            loss = criterion(log_prob, batch["sex"].to(device))
            sex_true.extend(batch["sex"].detach().cpu().numpy().tolist())
            sex_pred.extend(torch.argmax(log_prob.detach(), dim=1).cpu().numpy().tolist())
        loss.backward()
        optimizer.step()
        batch_size = image.shape[0]
        total += float(loss.detach().cpu()) * batch_size
        n_samples += batch_size
    values = {"loss": total / max(1, n_samples)}
    if task == "age":
        metrics = regression_metrics(age_true, age_pred)
        values["age_mae"] = metrics["mae"]
        values["age_mse"] = metrics["mse"]
    else:
        metrics = classification_metrics(sex_true, sex_pred)
        values["sex_acc"] = metrics["accuracy"]
        values["sex_balanced_acc"] = metrics["balanced_accuracy"]
    return values


def set_sfcn_trainable(model, mode):
    if mode == "frozen":
        for param in model.model.feature_extractor.parameters():
            param.requires_grad = False
        for param in model.model.classifier.parameters():
            param.requires_grad = True
    elif mode == "finetune":
        for param in model.parameters():
            param.requires_grad = True
    else:
        raise ValueError(f"Unknown SFCN train mode: {mode}")


def build_sfcn_optimizer(model, config):
    lr = float(config.get("lr", 1e-5))
    lr_backbone = float(config.get("lr_backbone", lr))
    weight_decay = float(config.get("weight_decay", 0))
    param_groups = []
    backbone_params = [param for param in model.model.feature_extractor.parameters() if param.requires_grad]
    head_params = [param for param in model.model.classifier.parameters() if param.requires_grad]
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": lr_backbone})
    if head_params:
        param_groups.append({"params": head_params, "lr": lr})
    if not param_groups:
        raise ValueError("No trainable SFCN parameters found.")
    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


def run_pretrained_eval(config, config_path, device, dataset, task, output_dir, experiment_name):
    model = SFCNWrapper(
        sfcn_repo=config["sfcn_repo"],
        task=task,
        checkpoint_path=config.get("checkpoint_path"),
        dropout=bool(config.get("dropout", True)),
        load_pretrained=True,
    ).to(device)
    num_folds = int(config.get("num_folds", 5))
    folds = stratified_k_fold_indices(
        age_sex_stratification_labels(dataset, num_folds),
        num_folds=num_folds,
        seed=int(config["seed"]),
    )
    all_pred_rows = []
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(folds):
        loader = make_loader(dataset, val_idx, config, device, shuffle=False)
        metrics, pred_rows = evaluate_model(model, loader, device, task, dataset.sex_classes)
        metrics.update(fold_baseline_metrics(dataset, train_idx, val_idx, task))
        metrics.update(
            {
                "experiment_name": experiment_name,
                "trainer": "sfcn_pretrained_eval",
                "task": task,
                "fold": fold,
                "train_size": len(train_idx),
                "val_size": len(val_idx),
            }
        )
        fold_metrics.append(fold_summary(metrics))
        all_pred_rows.extend(pred_rows)

        if task == "age":
            print(
                f"fold {fold + 1:02d}/{len(folds)} "
                f"val_loss={metrics['loss']} val_age_mae={metrics['val_age_mae']} "
                f"age_baseline_mae={metrics['age_baseline_mae']}"
            )
        else:
            print(
                f"fold {fold + 1:02d}/{len(folds)} "
                f"val_loss={metrics['loss']} val_sex_acc={metrics['val_sex_acc']} "
                f"sex_baseline_acc={metrics['sex_baseline_acc']}"
            )

    write_csv(output_dir / "pred.csv", sorted(all_pred_rows, key=lambda row: row["ID"]))
    write_json(
        output_dir / "metrics.json",
        {
            "experiment_name": experiment_name,
            "trainer": "sfcn_pretrained_eval",
            "task": task,
            "folds": fold_metrics,
            "mean": mean_metrics(fold_metrics),
        },
    )


def run_supervised_training(config, config_path, device, dataset, train_dataset, task, output_dir, experiment_name, mode):
    num_folds = int(config.get("num_folds", 5))
    folds = stratified_k_fold_indices(
        age_sex_stratification_labels(dataset, num_folds),
        num_folds=num_folds,
        seed=int(config["seed"]),
    )
    all_pred_rows = []
    fold_metrics = []
    log_rows = []

    for fold, (train_idx, val_idx) in enumerate(folds):
        set_seed(int(config["seed"]) + fold)
        if separate_augment_enabled(config):
            train_loader = make_loader(
                make_sfcn_separate_augment_dataset(train_dataset, train_idx, config),
                None,
                config,
                device,
                shuffle=True,
            )
        else:
            train_loader = make_loader(
                train_dataset,
                expanded_indices(train_idx, augment_multiplier(config)),
                config,
                device,
                shuffle=True,
            )
        val_loader = make_loader(dataset, val_idx, config, device, shuffle=False)
        model = SFCNWrapper(
            sfcn_repo=config["sfcn_repo"],
            task=task,
            checkpoint_path=config.get("checkpoint_path"),
            dropout=bool(config.get("dropout", True)),
            load_pretrained=bool(config.get("load_pretrained", True)),
        ).to(device)
        set_sfcn_trainable(model, mode)
        optimizer = build_sfcn_optimizer(model, config)
        best_metric = float("inf") if task == "age" else -float("inf")
        best_path = output_dir / f"fold_{fold}.pt"
        best_row = None
        baseline = fold_baseline_metrics(dataset, train_idx, val_idx, task)

        for epoch in range(1, int(config.get("epochs", 10)) + 1):
            train_metrics = train_one_epoch(model, train_loader, device, optimizer, task)
            val_metrics, _ = evaluate_model(model, val_loader, device, task, dataset.sex_classes)
            row = {
                "fold": fold,
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
            }
            if task == "age":
                row["train_age_mae"] = train_metrics["age_mae"]
                row["val_age_mae"] = val_metrics["val_age_mae"]
            else:
                row["train_sex_acc"] = train_metrics["sex_acc"]
                row["val_sex_acc"] = val_metrics["val_sex_acc"]
                row["val_sex_balanced_acc"] = val_metrics["val_sex_balanced_acc"]
            score = val_metrics["val_age_mae"] if task == "age" else val_metrics["val_sex_balanced_acc"]
            is_best = score < best_metric if task == "age" else score > best_metric
            row["is_best"] = int(is_best)
            if is_best:
                best_metric = score
                best_row = {
                    "fold": fold,
                    "best_epoch": epoch,
                    "train_size": len(train_idx),
                    "val_size": len(val_idx),
                    **baseline,
                    **{key: value for key, value in val_metrics.items() if key != "loss"},
                }
                torch.save({"config": config, "fold": fold, "epoch": epoch, "model_state_dict": model.state_dict(), "metrics": best_row}, best_path)
            log_rows.append(row)
            write_csv(output_dir / "train_log.csv", log_rows)
            if task == "age":
                print(
                    f"fold {fold + 1:02d}/{len(folds)} epoch {epoch:03d} "
                    f"train_loss={train_metrics['loss']} train_age_mae={train_metrics['age_mae']} "
                    f"val_loss={val_metrics['loss']} val_age_mae={val_metrics['val_age_mae']}"
                )
            else:
                print(
                    f"fold {fold + 1:02d}/{len(folds)} epoch {epoch:03d} "
                    f"train_loss={train_metrics['loss']} train_sex_acc={train_metrics['sex_acc']} "
                    f"val_loss={val_metrics['loss']} val_sex_acc={val_metrics['val_sex_acc']}"
                )

        checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        best_metrics, pred_rows = evaluate_model(model, val_loader, device, task, dataset.sex_classes)
        final_metrics = dict(best_row or {})
        final_metrics.update({key: value for key, value in best_metrics.items() if key != "loss"})
        fold_metrics.append(fold_summary(final_metrics))
        all_pred_rows.extend(pred_rows)

    write_csv(output_dir / "pred.csv", sorted(all_pred_rows, key=lambda row: row["ID"]))
    write_json(
        output_dir / "metrics.json",
        {
            "experiment_name": experiment_name,
            "trainer": f"sfcn_{mode}",
            "mode": mode,
            "task": task,
            "folds": fold_metrics,
            "mean": mean_metrics(fold_metrics),
        },
    )


def run_sfcn_training(config, config_path, device):
    task = str(config.get("task", "age")).lower()
    if task not in {"age", "sex"}:
        raise ValueError("SFCN configs should use task: age or task: sex.")
    output_dir = output_dir_from_config(config_path, config["output_dir"])
    experiment_name = str(config.get("experiment_name", Path(config_path).stem))
    transform = get_sfcn_transform(tuple(config.get("sfcn_input_shape", [160, 192, 160])))
    if not Path(config["metadata_csv"]).exists():
        raise FileNotFoundError(
            f"SFCN metadata not found: {config['metadata_csv']}\n"
            "Run preprocessing first:\n"
            "  python data.py --dataset ukb --model sfcn"
        )
    dataset = UKBAgeSexDataset(config["metadata_csv"], transform=transform)
    train_transform = get_sfcn_train_transform(config) if augment_enabled(config) and not separate_augment_enabled(config) else transform
    train_dataset = UKBAgeSexDataset(config["metadata_csv"], transform=train_transform, sex_classes=dataset.sex_classes)
    mode = str(config.get("mode", "pretrained_eval")).lower()

    if mode in {"pretrained_eval", "baseline"}:
        run_pretrained_eval(config, config_path, device, dataset, task, output_dir, experiment_name)
    elif mode in {"frozen", "finetune"}:
        run_supervised_training(config, config_path, device, dataset, train_dataset, task, output_dir, experiment_name, mode)
    else:
        raise ValueError(f"Unknown SFCN mode: {mode}")
    print(f"Saved outputs to {output_dir}")
