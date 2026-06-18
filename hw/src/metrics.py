from collections import Counter

import numpy as np


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    error = y_pred - y_true
    return {
        "mae": float(np.mean(np.abs(error))),
        "mse": float(np.mean(error**2)),
    }


def classification_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    acc = float(np.mean(y_true == y_pred)) if len(y_true) else 0.0
    recalls = []
    f1_values = []
    for label in sorted(set(y_true.tolist())):
        mask = y_true == label
        true_positive = float(np.sum((y_true == label) & (y_pred == label)))
        false_positive = float(np.sum((y_true != label) & (y_pred == label)))
        false_negative = float(np.sum((y_true == label) & (y_pred != label)))
        recalls.append(true_positive / max(1.0, true_positive + false_negative))
        precision = true_positive / max(1.0, true_positive + false_positive)
        recall = true_positive / max(1.0, true_positive + false_negative)
        f1_values.append(2 * precision * recall / max(1e-12, precision + recall))
    return {
        "accuracy": acc,
        "balanced_accuracy": float(np.mean(recalls)) if recalls else 0.0,
        "macro_f1": float(np.mean(f1_values)) if f1_values else 0.0,
    }


def baseline_age_metrics(train_age, val_age):
    pred = np.full(len(val_age), float(np.mean(train_age)), dtype=float)
    return regression_metrics(val_age, pred)


def majority_label(labels):
    counts = Counter(labels)
    return sorted(counts.items(), key=lambda item: (-item[1], str(item[0])))[0][0]


def baseline_sex_metrics(train_sex, val_sex):
    pred_label = majority_label(train_sex)
    pred = [pred_label for _ in val_sex]
    return classification_metrics(val_sex, pred)


SUMMARY_METRIC_ORDER = [
    "age_baseline_mae",
    "age_baseline_mse",
    "val_age_mae",
    "val_age_mse",
    "val_age_mae_origin",
    "val_age_mse_origin",
    "sex_baseline_acc",
    "sex_baseline_balanced_acc",
    "val_sex_acc",
    "val_sex_balanced_acc",
]

FOLD_SUMMARY_ORDER = [
    "seed",
    "fold",
    "best_epoch",
    "train_size",
    "val_size",
    *SUMMARY_METRIC_ORDER,
]


def ordered_metric_row(row, keys):
    return {key: row[key] for key in keys if key in row}


def fold_summary(row):
    summary = dict(row)
    if "epoch" in summary and "best_epoch" not in summary:
        summary["best_epoch"] = summary["epoch"]
    return ordered_metric_row(summary, FOLD_SUMMARY_ORDER)


def mean_metrics(rows):
    values = {key: [] for key in SUMMARY_METRIC_ORDER}
    for row in rows:
        for key in SUMMARY_METRIC_ORDER:
            value = row.get(key)
            if isinstance(value, (int, float, np.floating)):
                values[key].append(float(value))
    return {key: float(np.mean(items)) for key, items in values.items() if items}
