import csv
from pathlib import Path

from .config import model_family
from .io import load_nifti_dhw, save_nifti_dhw
from .infer import slice_stack_predict, sliding_window_predict
from .metrics import case_metrics, summarize_metrics
from .normalization import Normalizer
from .training import load_checkpoint


def write_summary_csv(path, summary):
    path = Path(path)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        for key, value in summary.items():
            writer.writerow({"metric": key, "value": value})


def load_model_checkpoint(checkpoint_path, model, device):
    ckpt = load_checkpoint(checkpoint_path, model, optimizer=None, scaler=None, map_location=device)
    normalizer = Normalizer.from_dict(ckpt["normalizer"])
    model.to(device)
    model.eval()
    return ckpt, normalizer


def evaluate_cases(cfg, model, normalizer, records, spec, output_paths, device, tag, write_outputs=True):
    rows = []
    patch_cfg = cfg.get("patch", {})
    infer_cfg = cfg.get("inference", {})
    prediction_dir = output_paths["predictions"]

    for idx, record in enumerate(records, start=1):
        source_raw, _ = load_nifti_dhw(record.path(spec["source"]))
        target_raw, target_img = load_nifti_dhw(record.path(spec["target"]))
        source_norm = normalizer.normalize(source_raw, spec["source"])
        family = model_family(cfg)
        if family == "2dunet":
            pred_norm = slice_stack_predict(
                model,
                source_norm,
                slice_offsets=cfg.get("slice", {}).get("slice_offsets", [-3, -2, -1, 0, 1, 2, 3]),
                device=device,
                amp=infer_cfg.get("amp", True),
                batch_size=infer_cfg.get("batch_size", 16),
                pad_multiple=infer_cfg.get("pad_multiple", 16),
            )
        else:
            pred_norm = sliding_window_predict(
                model,
                source_norm,
                patch_size=patch_cfg.get("size", [64, 128, 128]),
                stride=patch_cfg.get("stride", [32, 64, 64]),
                device=device,
                amp=infer_cfg.get("amp", True),
                batch_size=infer_cfg.get("batch_size", 1),
            )
        if normalizer.method == "minmax" and infer_cfg.get("clamp_minmax", True):
            import numpy as np

            pred_norm = np.clip(pred_norm, 0.0, 1.0)
        pred_raw = normalizer.inverse(pred_norm, spec["target"])
        pred_path = prediction_dir / f"{record.caseid}_{spec['target']}.nii.gz"
        save_nifti_dhw(pred_raw, target_img, pred_path)
        metrics = case_metrics(pred_raw, target_raw)
        row = {
            "caseid": record.caseid,
            "source": spec["source"],
            "target": spec["target"],
            "prediction_path": str(pred_path),
            **metrics,
        }
        rows.append(row)
        print(
            f"[eval:{tag}] {idx}/{len(records)} case={record.caseid} "
            f"mae={metrics['mae']:.6f} mse={metrics['mse']:.6f} "
            f"psnr={metrics['psnr']:.6f} ssim={metrics['ssim']:.6f}",
        )

    summary = summarize_metrics(rows)
    if write_outputs:
        csv_path = output_paths["base"] / "eval.csv"
        if rows:
            with Path(csv_path).open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
        summary_path = output_paths["base"] / "summary.csv"
        write_summary_csv(summary_path, summary)
        print(f"[eval:{tag}] summary={summary} metrics_csv={csv_path} summary_csv={summary_path}")
    else:
        print(f"[eval:{tag}] summary={summary}")
    return rows, summary
