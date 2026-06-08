import csv
import tempfile
from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from tqdm import tqdm

from src.preprocess_common import (
    PROJECT_ROOT,
    case_dirs,
    clean_value,
    find_input_image,
    read_csv_rows,
    read_input_as_image,
)


ADNI_RAW_CANDIDATES = [
    PROJECT_ROOT / "dataset" / "ADNI_data_105cases",
    PROJECT_ROOT / "dataset" / "ADNI_data",
]
PROCESSED_DIR = PROJECT_ROOT / "dataset" / "processed_3dcnn" / "ADNI"
ARRAYS_DIR = PROCESSED_DIR / "arrays"
TARGET_SHAPE = (96, 96, 73)
LABELS = {"CN", "MCI", "AD"}


def find_adni_raw_dir():
    for raw_dir in ADNI_RAW_CANDIDATES:
        if raw_dir.exists():
            return raw_dir
    raise FileNotFoundError("Missing ADNI directory. Extract dataset/ADNI_data_105cases.tar.gz first.")


def find_adni_csv(raw_dir):
    csv_files = sorted(Path(raw_dir).glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV found under {raw_dir}.")
    return csv_files[0]


def pick_column(columns, candidates):
    normalized = {c.lower().replace("_", "").replace("-", ""): c for c in columns}
    for candidate in candidates:
        key = candidate.lower().replace("_", "").replace("-", "")
        if key in normalized:
            return normalized[key]
    raise ValueError(f"Could not find any of {candidates} in CSV columns: {columns}")


def pick_label_column(columns, rows):
    try:
        return pick_column(columns, ["label", "diagnosis", "dx", "group", "class"])
    except ValueError:
        for column in columns:
            values = {str(row[column]).strip().upper() for row in rows if str(row[column]).strip()}
            if values and values <= (LABELS | {"DEMENTIA"}):
                return column
    raise ValueError(f"Could not infer ADNI label column from CSV columns: {columns}")


def normalize_label(value):
    text = str(value).strip().upper()
    if text == "DEMENTIA":
        return "AD"
    if text in LABELS:
        return text
    raise ValueError(f"Unsupported ADNI label: {value}")


def resize_data_volume_by_scale(data, scale):
    scale_list = [scale, scale, scale] if isinstance(scale, float) else scale
    return ndimage.zoom(data, scale_list, order=0)


def img_processing(image, scaling=0.5, final_size=TARGET_SHAPE):
    image = resize_data_volume_by_scale(image, scale=scaling)
    new_scaling = [final_size[i] / image.shape[i] for i in range(3)]
    return resize_data_volume_by_scale(image, scale=new_scaling).astype(np.float32)


def normalize_nonzero(image):
    image = image.astype(np.float32, copy=True)
    mask = image != 0
    if not np.any(mask):
        return image, 0.0, 0.0
    mean = float(image[mask].mean())
    std = float(image[mask].std())
    if std <= 1e-6:
        image = image - mean
    else:
        image = (image - mean) / std
    return image, mean, std


def read_nifti_zyx(path):
    data = nib.load(str(path)).get_fdata(dtype=np.float32)
    return np.transpose(np.asarray(data, dtype=np.float32), (2, 1, 0))


def read_image_zyx(input_path, input_kind, work_dir, case_id):
    if input_kind == "nifti":
        return read_nifti_zyx(input_path)
    image = read_input_as_image(input_path, input_kind, work_dir, case_id)
    return sitk.GetArrayFromImage(image).astype(np.float32)


def stats(values):
    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std()),
    }


def fail_status(reason):
    return f"fail: {' '.join(str(reason).split())[:300]}"


def build_adni_3dcnn_metadata():
    raw_dir = find_adni_raw_dir()
    csv_path = find_adni_csv(raw_dir)
    columns, rows = read_csv_rows(csv_path)
    id_col = pick_column(columns, ["ID", "eid", "subject", "subject_id", "caseid", "case_id"])
    label_col = pick_label_column(columns, rows)
    row_by_id = {clean_value(row[id_col]): row for row in rows}
    label_counts = Counter(normalize_label(row[label_col]) for row in rows)

    print("ADNI 3D-CNN preprocessing")
    print(f"Raw data: {raw_dir.relative_to(PROJECT_ROOT)}")
    print(f"CSV: {csv_path.relative_to(PROJECT_ROOT)}")
    print(f"Processed dir: {PROCESSED_DIR.relative_to(PROJECT_ROOT)}")
    print(f"Arrays: {ARRAYS_DIR.relative_to(PROJECT_ROOT)}")
    print(f"CSV samples: {len(rows)}")
    print("Label distribution:", dict(label_counts))

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    if ARRAYS_DIR.exists():
        for path in ARRAYS_DIR.glob("*.npy"):
            path.unlink()
    ARRAYS_DIR.mkdir(parents=True, exist_ok=True)

    dir_by_id = {clean_value(path.name): path for path in case_dirs(raw_dir)}
    metadata = []
    details = []
    failed = []

    for case_id in tqdm(sorted(row_by_id), desc="Preprocessing ADNI 3D-CNN", unit="case"):
        row = row_by_id[case_id]
        label = normalize_label(row[label_col])
        case_dir = dir_by_id.get(case_id)
        detail = {
            "ID": case_id,
            "source_file": "",
            "input_kind": "",
            "axis_order": "zyx",
            "input_shape": "",
            "scaled_shape": "",
            "final_shape": "",
            "normalization_method": "official_torch_norm_nonzero_mean_std",
            "raw_min": "",
            "raw_max": "",
            "raw_mean": "",
            "raw_std": "",
            "normalized_min": "",
            "normalized_max": "",
            "normalized_mean": "",
            "normalized_std": "",
            "nonzero_mean": "",
            "nonzero_std": "",
            "preprocessing_status": "",
            "error_message": "",
        }

        if case_dir is None:
            reason = "missing image folder"
            status = fail_status(reason)
            metadata.append({"ID": case_id, "image_path": "", "label": label, "preprocessing_status": status})
            detail["preprocessing_status"] = status
            detail["error_message"] = reason
            details.append(detail)
            failed.append(case_id)
            continue

        input_path, input_kind = find_input_image(case_dir)
        if input_path is None:
            reason = "missing image"
            status = fail_status(reason)
            metadata.append({"ID": case_id, "image_path": "", "label": label, "preprocessing_status": status})
            detail["preprocessing_status"] = status
            detail["error_message"] = reason
            details.append(detail)
            failed.append(case_id)
            continue

        output_path = ARRAYS_DIR / f"{case_id}.npy"
        try:
            with tempfile.TemporaryDirectory(prefix=f"3dcnn_{case_id}_", dir=str(PROCESSED_DIR)) as work_dir:
                image = read_image_zyx(input_path, input_kind, work_dir, case_id)
            scaled = resize_data_volume_by_scale(image, scale=0.5)
            processed = img_processing(image, scaling=0.5, final_size=TARGET_SHAPE)
            raw_stats = stats(processed)
            normalized, nonzero_mean, nonzero_std = normalize_nonzero(processed)
            norm_stats = stats(normalized)
            if normalized.shape != TARGET_SHAPE:
                raise RuntimeError(f"final shape is {normalized.shape}, expected {TARGET_SHAPE}")
            np.save(output_path, normalized.astype(np.float32))
            detail.update(
                {
                    "source_file": str(input_path.relative_to(PROJECT_ROOT)) if input_kind == "nifti" else str(case_dir.relative_to(PROJECT_ROOT)),
                    "input_kind": input_kind,
                    "input_shape": "x".join(str(v) for v in image.shape),
                    "scaled_shape": "x".join(str(v) for v in scaled.shape),
                    "final_shape": "x".join(str(v) for v in normalized.shape),
                    "raw_min": raw_stats["min"],
                    "raw_max": raw_stats["max"],
                    "raw_mean": raw_stats["mean"],
                    "raw_std": raw_stats["std"],
                    "normalized_min": norm_stats["min"],
                    "normalized_max": norm_stats["max"],
                    "normalized_mean": norm_stats["mean"],
                    "normalized_std": norm_stats["std"],
                    "nonzero_mean": nonzero_mean,
                    "nonzero_std": nonzero_std,
                    "preprocessing_status": "success",
                }
            )
            metadata.append(
                {
                    "ID": case_id,
                    "image_path": str(output_path.relative_to(PROJECT_ROOT)),
                    "label": label,
                    "preprocessing_status": "success",
                }
            )
        except Exception as exc:
            status = fail_status(exc)
            metadata.append({"ID": case_id, "image_path": "", "label": label, "preprocessing_status": status})
            detail["preprocessing_status"] = status
            detail["error_message"] = str(exc)
            failed.append(case_id)
        details.append(detail)

    metadata_csv = PROCESSED_DIR / "metadata.csv"
    with metadata_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["ID", "image_path", "label", "preprocessing_status"])
        writer.writeheader()
        writer.writerows(metadata)

    details_csv = PROCESSED_DIR / "details.csv"
    with details_csv.open("w", newline="") as f:
        fieldnames = list(details[0].keys()) if details else ["ID"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(details)

    success_rows = [row for row in metadata if row["image_path"] and not row["preprocessing_status"].startswith("fail")]
    print("\nADNI 3D-CNN preprocessing summary")
    print(f"Processed samples: {len(success_rows)}")
    print(f"Failed samples: {len(failed)}")
    if failed:
        print(f"Failed sample IDs: {failed}")
    print(f"metadata.csv: {metadata_csv.relative_to(PROJECT_ROOT)}")
    print(f"details.csv: {details_csv.relative_to(PROJECT_ROOT)}")
    print(f"Array output dir: {ARRAYS_DIR.relative_to(PROJECT_ROOT)}")
    print("Label distribution:", dict(Counter(row["label"] for row in success_rows)))
    return metadata_csv
