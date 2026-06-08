import csv
import tempfile
from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np
from monai.transforms import (
    CenterSpatialCropd,
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    Resized,
    ScaleIntensityRangePercentilesd,
    Spacingd,
)
from tqdm import tqdm

from src.preprocess_common import (
    PROJECT_ROOT,
    case_dirs,
    clean_value,
    convert_dicom_to_nifti,
    find_input_image,
    pick_column,
    read_csv_rows,
    shape_text,
)


ADNI_RAW_CANDIDATES = [
    PROJECT_ROOT / "dataset" / "ADNI_data_105cases",
    PROJECT_ROOT / "dataset" / "ADNI_data",
]
PROCESSED_DIR = PROJECT_ROOT / "dataset" / "processed_brainmvp" / "ADNI"
IMAGES_DIR = PROCESSED_DIR / "images"
RESIZE_SHAPE = (128, 128, 64)
TARGET_SHAPE = (96, 96, 64)
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


def fail_status(reason):
    return f"fail: {' '.join(str(reason).split())[:300]}"


def stats(data):
    return {
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "mean": float(np.mean(data)),
        "std": float(np.std(data)),
    }


def brainmvp_transform():
    return Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
            CropForegroundd(keys=["image"], source_key="image", margin=1),
            ScaleIntensityRangePercentilesd(
                keys=["image"],
                lower=5,
                upper=95,
                b_min=0.0,
                b_max=1.0,
                clip=True,
                channel_wise=True,
            ),
            Resized(keys=["image"], spatial_size=RESIZE_SHAPE, mode="trilinear"),
            CenterSpatialCropd(keys=["image"], roi_size=TARGET_SHAPE),
        ]
    )


def source_path_for_case(raw_dir, row, id_col):
    relative_path = row.get("relative_path")
    if relative_path:
        path = raw_dir / relative_path
        if path.exists():
            return path, "nifti"

    case_dir = raw_dir / clean_value(row[id_col])
    input_path, input_kind = find_input_image(case_dir)
    return input_path, input_kind


def preprocess_one(case_id, source_path, input_kind, transform):
    with tempfile.TemporaryDirectory(prefix=f"brainmvp_{case_id}_", dir=str(PROCESSED_DIR)) as work_dir:
        if input_kind == "dicom":
            nifti_path = Path(work_dir) / f"{case_id}_dicom.nii.gz"
            source_path = convert_dicom_to_nifti(source_path, nifti_path)

        raw_img = nib.load(str(source_path))
        raw_shape = raw_img.shape[:3]
        sample = transform({"image": str(source_path)})
        data = sample["image"]
        if hasattr(data, "detach"):
            data = data.detach().cpu().numpy()
        data = np.asarray(data, dtype=np.float32)
        if data.shape[0] != 1:
            raise RuntimeError(f"expected one channel after preprocessing, got {data.shape}")
        data = data[0]
        if tuple(data.shape) != TARGET_SHAPE:
            raise RuntimeError(f"final shape is {tuple(data.shape)}, expected {TARGET_SHAPE}")
        return raw_shape, data


def build_adni_brainmvp_metadata():
    raw_dir = find_adni_raw_dir()
    csv_path = find_adni_csv(raw_dir)
    columns, rows = read_csv_rows(csv_path)
    id_col = pick_column(columns, ["ID", "eid", "subject", "subject_id", "caseid", "case_id"])
    label_col = pick_label_column(columns, rows)
    label_counts = Counter(normalize_label(row[label_col]) for row in rows)

    print("ADNI BrainMVP preprocessing")
    print(f"Raw data: {raw_dir.relative_to(PROJECT_ROOT)}")
    print(f"CSV: {csv_path.relative_to(PROJECT_ROOT)}")
    print(f"Processed dir: {PROCESSED_DIR.relative_to(PROJECT_ROOT)}")
    print(f"Images: {IMAGES_DIR.relative_to(PROJECT_ROOT)}")
    print(f"CSV samples: {len(rows)}")
    print("Label distribution:", dict(label_counts))

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    transform = brainmvp_transform()
    metadata = []
    details = []
    failed = []

    for row in tqdm(rows, desc="Preprocessing ADNI BrainMVP", unit="case"):
        case_id = clean_value(row[id_col])
        label = normalize_label(row[label_col])
        detail = {
            "ID": case_id,
            "source_file": "",
            "input_kind": "",
            "input_shape": "",
            "final_shape": "",
            "preprocessing_pipeline": "ras_1mm_cropforeground_percentile5_95_resize128_center_crop96",
            "normalization_method": "brainmvp_downstream_percentile_5_95_to_0_1",
            "skull_strip_status": "assume_input_brain_or_foreground_crop_only",
            "final_min": "",
            "final_max": "",
            "final_mean": "",
            "final_std": "",
            "preprocessing_status": "",
            "error_message": "",
        }
        try:
            source_path, input_kind = source_path_for_case(raw_dir, row, id_col)
            if source_path is None:
                raise RuntimeError("missing image")
            raw_shape, data = preprocess_one(case_id, source_path, input_kind, transform)
            output_path = IMAGES_DIR / f"{case_id}.nii.gz"
            nib.save(nib.Nifti1Image(data.astype(np.float32), np.eye(4)), str(output_path))
            data_stats = stats(data)
            detail.update(
                {
                    "source_file": str(Path(source_path).relative_to(PROJECT_ROOT)) if Path(source_path).exists() else str(source_path),
                    "input_kind": input_kind,
                    "input_shape": shape_text(raw_shape),
                    "final_shape": shape_text(data.shape),
                    "final_min": data_stats["min"],
                    "final_max": data_stats["max"],
                    "final_mean": data_stats["mean"],
                    "final_std": data_stats["std"],
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
        writer = csv.DictWriter(f, fieldnames=list(details[0].keys()) if details else ["ID"])
        writer.writeheader()
        writer.writerows(details)

    success_rows = [row for row in metadata if row["image_path"] and not row["preprocessing_status"].startswith("fail")]
    print("\nADNI BrainMVP preprocessing summary")
    print(f"Processed samples: {len(success_rows)}")
    print(f"Failed samples: {len(failed)}")
    if failed:
        print(f"Failed sample IDs: {failed}")
    print(f"metadata.csv: {metadata_csv.relative_to(PROJECT_ROOT)}")
    print(f"details.csv: {details_csv.relative_to(PROJECT_ROOT)}")
    print(f"Image output dir: {IMAGES_DIR.relative_to(PROJECT_ROOT)}")
    print("Label distribution:", dict(Counter(row["label"] for row in success_rows)))
    return metadata_csv
