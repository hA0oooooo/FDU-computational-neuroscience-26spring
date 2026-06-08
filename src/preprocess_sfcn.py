import csv
import os
import shutil
import subprocess
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

from src.preprocess_common import (
    PROJECT_ROOT,
    TARGET_SPACING_1MM,
    UKB_RAW_DIR,
    case_dirs,
    clean_value,
    find_input_image,
    find_ukb_csv,
    n4_bias_field_correction,
    pick_column,
    read_csv_rows,
    read_input_as_image,
    resample_to_spacing,
    shape_text,
)


SFCN_PROCESSED_DIR = PROJECT_ROOT / "dataset" / "processed_sfcn" / "UKB"
SFCN_IMAGES_DIR = SFCN_PROCESSED_DIR / "images"
SFCN_ADNI_PROCESSED_DIR = PROJECT_ROOT / "dataset" / "processed_sfcn" / "ADNI"
ADNI_RAW_CANDIDATES = [
    PROJECT_ROOT / "dataset" / "ADNI_data_105cases",
    PROJECT_ROOT / "dataset" / "ADNI_data",
]
SFCN_TARGET_SHAPE = (160, 192, 160)
SFCN_NUM_WORKERS = 16
SFCN_NORMALIZATION_SOURCE = "UKBiobank_deep_pretrain/examples.ipynb"
SFCN_NORMALIZATION_METHOD = "sfcn_official_divide_by_mean"
STANDARD_MNI_NAMES = {
    "T1_brain_linearto_MNI.nii.gz",
    "T1_brain_to_MNI.nii.gz",
    "T1_unbiased_brain_linearto_MNI.nii.gz",
}
ADNI_LABELS = {"CN", "MCI", "AD"}


def fail_status(reason):
    return " ".join(str(reason).split())[:500]


def path_text(path):
    path = Path(path)
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def fsl_install_message():
    return (
        "SFCN preprocessing requires FSL flirt, bet, FSLDIR, and "
        "$FSLDIR/data/standard/MNI152_T1_1mm_brain.nii.gz.\n\n"
        "Try installing the conda FSL packages manually:\n"
        "  conda activate cn\n"
        "  conda install -y \\\n"
        "    -c https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/public/ \\\n"
        "    -c conda-forge \\\n"
        "    fsl-flirt fsl-bet2 fsl-data_standard\n\n"
        "If conda FSL is unavailable, install full FSL and export:\n"
        "  export FSLDIR=<FSL安装路径>\n"
        "  source $FSLDIR/etc/fslconf/fsl.sh\n"
        "  export PATH=$FSLDIR/bin:$PATH"
    )


def check_fsl_dependencies():
    flirt = shutil.which("flirt")
    bet = shutil.which("bet")
    fsldir = os.environ.get("FSLDIR")
    missing = []
    if not flirt:
        missing.append("flirt")
    if not bet:
        missing.append("bet")
    if not fsldir:
        missing.append("FSLDIR")
        mni = None
    else:
        mni = Path(fsldir) / "data" / "standard" / "MNI152_T1_1mm_brain.nii.gz"
        if not mni.exists():
            missing.append(str(mni))
    if missing:
        raise RuntimeError("Missing FSL dependency: " + ", ".join(missing) + "\n\n" + fsl_install_message())
    return Path(flirt), Path(bet), mni


def find_existing_mni_file(case_dir):
    by_name = {p.name: p for p in case_dir.rglob("*.nii.gz")}
    for name in STANDARD_MNI_NAMES:
        if name in by_name:
            return by_name[name]
    return None


def find_adni_raw_dir():
    for raw_dir in ADNI_RAW_CANDIDATES:
        if raw_dir.exists():
            return raw_dir
    raise FileNotFoundError("Missing ADNI directory. Extract dataset/ADNI_data_105cases.tar.gz first.")


def find_adni_csv(raw_dir):
    csv_files = sorted(Path(raw_dir).glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV found under {raw_dir}")
    return csv_files[0]


def pick_adni_label_column(columns, rows):
    try:
        return pick_column(columns, ["label", "diagnosis", "dx", "group", "class"])
    except ValueError:
        for column in columns:
            values = {str(row[column]).strip().upper() for row in rows if str(row[column]).strip()}
            if values and values <= (ADNI_LABELS | {"DEMENTIA"}):
                return column
    raise ValueError(f"Could not infer ADNI label column from CSV columns: {columns}")


def normalize_adni_label(value):
    text = str(value).strip().upper()
    if text == "DEMENTIA":
        return "AD"
    if text in ADNI_LABELS:
        return text
    raise ValueError(f"Unsupported ADNI label: {value}")


def run_command(cmd, env):
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    if result.returncode != 0:
        stderr = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(stderr or f"command failed with code {result.returncode}: {' '.join(map(str, cmd))}")


def image_shape(path):
    return tuple(int(v) for v in nib.load(str(path)).shape[:3])


def save_sitk(image, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(sitk.Cast(image, sitk.sitkFloat32), str(path))
    return path


def center_crop_pad_array(data, target_shape):
    result = np.zeros(target_shape, dtype=np.float32)
    source_slices = []
    target_slices = []
    crop_start = []
    pad_low = []
    for axis, target in enumerate(target_shape):
        size = data.shape[axis]
        if size >= target:
            start = (size - target) // 2
            source_slices.append(slice(start, start + target))
            target_slices.append(slice(0, target))
            crop_start.append(start)
            pad_low.append(0)
        else:
            start = (target - size) // 2
            source_slices.append(slice(0, size))
            target_slices.append(slice(start, start + size))
            crop_start.append(0)
            pad_low.append(start)
    result[tuple(target_slices)] = data[tuple(source_slices)]
    return result, crop_start, pad_low


def array_stats(data):
    return {
        "min": float(data.min()),
        "max": float(data.max()),
        "mean": float(data.mean()),
        "std": float(data.std()),
    }


def sfcn_official_scale(data):
    data = data.astype(np.float32, copy=True)
    mean = float(data.mean())
    if abs(mean) <= 1e-6:
        raise RuntimeError("SFCN official divide-by-mean normalization failed because image mean is too small")
    return data / mean


def crop_pad_normalize_to_final(input_path, output_path):
    img = nib.load(str(input_path))
    data = np.asarray(img.get_fdata(dtype=np.float32), dtype=np.float32)
    raw_stats = array_stats(data)
    scaled = sfcn_official_scale(data)
    cropped, crop_start, pad_low = center_crop_pad_array(scaled, SFCN_TARGET_SHAPE)
    affine = img.affine.copy()
    voxel_offset = np.asarray(crop_start, dtype=float) - np.asarray(pad_low, dtype=float)
    affine[:3, 3] = (img.affine @ np.array([voxel_offset[0], voxel_offset[1], voxel_offset[2], 1.0]))[:3]
    normalized_stats = array_stats(cropped)
    final_img = nib.Nifti1Image(cropped, affine, img.header)
    final_img.set_data_dtype(np.float32)
    final_img.header.set_zooms(TARGET_SPACING_1MM)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    nib.save(final_img, str(output_path))
    verified = nib.load(str(output_path))
    if tuple(verified.shape[:3]) != SFCN_TARGET_SHAPE:
        raise RuntimeError(f"final shape is {verified.shape[:3]}, expected {SFCN_TARGET_SHAPE}")
    return {
        "normalization_method": SFCN_NORMALIZATION_METHOD,
        "raw_min": raw_stats["min"],
        "raw_max": raw_stats["max"],
        "raw_mean": raw_stats["mean"],
        "raw_std": raw_stats["std"],
        "normalized_min": normalized_stats["min"],
        "normalized_max": normalized_stats["max"],
        "normalized_mean": normalized_stats["mean"],
        "normalized_std": normalized_stats["std"],
    }


def final_detail_values(path):
    img = nib.load(str(path))
    data = np.asarray(img.get_fdata(dtype=np.float32), dtype=np.float32)
    mask = np.abs(data) > 1e-6
    if np.any(mask):
        coords = np.argwhere(mask)
        bbox_min = tuple(int(v) for v in coords.min(axis=0))
        bbox_max = tuple(int(v) for v in coords.max(axis=0))
        margin_low = bbox_min
        margin_high = tuple(int(SFCN_TARGET_SHAPE[i] - 1 - bbox_max[i]) for i in range(3))
        nonzero_ratio = float(mask.mean())
        mean = float(data.mean())
        std = float(data.std())
    else:
        bbox_min = bbox_max = margin_low = margin_high = ("", "", "")
        nonzero_ratio = 0.0
        mean = float(data.mean())
        std = float(data.std())
    return {
        "final_shape": shape_text(img.shape[:3]),
        "final_spacing": shape_text(img.header.get_zooms()[:3]),
        "nonzero_ratio": nonzero_ratio,
        "bbox_min": shape_text(bbox_min),
        "bbox_max": shape_text(bbox_max),
        "margin_low": shape_text(margin_low),
        "margin_high": shape_text(margin_high),
        "min": float(data.min()),
        "max": float(data.max()),
        "mean": mean,
        "std": std,
    }


def preprocess_case(case_dir, row, age_col, sex_col, fsl, images_dir=SFCN_IMAGES_DIR, label_col=None):
    case_id = clean_value(case_dir.name)
    output_path = Path(images_dir) / f"{case_id}.nii.gz"
    details = {
        "ID": case_id,
        "source_file": "",
        "used_existing_mni_file": 0,
        "input_shape": "",
        "shape_after_n4": "",
        "shape_after_1mm": "",
        "shape_after_bet": "",
        "shape_after_flirt": "",
        "final_shape": "",
        "final_spacing": "",
        "nonzero_ratio": "",
        "bbox_min": "",
        "bbox_max": "",
        "margin_low": "",
        "margin_high": "",
        "normalization_method": "",
        "raw_min": "",
        "raw_max": "",
        "raw_mean": "",
        "raw_std": "",
        "normalized_min": "",
        "normalized_max": "",
        "normalized_mean": "",
        "normalized_std": "",
        "status": "failed",
        "error_message": "",
    }

    existing_mni = find_existing_mni_file(case_dir)
    if existing_mni is not None:
        details["source_file"] = path_text(existing_mni)
        details["used_existing_mni_file"] = 1
        details["input_shape"] = shape_text(image_shape(existing_mni))
        details.update(crop_pad_normalize_to_final(existing_mni, output_path))
    else:
        with tempfile.TemporaryDirectory(prefix=f"sfcn_{case_id}_") as temp_dir:
            work_dir = Path(temp_dir)
            input_path, input_kind = find_input_image(case_dir)
            if input_path is None:
                raise RuntimeError("missing readable NIfTI or DICOM input")
            details["source_file"] = path_text(input_path)
            image = read_input_as_image(input_path, input_kind, work_dir, case_id)
            details["input_shape"] = shape_text(image.GetSize())

            n4_path = work_dir / f"{case_id}_n4.nii.gz"
            image = n4_bias_field_correction(image)
            save_sitk(image, n4_path)
            details["shape_after_n4"] = shape_text(image_shape(n4_path))

            one_mm_path = work_dir / f"{case_id}_1mm.nii.gz"
            image = resample_to_spacing(image, spacing=TARGET_SPACING_1MM)
            save_sitk(image, one_mm_path)
            details["shape_after_1mm"] = shape_text(image_shape(one_mm_path))

            brain_path = work_dir / f"{case_id}_brain_1mm.nii.gz"
            run_command([fsl["bet"], one_mm_path, brain_path, "-R", "-f", "0.5", "-g", "0"], fsl["env"])
            if not brain_path.exists():
                raise RuntimeError("BET finished but output file is missing")
            details["shape_after_bet"] = shape_text(image_shape(brain_path))

            flirt_path = work_dir / f"{case_id}_brain_linearto_MNI.nii.gz"
            mat_path = work_dir / f"{case_id}_brain_to_MNI.mat"
            run_command(
                [
                    fsl["flirt"],
                    "-in",
                    brain_path,
                    "-ref",
                    fsl["mni"],
                    "-out",
                    flirt_path,
                    "-omat",
                    mat_path,
                    "-dof",
                    "12",
                    "-cost",
                    "corratio",
                    "-interp",
                    "trilinear",
                ],
                fsl["env"],
            )
            if not flirt_path.exists() or not mat_path.exists():
                raise RuntimeError("FLIRT finished but output image or matrix is missing")
            details["shape_after_flirt"] = shape_text(image_shape(flirt_path))
            details.update(crop_pad_normalize_to_final(flirt_path, output_path))

    values = final_detail_values(output_path)
    details.update({key: values[key] for key in details.keys() if key in values})
    if tuple(image_shape(output_path)) != SFCN_TARGET_SHAPE:
        raise RuntimeError(f"final shape is {image_shape(output_path)}, expected {SFCN_TARGET_SHAPE}")
    details["status"] = "success"
    if label_col:
        metadata = {
            "ID": case_id,
            "image_path": str(output_path.relative_to(PROJECT_ROOT)),
            "label": normalize_adni_label(row[label_col]) if row is not None else "",
        }
    else:
        metadata = {
            "ID": case_id,
            "image_path": str(output_path.relative_to(PROJECT_ROOT)),
            "Age": clean_value(row[age_col]) if row is not None and age_col else "",
            "Sex": clean_value(row[sex_col]) if row is not None and sex_col else "",
        }
    return metadata, details


def preprocess_case_worker(payload):
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(int(payload.get("itk_threads", 1)))
    case_dir = Path(payload["case_dir"])
    case_id = clean_value(case_dir.name)
    row = payload["row"]
    age_col = payload["age_col"]
    sex_col = payload["sex_col"]
    label_col = payload.get("label_col")
    require_labels = bool(payload.get("require_labels", True))

    def failed_metadata(reason):
        if label_col:
            return {
                "ID": case_id,
                "image_path": "",
                "label": normalize_adni_label(row[label_col]) if row is not None else "",
                "preprocessing_status": f"fail: {fail_status(reason)}",
            }
        return {
            "ID": case_id,
            "image_path": "",
            "Age": clean_value(row[age_col]) if row is not None and age_col else "",
            "Sex": clean_value(row[sex_col]) if row is not None and sex_col else "",
            "preprocessing_status": f"fail: {fail_status(reason)}",
        }

    if row is None and require_labels:
        reason = "missing CSV row"
        return {
            "metadata": failed_metadata(reason),
            "details": {"ID": case_id, "status": "failed", "error_message": reason},
        }
    try:
        metadata_row, details_row = preprocess_case(
            case_dir,
            row,
            age_col,
            sex_col,
            payload["fsl"],
            Path(payload["images_dir"]),
            label_col=label_col,
        )
        metadata_row["preprocessing_status"] = "success"
        return {"metadata": metadata_row, "details": details_row}
    except Exception as exc:
        reason = fail_status(exc)
        return {
            "metadata": failed_metadata(reason),
            "details": {"ID": case_id, "status": "failed", "error_message": reason},
        }


def write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def prepare_output_dirs(processed_dir=SFCN_PROCESSED_DIR):
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    (processed_dir / "images").mkdir(parents=True, exist_ok=True)


def fsl_context():
    flirt, bet, mni = check_fsl_dependencies()
    env = os.environ.copy()
    env["FSLOUTPUTTYPE"] = "NIFTI_GZ"
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "1")
    return {"flirt": str(flirt), "bet": str(bet), "mni": str(mni), "env": env}


def build_sfcn_metadata(raw_dir, processed_dir, require_labels=True, dataset_name="UKB", label_task=False):
    if not (PROJECT_ROOT / "UKBiobank_deep_pretrain").exists():
        raise FileNotFoundError(
            "Missing UKBiobank_deep_pretrain/. Please run:\n"
            "  git clone --depth=1 https://github.com/ha-ha-ha-han/UKBiobank_deep_pretrain.git"
        )
    raw_dir = Path(raw_dir).resolve()
    processed_dir = Path(processed_dir).resolve()
    images_dir = processed_dir / "images"
    if not raw_dir.exists():
        raise FileNotFoundError(f"Missing raw image directory: {raw_dir}")
    fsl = fsl_context()

    csv_path = None
    row_by_id = {}
    age_col = None
    sex_col = None
    label_col = None
    csv_files = sorted(raw_dir.glob("*.csv"))
    if csv_files:
        csv_path = csv_files[0]
        columns, rows = read_csv_rows(csv_path)
        id_col = pick_column(columns, ["ID", "eid", "caseid", "case_id", "subject"])
        if label_task:
            label_col = pick_adni_label_column(columns, rows)
        elif require_labels:
            age_col = pick_column(columns, ["Age", "age"])
            sex_col = pick_column(columns, ["Sex", "sex", "gender"])
        else:
            try:
                age_col = pick_column(columns, ["Age", "age"])
            except ValueError:
                age_col = None
            try:
                sex_col = pick_column(columns, ["Sex", "sex", "gender"])
            except ValueError:
                sex_col = None
        row_by_id = {clean_value(row[id_col]): row for row in rows}
    elif require_labels:
        raise FileNotFoundError(f"No CSV found under {raw_dir}")
    dirs = case_dirs(raw_dir)

    print(f"{dataset_name} SFCN preprocessing")
    print(f"Raw data: {path_text(raw_dir)}")
    if csv_path:
        print(f"CSV: {path_text(csv_path)}")
    print(f"Processed dir: {path_text(processed_dir)}")
    print(f"MNI reference: {Path(fsl['mni'])}")
    print(f"Workers: {SFCN_NUM_WORKERS}")

    prepare_output_dirs(processed_dir)

    tasks = [
        {
            "case_dir": str(case_dir),
            "row": row_by_id.get(clean_value(case_dir.name)),
            "age_col": age_col,
            "sex_col": sex_col,
            "label_col": label_col,
            "fsl": fsl,
            "images_dir": str(images_dir),
            "require_labels": require_labels,
            "itk_threads": 1,
        }
        for case_dir in dirs
    ]

    metadata = []
    details_rows = []
    if SFCN_NUM_WORKERS == 1:
        iterator = (preprocess_case_worker(task) for task in tasks)
        for result in tqdm(iterator, total=len(tasks), desc=f"Preprocessing SFCN {dataset_name}", unit="case"):
            if result["metadata"] is not None:
                metadata.append(result["metadata"])
            details_rows.append(result["details"])
    else:
        with ProcessPoolExecutor(max_workers=int(SFCN_NUM_WORKERS)) as executor:
            futures = [executor.submit(preprocess_case_worker, task) for task in tasks]
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Preprocessing SFCN {dataset_name}", unit="case"):
                result = future.result()
                if result["metadata"] is not None:
                    metadata.append(result["metadata"])
                details_rows.append(result["details"])

    metadata = sorted(metadata, key=lambda row: row["ID"])
    details_rows = sorted(details_rows, key=lambda row: row.get("ID", ""))

    metadata_fields = ["ID", "image_path", "label", "preprocessing_status"] if label_task else ["ID", "image_path", "Age", "Sex", "preprocessing_status"]
    metadata_csv = write_csv(processed_dir / "metadata.csv", metadata, metadata_fields)
    details_csv = write_csv(
        processed_dir / "details.csv",
        details_rows,
        [
            "ID",
            "source_file",
            "used_existing_mni_file",
            "input_shape",
            "shape_after_n4",
            "shape_after_1mm",
            "shape_after_bet",
            "shape_after_flirt",
            "final_shape",
            "final_spacing",
            "nonzero_ratio",
            "bbox_min",
            "bbox_max",
            "margin_low",
            "margin_high",
            "normalization_method",
            "raw_min",
            "raw_max",
            "raw_mean",
            "raw_std",
            "normalized_min",
            "normalized_max",
            "normalized_mean",
            "normalized_std",
            "status",
            "error_message",
        ],
    )
    successful = [row for row in metadata if row.get("image_path")]
    failed = [row for row in metadata if not row.get("image_path")]
    ages = [float(row["Age"]) for row in successful if row.get("Age") not in {"", None}] if not label_task else []
    sex_counts = Counter(row["Sex"] for row in successful) if not label_task else Counter()
    label_counts = Counter(row["label"] for row in successful) if label_task else Counter()
    shape_counts = Counter(row["final_shape"] for row in details_rows if row.get("status") == "success")

    print(f"\n{dataset_name} SFCN preprocessing summary")
    print(f"SFCN official normalization source file: {SFCN_NORMALIZATION_SOURCE}")
    print(f"normalization_method: {SFCN_NORMALIZATION_METHOD}")
    print(f"processed image count: {len(successful)}")
    print(f"failed case count: {len(failed)}")
    print(f"metadata.csv: {metadata_csv.relative_to(PROJECT_ROOT)}")
    print(f"details.csv: {details_csv.relative_to(PROJECT_ROOT)}")
    print("final shape statistics:", dict(shape_counts))
    if ages:
        print(f"Age range: min={min(ages):.1f}, max={max(ages):.1f}, mean={sum(ages) / len(ages):.2f}")
    if label_task:
        print("Label distribution:", dict(label_counts))
    else:
        print("Sex distribution:", dict(sex_counts))

    if successful:
        first_id = successful[0]["ID"]
        first_details = next(row for row in details_rows if row.get("ID") == first_id)
        first = PROJECT_ROOT / successful[0]["image_path"]
        values = final_detail_values(first)
        print("\nFirst processed sample")
        print(f"shape: {values['final_shape']}")
        print(f"spacing: {values['final_spacing']}")
        print(
            "before normalization min/max/mean/std: "
            f"{first_details['raw_min']}, {first_details['raw_max']}, "
            f"{first_details['raw_mean']}, {first_details['raw_std']}"
        )
        print(
            "after normalization min/max/mean/std: "
            f"{first_details['normalized_min']}, {first_details['normalized_max']}, "
            f"{first_details['normalized_mean']}, {first_details['normalized_std']}"
        )
        print(f"nonzero_ratio: {values['nonzero_ratio']}")

    return metadata_csv


def build_ukb_sfcn_metadata():
    return build_sfcn_metadata(UKB_RAW_DIR, SFCN_PROCESSED_DIR, require_labels=True, dataset_name="UKB", label_task=False)


def build_adni_sfcn_metadata():
    return build_sfcn_metadata(find_adni_raw_dir(), SFCN_ADNI_PROCESSED_DIR, require_labels=True, dataset_name="ADNI", label_task=True)
