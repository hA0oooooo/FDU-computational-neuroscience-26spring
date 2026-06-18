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
    case_dirs,
    clean_value,
    find_input_image,
    pick_column,
    read_csv_rows,
    read_input_as_image,
    shape_text,
)


ADNI_RAW_CANDIDATES = [
    PROJECT_ROOT / "dataset" / "ADNI_data_105cases",
    PROJECT_ROOT / "dataset" / "ADNI_data",
]
PROCESSED_DIR = PROJECT_ROOT / "dataset" / "processed_rootstrap" / "ADNI"
IMAGES_DIR = PROCESSED_DIR / "images"
ROOTSTRAP_NUM_WORKERS = 16
ADNI_LABELS = {"CN", "MCI", "AD"}


def path_text(path):
    path = Path(path)
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def fail_status(reason):
    return " ".join(str(reason).split())[:500]


def find_adni_raw_dir():
    for raw_dir in ADNI_RAW_CANDIDATES:
        if raw_dir.exists():
            return raw_dir
    raise FileNotFoundError("Missing ADNI directory. Extract dataset/ADNI_data_105cases.tar.gz first.")


def find_adni_csv(raw_dir, require=True):
    csv_files = sorted(Path(raw_dir).glob("*.csv"))
    if not csv_files:
        if not require:
            return None
        raise FileNotFoundError(f"No CSV found under {raw_dir}")
    return csv_files[0]


def pick_label_column(columns, rows):
    try:
        return pick_column(columns, ["label", "diagnosis", "dx", "group", "class"])
    except ValueError:
        for column in columns:
            values = {str(row[column]).strip().upper() for row in rows if str(row[column]).strip()}
            if values and values <= (ADNI_LABELS | {"DEMENTIA"}):
                return column
    raise ValueError(f"Could not infer ADNI label column from CSV columns: {columns}")


def normalize_label(value):
    text = str(value).strip().upper()
    if text == "DEMENTIA":
        return "AD"
    if text in ADNI_LABELS:
        return text
    raise ValueError(f"Unsupported ADNI label: {value}")


def rootstrap_install_message():
    return (
        "Rootstrap preprocessing requires FSL fslreorient2std, flirt, bet, FSLDIR, and "
        "an MNI152_T1_1mm template.\n\n"
        "Install FSL packages manually if needed:\n"
        "  conda activate cn\n"
        "  conda install -y \\\n"
        "    -c https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/public/ \\\n"
        "    -c conda-forge \\\n"
        "    fsl-flirt fsl-bet2 fsl-data_standard\n\n"
        "Or install full FSL and export:\n"
        "  export FSLDIR=<FSL安装路径>\n"
        "  source $FSLDIR/etc/fslconf/fsl.sh\n"
        "  export PATH=$FSLDIR/bin:$PATH"
    )


def check_fsl_dependencies():
    fslreorient = shutil.which("fslreorient2std")
    flirt = shutil.which("flirt")
    bet = shutil.which("bet")
    fsldir = os.environ.get("FSLDIR")
    missing = []
    if not fslreorient:
        missing.append("fslreorient2std")
    if not flirt:
        missing.append("flirt")
    if not bet:
        missing.append("bet")
    mni = None
    if not fsldir:
        missing.append("FSLDIR")
    else:
        standard = Path(fsldir) / "data" / "standard"
        candidates = [
            standard / "MNI152_T1_1mm.nii.gz",
            standard / "MNI152_T1_1mm.nii",
        ]
        mni = next((path for path in candidates if path.exists()), None)
        if mni is None:
            missing.append(str(candidates[0]))
    if missing:
        raise RuntimeError("Missing FSL dependency: " + ", ".join(missing) + "\n\n" + rootstrap_install_message())
    return {
        "fslreorient": str(fslreorient),
        "flirt": str(flirt),
        "bet": str(bet),
        "mni": str(mni),
    }


def fsl_context():
    fsl = check_fsl_dependencies()
    env = os.environ.copy()
    env["FSLOUTPUTTYPE"] = "NIFTI_GZ"
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "1")
    fsl["env"] = env
    return fsl


def run_command(cmd, env):
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    if result.returncode != 0:
        stderr = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(stderr or f"command failed with code {result.returncode}: {' '.join(map(str, cmd))}")


def image_shape(path):
    return tuple(int(v) for v in nib.load(str(path)).shape[:3])


def image_spacing(path):
    return tuple(float(v) for v in nib.load(str(path)).header.get_zooms()[:3])


def array_stats(path):
    data = np.asarray(nib.load(str(path)).get_fdata(dtype=np.float32), dtype=np.float32)
    mask = np.abs(data) > 1e-6
    return {
        "min": float(data.min()),
        "max": float(data.max()),
        "mean": float(data.mean()),
        "std": float(data.std()),
        "nonzero_ratio": float(mask.mean()),
    }


def n4_bias_field_correction_rootstrap(image):
    image = sitk.Cast(image, sitk.sitkFloat32)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([100, 100, 60, 40])
    corrector.SetConvergenceThreshold(1e-4)
    corrector.SetSplineOrder(3)
    small = sitk.Shrink(image, [3, 3, 3])
    corrected_small = corrector.Execute(small)
    _ = corrected_small
    log_bias = corrector.GetLogBiasFieldAsImage(image)
    corrected = image / sitk.Exp(log_bias)
    corrected.CopyInformation(image)
    return sitk.Cast(corrected, sitk.sitkFloat32)


def save_sitk(image, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(sitk.Cast(image, sitk.sitkFloat32), str(path))
    return path


def preprocess_case(case_dir, row, label_col, fsl, images_dir):
    case_id = clean_value(case_dir.name)
    output_path = Path(images_dir) / f"{case_id}.nii.gz"
    details = {
        "ID": case_id,
        "source_file": "",
        "input_shape": "",
        "shape_after_reorient": "",
        "shape_after_flirt": "",
        "shape_after_bet": "",
        "shape_after_n4": "",
        "final_shape": "",
        "final_spacing": "",
        "nonzero_ratio": "",
        "normalization_method": "monai_scale_intensity_at_training_time",
        "registration": "fslreorient2std_flirt_mni152_t1_1mm_affine",
        "skull_stripping": "fsl_bet_R_f0.4_g0",
        "n4_method": "simpleitk_n4_after_bet",
        "min": "",
        "max": "",
        "mean": "",
        "std": "",
        "status": "failed",
        "error_message": "",
    }

    with tempfile.TemporaryDirectory(prefix=f"rootstrap_{case_id}_") as temp_dir:
        work_dir = Path(temp_dir)
        input_path, input_kind = find_input_image(case_dir)
        if input_path is None:
            raise RuntimeError("missing readable NIfTI or DICOM input")
        details["source_file"] = path_text(input_path)
        image = read_input_as_image(input_path, input_kind, work_dir, case_id)
        input_nii = work_dir / f"{case_id}_input.nii.gz"
        save_sitk(image, input_nii)
        details["input_shape"] = shape_text(image_shape(input_nii))

        reoriented = work_dir / f"{case_id}_reorient.nii.gz"
        run_command([fsl["fslreorient"], str(input_nii), str(reoriented)], fsl["env"])
        if not reoriented.exists():
            raise RuntimeError("fslreorient2std finished but output file is missing")
        details["shape_after_reorient"] = shape_text(image_shape(reoriented))

        registered = work_dir / f"{case_id}_mni_affine.nii.gz"
        mat_path = work_dir / f"{case_id}_to_mni.mat"
        run_command(
            [
                fsl["flirt"],
                "-in",
                str(reoriented),
                "-ref",
                fsl["mni"],
                "-out",
                str(registered),
                "-omat",
                str(mat_path),
                "-bins",
                "256",
                "-cost",
                "corratio",
                "-searchrx",
                "-90",
                "90",
                "-searchry",
                "-90",
                "90",
                "-searchrz",
                "-90",
                "90",
                "-dof",
                "12",
                "-interp",
                "spline",
            ],
            fsl["env"],
        )
        if not registered.exists() or not mat_path.exists():
            raise RuntimeError("FLIRT finished but output image or matrix is missing")
        details["shape_after_flirt"] = shape_text(image_shape(registered))

        brain = work_dir / f"{case_id}_brain.nii.gz"
        run_command([fsl["bet"], str(registered), str(brain), "-R", "-f", "0.4", "-g", "0"], fsl["env"])
        if not brain.exists():
            raise RuntimeError("BET finished but output file is missing")
        details["shape_after_bet"] = shape_text(image_shape(brain))

        corrected = n4_bias_field_correction_rootstrap(sitk.ReadImage(str(brain), sitk.sitkFloat32))
        save_sitk(corrected, output_path)

    if not output_path.exists():
        raise RuntimeError("N4 finished but final output is missing")
    details["shape_after_n4"] = shape_text(image_shape(output_path))
    details["final_shape"] = shape_text(image_shape(output_path))
    details["final_spacing"] = shape_text(image_spacing(output_path))
    details.update(array_stats(output_path))
    details["status"] = "success"
    metadata = {
        "ID": case_id,
        "image_path": str(output_path.relative_to(PROJECT_ROOT)),
        "label": normalize_label(row[label_col]) if row is not None and label_col else "",
        "preprocessing_status": "success",
    }
    return metadata, details


def preprocess_case_worker(payload):
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    case_dir = Path(payload["case_dir"])
    case_id = clean_value(case_dir.name)
    row = payload["row"]
    label_col = payload["label_col"]
    require_labels = bool(payload.get("require_labels", True))

    def failed_metadata(reason):
        return {
            "ID": case_id,
            "image_path": "",
            "label": normalize_label(row[label_col]) if row is not None and label_col else "",
            "preprocessing_status": f"fail: {fail_status(reason)}",
        }

    if row is None and require_labels:
        reason = "missing CSV row"
        return {
            "metadata": failed_metadata(reason),
            "details": {"ID": case_id, "status": "failed", "error_message": reason},
        }
    try:
        metadata, details = preprocess_case(
            case_dir,
            row,
            label_col,
            payload["fsl"],
            Path(payload["images_dir"]),
        )
        return {"metadata": metadata, "details": details}
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


def build_adni_rootstrap_metadata(raw_dir=None, processed_dir=None, require_labels=True, dataset_name="ADNI"):
    raw_dir = Path(raw_dir).resolve() if raw_dir is not None else find_adni_raw_dir().resolve()
    processed_dir = Path(processed_dir).resolve() if processed_dir is not None else PROCESSED_DIR.resolve()
    images_dir = processed_dir / "images"
    csv_path = find_adni_csv(raw_dir, require=require_labels)
    label_col = None
    row_by_id = {}
    if csv_path is not None and require_labels:
        columns, rows = read_csv_rows(csv_path)
        id_col = pick_column(columns, ["ID", "eid", "caseid", "case_id", "subject"])
        label_col = pick_label_column(columns, rows)
        row_by_id = {clean_value(row[id_col]): row for row in rows}
    dirs = case_dirs(raw_dir)
    fsl = fsl_context()

    processed_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    print(f"{dataset_name} Rootstrap preprocessing")
    print(f"Raw data: {path_text(raw_dir)}")
    if csv_path is not None:
        print(f"CSV: {path_text(csv_path)}")
    print(f"Processed dir: {path_text(processed_dir)}")
    print(f"MNI reference: {fsl['mni']}")
    print(f"Workers: {ROOTSTRAP_NUM_WORKERS}")

    tasks = [
        {
            "case_dir": str(case_dir),
            "row": row_by_id.get(clean_value(case_dir.name)),
            "label_col": label_col,
            "fsl": fsl,
            "images_dir": str(images_dir),
            "require_labels": require_labels,
        }
        for case_dir in dirs
    ]

    metadata = []
    details = []
    with ProcessPoolExecutor(max_workers=int(ROOTSTRAP_NUM_WORKERS)) as executor:
        futures = [executor.submit(preprocess_case_worker, task) for task in tasks]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Preprocessing ADNI Rootstrap", unit="case"):
            result = future.result()
            metadata.append(result["metadata"])
            details.append(result["details"])

    metadata = sorted(metadata, key=lambda row: row["ID"])
    details = sorted(details, key=lambda row: row.get("ID", ""))
    metadata_csv = write_csv(
        processed_dir / "metadata.csv",
        metadata,
        ["ID", "image_path", "label", "preprocessing_status"],
    )
    details_csv = write_csv(
        processed_dir / "details.csv",
        details,
        [
            "ID",
            "source_file",
            "input_shape",
            "shape_after_reorient",
            "shape_after_flirt",
            "shape_after_bet",
            "shape_after_n4",
            "final_shape",
            "final_spacing",
            "nonzero_ratio",
            "normalization_method",
            "registration",
            "skull_stripping",
            "n4_method",
            "min",
            "max",
            "mean",
            "std",
            "status",
            "error_message",
        ],
    )

    successful = [row for row in metadata if row.get("image_path")]
    failed = [row for row in metadata if not row.get("image_path")]
    print(f"\n{dataset_name} Rootstrap preprocessing summary")
    print(f"processed image count: {len(successful)}")
    print(f"failed case count: {len(failed)}")
    print(f"metadata.csv: {metadata_csv.relative_to(PROJECT_ROOT)}")
    print(f"details.csv: {details_csv.relative_to(PROJECT_ROOT)}")
    if require_labels:
        print("label distribution:", dict(Counter(row["label"] for row in successful)))
    print("final shape statistics:", dict(Counter(row.get("final_shape", "") for row in details if row.get("status") == "success")))
    return metadata_csv
