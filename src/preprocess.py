import csv
import contextlib
import io
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UKB_RAW_DIR = PROJECT_ROOT / "dataset" / "UKB_T1_100cases"
UKB_PROCESSED_DIR = PROJECT_ROOT / "dataset" / "processed_brainiac" / "UKB"
UKB_IMAGES_DIR = UKB_PROCESSED_DIR / "images"

BRAINIAC_PREPROCESS_DIR = PROJECT_ROOT / "BrainIAC" / "src" / "preprocessing"
BRAINIAC_TEMPLATE = BRAINIAC_PREPROCESS_DIR / "atlases" / "temp_head.nii.gz"

REGISTRATION_ENABLED = True
SKULL_STRIP_ENABLED = True
TARGET_SPACING = (1.0, 1.0, 1.0)
TARGET_SIZE = (96, 96, 96)


def clean_value(value):
    text = str(value).strip()
    try:
        number = float(text)
    except ValueError:
        return text
    if number.is_integer():
        return str(int(number))
    return str(number)


def read_csv_rows(csv_path):
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames or [], list(reader)


def find_ukb_csv(raw_dir):
    csv_files = sorted(raw_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV found under {raw_dir}. Re-extract dataset/UKB_T1_100cases.tar.gz "
            "so selected_100_age_sex.csv is present."
        )
    if len(csv_files) > 1:
        print(f"WARNING: multiple CSV files found, using {csv_files[0].relative_to(PROJECT_ROOT)}")
    return csv_files[0]


def pick_column(columns, candidates):
    normalized = {c.lower().replace("_", "").replace("-", ""): c for c in columns}
    for candidate in candidates:
        key = candidate.lower().replace("_", "").replace("-", "")
        if key in normalized:
            return normalized[key]
    raise ValueError(f"Could not find any of {candidates} in CSV columns: {columns}")


def case_dirs(raw_dir):
    return sorted([p for p in raw_dir.iterdir() if p.is_dir()], key=lambda p: p.name)


def find_input_image(case_dir):
    nifti_files = sorted(case_dir.glob("*.nii.gz")) + sorted(case_dir.glob("*.nii"))
    if nifti_files:
        return nifti_files[0], "nifti"

    files = [p for p in case_dir.rglob("*") if p.is_file()]
    if files:
        return case_dir, "dicom"
    return None, "missing"


def convert_dicom_to_nifti(case_dir, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    series_reader = sitk.ImageSeriesReader()
    series_ids = series_reader.GetGDCMSeriesIDs(str(case_dir))
    if series_ids:
        file_names = series_reader.GetGDCMSeriesFileNames(str(case_dir), series_ids[0])
        series_reader.SetFileNames(file_names)
        image = series_reader.Execute()
        sitk.WriteImage(image, str(output_path))
        return output_path

    if shutil.which("dcm2niix") is None:
        raise RuntimeError(
            f"{case_dir.name}: DICOM-like input found, but SimpleITK found no DICOM series "
            "and dcm2niix is not available."
        )

    tmp_dir = output_path.parent / f".{case_dir.name}_dcm2niix"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["dcm2niix", "-z", "y", "-f", case_dir.name, "-o", str(tmp_dir), str(case_dir)],
        check=True,
    )
    converted = sorted(tmp_dir.glob("*.nii.gz"))
    if not converted:
        raise RuntimeError(f"{case_dir.name}: dcm2niix finished but no .nii.gz was created.")
    shutil.move(str(converted[0]), output_path)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return output_path


def read_input_as_image(input_path, input_kind, temp_dir, case_id):
    if input_kind == "nifti":
        return sitk.ReadImage(str(input_path), sitk.sitkFloat32)
    if input_kind == "dicom":
        nifti_path = Path(temp_dir) / f"{case_id}_dicom.nii.gz"
        return sitk.ReadImage(str(convert_dicom_to_nifti(input_path, nifti_path)), sitk.sitkFloat32)
    raise RuntimeError(f"Unsupported input kind: {input_kind}")


def n4_bias_field_correction(image):
    image = sitk.Cast(image, sitk.sitkFloat32)
    mask = sitk.OtsuThreshold(image, 0, 1, 200)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([50, 50, 30, 20])

    shrink_factors = [2, 2, 2]
    small_image = sitk.Shrink(image, shrink_factors)
    small_mask = sitk.Shrink(mask, shrink_factors)
    corrector.Execute(small_image, small_mask)
    log_bias = corrector.GetLogBiasFieldAsImage(image)
    corrected = image / sitk.Exp(log_bias)
    corrected.CopyInformation(image)
    return sitk.Cast(corrected, sitk.sitkFloat32)


def resample_to_spacing(image, spacing=TARGET_SPACING, interpolator=sitk.sitkLinear):
    image = sitk.Cast(image, sitk.sitkFloat32)
    old_size = image.GetSize()
    old_spacing = image.GetSpacing()
    new_size = [
        max(1, int(round(old_size[i] * old_spacing[i] / spacing[i])))
        for i in range(3)
    ]
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(spacing)
    resample.SetSize(new_size)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetInterpolator(interpolator)
    resample.SetDefaultPixelValue(0.0)
    resample.SetOutputPixelType(sitk.sitkFloat32)
    return resample.Execute(image)


def make_registration_reference():
    fixed = sitk.ReadImage(str(BRAINIAC_TEMPLATE), sitk.sitkFloat32)
    return resample_to_spacing(fixed, spacing=TARGET_SPACING)


def rigid_register_to_template(image, fixed):
    transform = sitk.CenteredTransformInitializer(
        fixed,
        image,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    method = sitk.ImageRegistrationMethod()
    method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    method.SetMetricSamplingStrategy(method.RANDOM)
    method.SetMetricSamplingPercentage(0.01)
    method.SetInterpolator(sitk.sitkLinear)
    method.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=100,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10,
    )
    method.SetOptimizerScalesFromPhysicalShift()
    method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    method.SetInitialTransform(transform, inPlace=False)
    final_transform = method.Execute(fixed, image)
    return sitk.Resample(image, fixed, final_transform, sitk.sitkLinear, 0.0, sitk.sitkFloat32)


def get_hd_bet_runner():
    if not SKULL_STRIP_ENABLED:
        raise RuntimeError("HD-BET skull stripping is disabled.")
    if not BRAINIAC_PREPROCESS_DIR.exists():
        raise FileNotFoundError(f"Missing BrainIAC preprocessing directory: {BRAINIAC_PREPROCESS_DIR}")
    sys.path.insert(0, str(BRAINIAC_PREPROCESS_DIR))
    try:
        from HD_BET.hd_bet import hd_bet
    except Exception as exc:
        raise RuntimeError(f"BrainIAC HD-BET import failed: {exc}") from exc
    return hd_bet


def hd_bet_device():
    return os.environ.get("BRAINIAC_HDBET_DEVICE", "0")


def run_hd_bet(image, temp_dir, case_id, hd_bet_runner):
    input_path = Path(temp_dir) / f"{case_id}_0000.nii.gz"
    output_path = Path(temp_dir) / f"{case_id}_brain.nii.gz"
    sitk.WriteImage(image, str(input_path))
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        hd_bet_runner(
            str(input_path),
            str(output_path),
            mode="fast",
            device=hd_bet_device(),
            tta=0,
            pp=1,
            save_mask=0,
            overwrite_existing=1,
        )
    if not output_path.exists():
        raise RuntimeError("HD-BET finished but did not create output image.")
    return sitk.ReadImage(str(output_path), sitk.sitkFloat32)


def skull_strip(image, temp_dir, case_id, hd_bet_runner):
    if hd_bet_runner is None:
        raise RuntimeError("HD-BET runner is unavailable.")
    return run_hd_bet(image, temp_dir, case_id, hd_bet_runner), "hd_bet"


def center_crop_or_pad(image, target_size=TARGET_SIZE):
    size = image.GetSize()
    lower_crop = [max(0, (size[i] - target_size[i]) // 2) for i in range(3)]
    upper_crop = [max(0, size[i] - target_size[i] - lower_crop[i]) for i in range(3)]
    if any(lower_crop) or any(upper_crop):
        image = sitk.Crop(image, lower_crop, upper_crop)

    size = image.GetSize()
    lower_pad = [max(0, (target_size[i] - size[i]) // 2) for i in range(3)]
    upper_pad = [max(0, target_size[i] - size[i] - lower_pad[i]) for i in range(3)]
    if any(lower_pad) or any(upper_pad):
        image = sitk.ConstantPad(image, lower_pad, upper_pad, 0.0)
    return sitk.Cast(image, sitk.sitkFloat32)


def z_normalize_nonzero(image):
    array = sitk.GetArrayFromImage(image).astype(np.float32)
    mask = array != 0
    if np.any(mask):
        mean = float(array[mask].mean())
        std = float(array[mask].std())
        if std > 1e-6:
            array[mask] = (array[mask] - mean) / std
        else:
            array[mask] = array[mask] - mean
    normalized = sitk.GetImageFromArray(array)
    normalized.CopyInformation(image)
    return sitk.Cast(normalized, sitk.sitkFloat32)


def preprocess_to_brainiac(input_path, input_kind, output_path, case_id, fixed_reference, hd_bet_runner):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    notes = []
    with tempfile.TemporaryDirectory(prefix=f"brainiac_{case_id}_", dir=str(UKB_PROCESSED_DIR)) as temp_dir:
        image = read_input_as_image(input_path, input_kind, temp_dir, case_id)
        image = n4_bias_field_correction(image)
        image = resample_to_spacing(image, spacing=TARGET_SPACING)

        if REGISTRATION_ENABLED:
            if fixed_reference is None:
                notes.append("registration_skipped_missing_template")
            else:
                try:
                    image = rigid_register_to_template(image, fixed_reference)
                except Exception as exc:
                    notes.append(f"registration_failed_skipped:{exc}")
                    print(f"WARNING: {case_id}: registration failed, continuing without MNI registration: {exc}")
        else:
            notes.append("registration_disabled")

        image, skull_status = skull_strip(image, temp_dir, case_id, hd_bet_runner)
        notes.append(skull_status)
        image = center_crop_or_pad(image, target_size=TARGET_SIZE)
        image = z_normalize_nonzero(image)
        image.SetSpacing(TARGET_SPACING)
        sitk.WriteImage(image, str(output_path))
    return ";".join(notes)


def write_metadata(metadata):
    metadata_csv = UKB_PROCESSED_DIR / "metadata.csv"
    with metadata_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["ID", "image_path", "Age", "Sex", "preprocessing_status"])
        writer.writeheader()
        writer.writerows(metadata)
    return metadata_csv


def validate_outputs(metadata):
    success_rows = [row for row in metadata if row["preprocessing_status"] == "hd_bet"]
    missing = [row["ID"] for row in success_rows if not (PROJECT_ROOT / row["image_path"]).exists()]
    image_count = len(list(UKB_IMAGES_DIR.glob("*.nii.gz")))
    return missing, image_count


def fail_status(reason):
    text = " ".join(str(reason).split())
    return f"fail: {text[:300]}"


def build_ukb_metadata():
    if not UKB_RAW_DIR.exists():
        raise FileNotFoundError(f"Missing UKB directory: {UKB_RAW_DIR}")

    csv_path = find_ukb_csv(UKB_RAW_DIR)
    columns, rows = read_csv_rows(csv_path)
    print("UKB BrainIAC preprocessing")
    print(f"Raw data: {UKB_RAW_DIR.relative_to(PROJECT_ROOT)}")
    print(f"CSV: {csv_path.relative_to(PROJECT_ROOT)}")
    print(f"Processed dir: {UKB_PROCESSED_DIR.relative_to(PROJECT_ROOT)}")
    print(f"Images: {UKB_IMAGES_DIR.relative_to(PROJECT_ROOT)}")
    print(f"metadata.csv: {(UKB_PROCESSED_DIR / 'metadata.csv').relative_to(PROJECT_ROOT)}")

    id_col = pick_column(columns, ["ID", "eid", "caseid", "case_id", "subject"])
    age_col = pick_column(columns, ["Age", "age"])
    sex_col = pick_column(columns, ["Sex", "sex", "gender"])

    row_by_id = {clean_value(row[id_col]): row for row in rows}
    dirs = case_dirs(UKB_RAW_DIR)
    print(f"Input cases: {len(dirs)}")

    UKB_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    if UKB_IMAGES_DIR.exists():
        shutil.rmtree(UKB_IMAGES_DIR)
    UKB_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    fixed_reference = None
    if REGISTRATION_ENABLED:
        if BRAINIAC_TEMPLATE.exists():
            fixed_reference = make_registration_reference()
            print(f"Registration: enabled, template={BRAINIAC_TEMPLATE.relative_to(PROJECT_ROOT)}")
        else:
            print("WARNING: registration enabled but BrainIAC template is missing; registration will be skipped.")

    hd_bet_runner = get_hd_bet_runner()
    print(f"HD-BET: enabled, device={hd_bet_device()}")

    metadata = []
    failed = []
    for case_dir in tqdm(dirs, desc="Preprocessing UKB", unit="case"):
        case_id = clean_value(case_dir.name)
        row = row_by_id.get(case_id)
        if row is None:
            failed.append((case_id, "missing CSV row"))
            metadata.append(
                {
                    "ID": case_id,
                    "image_path": "",
                    "Age": "",
                    "Sex": "",
                    "preprocessing_status": fail_status("missing CSV row"),
                }
            )
            continue

        input_path, input_kind = find_input_image(case_dir)
        if input_path is None:
            failed.append((case_id, "missing image"))
            metadata.append(
                {
                    "ID": case_id,
                    "image_path": "",
                    "Age": clean_value(row[age_col]),
                    "Sex": clean_value(row[sex_col]),
                    "preprocessing_status": fail_status("missing image"),
                }
            )
            continue

        output_path = UKB_IMAGES_DIR / f"{case_id}.nii.gz"
        try:
            status = preprocess_to_brainiac(input_path, input_kind, output_path, case_id, fixed_reference, hd_bet_runner)
        except Exception as exc:
            failed.append((case_id, str(exc)))
            metadata.append(
                {
                    "ID": case_id,
                    "image_path": "",
                    "Age": clean_value(row[age_col]),
                    "Sex": clean_value(row[sex_col]),
                    "preprocessing_status": fail_status(exc),
                }
            )
            continue

        metadata.append(
            {
                "ID": case_id,
                "image_path": str(output_path.relative_to(PROJECT_ROOT)),
                "Age": clean_value(row[age_col]),
                "Sex": clean_value(row[sex_col]),
                "preprocessing_status": status,
            }
        )

    metadata_csv = write_metadata(metadata)

    success_rows = [row for row in metadata if row["preprocessing_status"] == "hd_bet"]
    ages = [float(row["Age"]) for row in success_rows]
    sex_counts = Counter(row["Sex"] for row in success_rows)

    print("\nUKB BrainIAC preprocessing summary")
    print(f"Processed samples: {len(metadata)}")
    print(f"Failed samples: {len(failed)}")
    if failed:
        print(f"Failed sample IDs: {[case_id for case_id, _ in failed]}")
    print(f"metadata.csv: {metadata_csv.relative_to(PROJECT_ROOT)}")
    print(f"Image output dir: {UKB_IMAGES_DIR.relative_to(PROJECT_ROOT)}")
    if ages:
        print(f"Age range: min={min(ages):.1f}, max={max(ages):.1f}, mean={sum(ages) / len(ages):.2f}")
    print("Sex distribution:", dict(sex_counts))
    missing, image_count = validate_outputs(metadata)
    print(f"metadata image_path missing count: {len(missing)}")
    print(f"processed image file count: {image_count}")
