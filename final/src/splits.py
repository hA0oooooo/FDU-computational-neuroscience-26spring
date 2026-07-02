import csv
import random
from dataclasses import dataclass
from pathlib import Path


MODALITY_FILENAMES = {
    "T1": "T1.nii.gz",
    "T2_FLAIR": "T2_FLAIR.nii.gz",
}


@dataclass(frozen=True)
class CaseRecord:
    caseid: str
    data_root: Path

    def path(self, modality):
        if modality not in MODALITY_FILENAMES:
            raise ValueError(f"Unknown modality: {modality}")
        return self.data_root / self.caseid / MODALITY_FILENAMES[modality]


def read_manifest(data_root, manifest_path):
    data_root = Path(data_root)
    manifest_path = Path(manifest_path)
    records = []
    with manifest_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(CaseRecord(caseid=str(row["caseid"]), data_root=data_root))
    if len(records) == 0:
        raise ValueError(f"No cases found in manifest: {manifest_path}")
    return records


def check_required_files(records):
    missing = []
    for record in records:
        for modality in MODALITY_FILENAMES:
            path = record.path(modality)
            if not path.exists():
                missing.append(str(path))
    return missing


def _shuffled(records, seed):
    records = list(records)
    rng = random.Random(int(seed))
    rng.shuffle(records)
    return records


def make_splits(records, split_cfg):
    mode = str(split_cfg.get("mode", "seed42")).lower()
    if mode != "seed42":
        raise ValueError("Only seed42 split is supported.")
    seed = int(split_cfg.get("seed", 42))
    shuffled = _shuffled(records, seed)
    n_cases = len(shuffled)

    train_count = int(split_cfg.get("train_count", 480))
    val_count = int(split_cfg.get("val_count", 60))
    test_count = int(split_cfg.get("test_count", n_cases - train_count - val_count))
    if train_count + val_count + test_count != n_cases:
        raise ValueError(
            f"seed42 split counts must sum to {n_cases}; got "
            f"{train_count}+{val_count}+{test_count}."
        )
    train = shuffled[:train_count]
    val = shuffled[train_count : train_count + val_count]
    test = shuffled[train_count + val_count :]
    return {"train": train, "val": val, "test": test}


def read_split_csv(path, records):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")
    by_caseid = {record.caseid: record for record in records}
    splits = {"train": [], "val": [], "test": []}
    seen = set()
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            split_name = str(row["split"])
            caseid = str(row["caseid"])
            if split_name not in splits:
                raise ValueError(f"Unknown split={split_name!r} in {path}")
            if caseid not in by_caseid:
                raise ValueError(f"Split file references unknown caseid={caseid!r}: {path}")
            if caseid in seen:
                raise ValueError(f"Duplicate caseid={caseid!r} in split file: {path}")
            splits[split_name].append(by_caseid[caseid])
            seen.add(caseid)
    return splits


def caseids(records):
    return [record.caseid for record in records]
