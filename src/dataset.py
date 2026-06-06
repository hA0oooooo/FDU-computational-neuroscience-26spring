import csv
from pathlib import Path

import torch
from torch.utils.data import Dataset
from monai.transforms import Compose, EnsureChannelFirstd, LoadImaged, NormalizeIntensityd, Resized, ToTensord


def get_brainiac_transform(image_size=(96, 96, 96)):
    return Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Resized(keys=["image"], spatial_size=image_size, mode="trilinear"),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            ToTensord(keys=["image"]),
        ]
    )


def get_sfcn_transform(image_size=(160, 192, 160)):
    return Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Resized(keys=["image"], spatial_size=image_size, mode="trilinear"),
            ToTensord(keys=["image"]),
        ]
    )


def read_metadata(metadata_csv):
    with Path(metadata_csv).open("r", newline="") as f:
        return list(csv.DictReader(f))


def is_preprocessed(row):
    status = row.get("preprocessing_status", "")
    return bool(row.get("image_path")) and not status.startswith("fail")


def sort_labels(labels):
    def key(value):
        try:
            return (0, float(value))
        except ValueError:
            return (1, value)

    return sorted(set(str(v) for v in labels), key=key)


class UKBAgeSexDataset(Dataset):
    def __init__(
        self,
        metadata_csv,
        transform=None,
        sex_classes=None,
        age_standardize=False,
        age_mean=None,
        age_std=None,
    ):
        self.metadata_csv = Path(metadata_csv)
        raw_rows = read_metadata(self.metadata_csv)
        self.rows = [row for row in raw_rows if is_preprocessed(row)]
        if not self.rows:
            raise ValueError(f"No usable preprocessed rows found in metadata CSV: {metadata_csv}")

        self.transform = transform if transform is not None else get_brainiac_transform()
        self.sex_classes = sex_classes if sex_classes is not None else sort_labels([r["Sex"] for r in self.rows])
        self.sex_to_index = {label: idx for idx, label in enumerate(self.sex_classes)}
        self.age_standardize = age_standardize
        self.age_mean = float(age_mean) if age_mean is not None else 0.0
        self.age_std = float(age_std) if age_std is not None else 1.0
        if self.age_std <= 0:
            self.age_std = 1.0

    def __len__(self):
        return len(self.rows)

    def ages(self, indices=None):
        rows = self.rows if indices is None else [self.rows[i] for i in indices]
        return [float(row["Age"]) for row in rows]

    def age_stats(self, indices=None):
        ages = self.ages(indices)
        mean = sum(ages) / len(ages)
        variance = sum((age - mean) ** 2 for age in ages) / max(1, len(ages))
        std = variance ** 0.5
        return mean, std if std > 0 else 1.0

    def standardize_age(self, age):
        if not self.age_standardize:
            return age
        return (age - self.age_mean) / self.age_std

    def __getitem__(self, idx):
        row = self.rows[idx]
        sample = self.transform({"image": row["image_path"]})
        sex_label = str(row["Sex"])
        age_raw = float(row["Age"])
        return {
            "id": row["ID"],
            "image": sample["image"],
            "age": torch.tensor(self.standardize_age(age_raw), dtype=torch.float32),
            "age_raw": torch.tensor(age_raw, dtype=torch.float32),
            "sex": torch.tensor(self.sex_to_index[sex_label], dtype=torch.long),
            "sex_label": sex_label,
        }
