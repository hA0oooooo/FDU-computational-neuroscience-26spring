import math

import torch
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    NormalizeIntensityd,
    RandAdjustContrastd,
    RandAffined,
    RandFlipd,
    Resized,
    ToTensord,
)
from torch.utils.data import Dataset


def augment_enabled(config):
    settings = config.get("data_augment", False)
    if isinstance(settings, bool):
        return settings
    return bool(settings.get("enabled", False))


def augment_multiplier(config):
    settings = config.get("data_augment", False)
    if not isinstance(settings, dict):
        return 1
    return max(1, int(settings.get("multiplier", 1)))


def expanded_indices(indices, multiplier):
    return list(indices) * max(1, int(multiplier))


def separate_augment_enabled(config):
    settings = config.get("data_augment", False)
    if not isinstance(settings, dict):
        return False
    return bool(settings.get("separate_transforms", False))


def separate_transform_names(config):
    settings = config.get("data_augment", {})
    if not isinstance(settings, dict):
        return []
    names = settings.get("separate_transform_names", ["zoom", "shift", "rotation"])
    names = [str(name).lower() for name in names]
    multiplier = max(1, int(settings.get("multiplier", len(names))))
    if multiplier <= len(names):
        return names[:multiplier]
    return [names[i % len(names)] for i in range(multiplier)]


def _sfcn_base_transforms(image_size):
    return [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Resized(keys=["image"], spatial_size=image_size, mode="trilinear"),
    ]


def get_sfcn_train_transform(config):
    image_size = tuple(config.get("sfcn_input_shape", [160, 192, 160]))
    settings = config.get("data_augment", {})
    if not isinstance(settings, dict):
        settings = {}

    rotate_degrees = float(settings.get("rotate_degrees", 8))
    rotate_radians = math.radians(rotate_degrees)
    shift_voxels = int(settings.get("shift_voxels", 5))
    scale_range = settings.get("scale_range", [0.95, 1.05])
    gamma_range = settings.get("gamma_range", [0.85, 1.15])
    scale_delta = max(abs(float(scale_range[0]) - 1.0), abs(float(scale_range[1]) - 1.0))

    return Compose(
        [
            *_sfcn_base_transforms(image_size),
            RandFlipd(keys=["image"], spatial_axis=0, prob=float(settings.get("flip_prob", 0.5))),
            RandAffined(
                keys=["image"],
                prob=float(settings.get("affine_prob", 0.5)),
                rotate_range=(rotate_radians, rotate_radians, rotate_radians),
                translate_range=(shift_voxels, shift_voxels, shift_voxels),
                scale_range=(scale_delta, scale_delta, scale_delta),
                mode="bilinear",
                padding_mode="zeros",
            ),
            RandAdjustContrastd(
                keys=["image"],
                prob=float(settings.get("contrast_prob", 0.2)),
                gamma=(float(gamma_range[0]), float(gamma_range[1])),
            ),
            ToTensord(keys=["image"]),
        ]
    )


def get_sfcn_separate_train_transforms(config):
    image_size = tuple(config.get("sfcn_input_shape", [160, 192, 160]))
    settings = config.get("data_augment", {})
    if not isinstance(settings, dict):
        settings = {}

    rotate_degrees = float(settings.get("rotate_degrees", 8))
    rotate_radians = math.radians(rotate_degrees)
    shift_voxels = int(settings.get("shift_voxels", 5))
    scale_range = settings.get("scale_range", [0.95, 1.05])
    scale_delta = max(abs(float(scale_range[0]) - 1.0), abs(float(scale_range[1]) - 1.0))

    transform_map = {
        "zoom": Compose(
            [
                *_sfcn_base_transforms(image_size),
                RandAffined(
                    keys=["image"],
                    prob=1.0,
                    rotate_range=(0.0, 0.0, 0.0),
                    translate_range=(0, 0, 0),
                    scale_range=(scale_delta, scale_delta, scale_delta),
                    mode="bilinear",
                    padding_mode="zeros",
                ),
                ToTensord(keys=["image"]),
            ]
        ),
        "shift": Compose(
            [
                *_sfcn_base_transforms(image_size),
                RandAffined(
                    keys=["image"],
                    prob=1.0,
                    rotate_range=(0.0, 0.0, 0.0),
                    translate_range=(shift_voxels, shift_voxels, shift_voxels),
                    scale_range=(0.0, 0.0, 0.0),
                    mode="bilinear",
                    padding_mode="zeros",
                ),
                ToTensord(keys=["image"]),
            ]
        ),
        "rotation": Compose(
            [
                *_sfcn_base_transforms(image_size),
                RandAffined(
                    keys=["image"],
                    prob=1.0,
                    rotate_range=(rotate_radians, rotate_radians, rotate_radians),
                    translate_range=(0, 0, 0),
                    scale_range=(0.0, 0.0, 0.0),
                    mode="bilinear",
                    padding_mode="zeros",
                ),
                ToTensord(keys=["image"]),
            ]
        ),
    }
    return [transform_map[name] for name in separate_transform_names(config)]


class SFCNSeparateAugmentDataset(Dataset):
    def __init__(self, base_dataset, indices, transforms):
        self.base_dataset = base_dataset
        self.items = [(index, transform) for index in indices for transform in transforms]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        row_index, transform = self.items[idx]
        row = self.base_dataset.rows[row_index]
        sample = transform({"image": row["image_path"]})
        sex_label = str(row["Sex"])
        age_raw = float(row["Age"])
        return {
            "id": row["ID"],
            "image": sample["image"],
            "age": torch.tensor(self.base_dataset.standardize_age(age_raw), dtype=torch.float32),
            "age_raw": torch.tensor(age_raw, dtype=torch.float32),
            "sex": torch.tensor(self.base_dataset.sex_to_index[sex_label], dtype=torch.long),
            "sex_label": sex_label,
        }


def make_sfcn_separate_augment_dataset(base_dataset, indices, config):
    return SFCNSeparateAugmentDataset(base_dataset, indices, get_sfcn_separate_train_transforms(config))


def get_brainiac_train_transform(config):
    image_size = tuple(config.get("image_size", [96, 96, 96]))
    settings = config.get("data_augment", {})
    if not isinstance(settings, dict):
        settings = {}

    rotate_degrees = float(settings.get("rotate_degrees", 8))
    rotate_radians = math.radians(rotate_degrees)
    shift_voxels = int(settings.get("shift_voxels", 3))
    scale_range = settings.get("scale_range", [0.95, 1.05])
    gamma_range = settings.get("gamma_range", [0.85, 1.15])
    scale_delta = max(abs(float(scale_range[0]) - 1.0), abs(float(scale_range[1]) - 1.0))

    return Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Resized(keys=["image"], spatial_size=image_size, mode="trilinear"),
            RandFlipd(keys=["image"], spatial_axis=0, prob=float(settings.get("flip_prob", 0.5))),
            RandAffined(
                keys=["image"],
                prob=float(settings.get("affine_prob", 0.5)),
                rotate_range=(rotate_radians, rotate_radians, rotate_radians),
                translate_range=(shift_voxels, shift_voxels, shift_voxels),
                scale_range=(scale_delta, scale_delta, scale_delta),
                mode="bilinear",
                padding_mode="zeros",
            ),
            RandAdjustContrastd(
                keys=["image"],
                prob=float(settings.get("contrast_prob", 0.2)),
                gamma=(float(gamma_range[0]), float(gamma_range[1])),
            ),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            ToTensord(keys=["image"]),
        ]
    )
