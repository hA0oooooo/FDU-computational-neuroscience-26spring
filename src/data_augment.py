import math

from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    RandAdjustContrastd,
    RandAffined,
    RandFlipd,
    Resized,
    ToTensord,
)


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
            ToTensord(keys=["image"]),
        ]
    )
