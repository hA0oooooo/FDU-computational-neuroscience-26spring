import random
from collections import OrderedDict
from math import ceil

from .io import load_nifti_dhw

try:
    from torch.utils.data import Dataset
except ImportError:
    class Dataset:  # pragma: no cover - real training requires torch.
        pass


def _pad_to_min_shape(arr, min_shape):
    import numpy as np

    pad_width = []
    for size, target in zip(arr.shape, min_shape):
        pad_width.append((0, max(0, int(target) - int(size))))
    if any(right > 0 for _, right in pad_width):
        arr = np.pad(arr, pad_width, mode="constant")
    return arr


def _uniform_start(shape, patch_size):
    starts = []
    for size, patch in zip(shape, patch_size):
        max_start = max(0, int(size) - int(patch))
        starts.append(random.randint(0, max_start) if max_start > 0 else 0)
    return tuple(starts)


def _foreground_start(source_raw, target_raw, patch_size):
    import numpy as np

    mask = (np.abs(source_raw) > 0) | (np.abs(target_raw) > 0)
    axes = [
        np.where(mask.any(axis=(1, 2)))[0],
        np.where(mask.any(axis=(0, 2)))[0],
        np.where(mask.any(axis=(0, 1)))[0],
    ]
    if any(len(axis) == 0 for axis in axes):
        return _uniform_start(source_raw.shape, patch_size)

    starts = []
    for axis, size, patch in zip(axes, source_raw.shape, patch_size):
        center = random.randint(int(axis[0]), int(axis[-1]))
        max_start = max(0, int(size) - int(patch))
        offset = random.randint(0, int(patch) - 1)
        starts.append(min(max(center - offset, 0), max_start))
    return tuple(starts)


def _deterministic_start(shape, patch_size, item_index, seed):
    rng = random.Random(int(seed) + int(item_index))
    starts = []
    for size, patch in zip(shape, patch_size):
        max_start = max(0, int(size) - int(patch))
        if max_start == 0:
            starts.append(0)
        elif item_index % 2 == 0:
            starts.append(max_start // 2)
        else:
            starts.append(rng.randint(0, max_start))
    return tuple(starts)


def _crop(arr, start, patch_size):
    d, h, w = start
    pd, ph, pw = patch_size
    return arr[d : d + pd, h : h + ph, w : w + pw]


def _pad_hw_to_min_shape(arr, crop_size):
    import numpy as np

    crop_h, crop_w = (int(x) for x in crop_size)
    pad_width = [(0, 0), (0, max(0, crop_h - int(arr.shape[1]))), (0, max(0, crop_w - int(arr.shape[2])))]
    if any(right > 0 for _, right in pad_width):
        arr = np.pad(arr, pad_width, mode="constant")
    return arr


def _deterministic_slice(depth, local_index, samples_per_case):
    depth = int(depth)
    local_index = int(local_index)
    samples_per_case = int(samples_per_case)
    if samples_per_case <= 1:
        return depth // 2
    return int(round(local_index * (depth - 1) / (samples_per_case - 1)))


def _uniform_crop_start_2d(shape_hw, crop_size):
    starts = []
    for size, crop in zip(shape_hw, crop_size):
        max_start = max(0, int(size) - int(crop))
        starts.append(random.randint(0, max_start) if max_start > 0 else 0)
    return tuple(starts)


def _foreground_crop_start_2d(source_slice, target_slice, crop_size):
    import numpy as np

    mask = (np.abs(source_slice) > 0) | (np.abs(target_slice) > 0)
    coords = np.argwhere(mask)
    if coords.size == 0:
        return _uniform_crop_start_2d(source_slice.shape, crop_size)
    center_h, center_w = coords[random.randint(0, len(coords) - 1)]
    starts = []
    for center, size, crop in zip((center_h, center_w), source_slice.shape, crop_size):
        max_start = max(0, int(size) - int(crop))
        offset = random.randint(0, int(crop) - 1)
        starts.append(min(max(int(center) - offset, 0), max_start))
    return tuple(starts)


def _deterministic_crop_start_2d(shape_hw, crop_size, item_index, seed):
    rng = random.Random(int(seed) + int(item_index))
    starts = []
    for size, crop in zip(shape_hw, crop_size):
        max_start = max(0, int(size) - int(crop))
        if max_start == 0:
            starts.append(0)
        elif item_index % 2 == 0:
            starts.append(max_start // 2)
        else:
            starts.append(rng.randint(0, max_start))
    return tuple(starts)


def _slice_stack(volume, z, offsets):
    import numpy as np

    d = int(volume.shape[0])
    indices = [min(max(int(z) + int(offset), 0), d - 1) for offset in offsets]
    return np.stack([volume[idx] for idx in indices], axis=0)


def _crop_2d(arr, start_hw, crop_size):
    h, w = start_hw
    ch, cw = crop_size
    return arr[..., h : h + ch, w : w + cw]


def _items_per_case(dataset):
    if hasattr(dataset, "samples_per_case"):
        return int(dataset.samples_per_case)
    if hasattr(dataset, "patches_per_case"):
        return int(dataset.patches_per_case)
    return None


class CaseGroupedSampler:
    def __init__(self, dataset, shuffle=True, seed=42, distributed=False, rank=0, world_size=1, drop_last=False):
        self.dataset = dataset
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.distributed = bool(distributed)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.drop_last = bool(drop_last)
        self.epoch = 0
        self.items_per_case = _items_per_case(dataset)
        if self.items_per_case is None:
            raise ValueError("CaseGroupedSampler requires samples_per_case or patches_per_case.")

    def __len__(self):
        case_count = len(self.dataset.records)
        if self.distributed:
            if self.drop_last:
                case_count = case_count - (case_count % self.world_size)
            else:
                case_count = int(ceil(case_count / self.world_size) * self.world_size)
            case_count = case_count // self.world_size
        return case_count * self.items_per_case

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        case_count = len(self.dataset.records)
        cases = list(range(case_count))
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(cases)

        if self.distributed:
            if self.drop_last:
                keep = case_count - (case_count % self.world_size)
                cases = cases[:keep]
            else:
                total = int(ceil(case_count / self.world_size) * self.world_size)
                cases += cases[: total - case_count]
            cases = cases[self.rank : len(cases) : self.world_size]

        for case_idx in cases:
            start = int(case_idx) * self.items_per_case
            for item_idx in range(self.items_per_case):
                yield start + item_idx


class PairedPatchDataset(Dataset):
    def __init__(
        self,
        records,
        source_modality,
        target_modality,
        normalizer,
        patch_size,
        patches_per_case=8,
        foreground_prob=0.7,
        deterministic=False,
        seed=42,
        cache_size=0,
    ):
        self.records = list(records)
        self.source_modality = source_modality
        self.target_modality = target_modality
        self.normalizer = normalizer
        self.patch_size = tuple(int(x) for x in patch_size)
        self.patches_per_case = int(patches_per_case)
        self.foreground_prob = float(foreground_prob)
        self.deterministic = bool(deterministic)
        self.seed = int(seed)
        self.cache_size = int(cache_size)
        self._cache = OrderedDict()

    def __len__(self):
        return len(self.records) * self.patches_per_case

    def _load_pair(self, record):
        if self.cache_size > 0 and record.caseid in self._cache:
            pair = self._cache.pop(record.caseid)
            self._cache[record.caseid] = pair
            return pair

        source_raw, _ = load_nifti_dhw(record.path(self.source_modality))
        target_raw, _ = load_nifti_dhw(record.path(self.target_modality))
        if source_raw.shape != target_raw.shape:
            raise ValueError(
                f"Shape mismatch for case {record.caseid}: "
                f"{self.source_modality}={source_raw.shape}, {self.target_modality}={target_raw.shape}"
            )

        pair = (source_raw, target_raw)
        if self.cache_size > 0:
            self._cache[record.caseid] = pair
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return pair

    def __getitem__(self, index):
        import torch

        record = self.records[index // self.patches_per_case]
        source_raw, target_raw = self._load_pair(record)

        source_raw = _pad_to_min_shape(source_raw, self.patch_size)
        target_raw = _pad_to_min_shape(target_raw, self.patch_size)
        if self.deterministic:
            start = _deterministic_start(source_raw.shape, self.patch_size, index, self.seed)
        elif random.random() < self.foreground_prob:
            start = _foreground_start(source_raw, target_raw, self.patch_size)
        else:
            start = _uniform_start(source_raw.shape, self.patch_size)

        source = self.normalizer.normalize(_crop(source_raw, start, self.patch_size), self.source_modality)
        target = self.normalizer.normalize(_crop(target_raw, start, self.patch_size), self.target_modality)
        source = torch.from_numpy(source[None])
        target = torch.from_numpy(target[None])
        return {
            "source": source,
            "target": target,
            "caseid": record.caseid,
            "start": torch.tensor(start, dtype=torch.long),
        }


class PairedSliceDataset(Dataset):
    def __init__(
        self,
        records,
        source_modality,
        target_modality,
        normalizer,
        crop_size,
        slice_offsets,
        samples_per_case=16,
        crop_foreground_prob=0.8,
        deterministic=False,
        seed=42,
        cache_size=0,
    ):
        self.records = list(records)
        self.source_modality = source_modality
        self.target_modality = target_modality
        self.normalizer = normalizer
        self.crop_size = tuple(int(x) for x in crop_size)
        self.slice_offsets = tuple(int(x) for x in slice_offsets)
        self.samples_per_case = int(samples_per_case)
        self.crop_foreground_prob = float(crop_foreground_prob)
        self.deterministic = bool(deterministic)
        self.seed = int(seed)
        self.cache_size = int(cache_size)
        self._cache = OrderedDict()

    def __len__(self):
        return len(self.records) * self.samples_per_case

    def _load_pair(self, record):
        if self.cache_size > 0 and record.caseid in self._cache:
            item = self._cache.pop(record.caseid)
            self._cache[record.caseid] = item
            return item

        source_raw, _ = load_nifti_dhw(record.path(self.source_modality))
        target_raw, _ = load_nifti_dhw(record.path(self.target_modality))
        if source_raw.shape != target_raw.shape:
            raise ValueError(
                f"Shape mismatch for case {record.caseid}: "
                f"{self.source_modality}={source_raw.shape}, {self.target_modality}={target_raw.shape}"
            )

        item = (source_raw, target_raw)
        if self.cache_size > 0:
            self._cache[record.caseid] = item
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return item

    def __getitem__(self, index):
        import torch

        record = self.records[index // self.samples_per_case]
        source_raw, target_raw = self._load_pair(record)

        source_raw = _pad_hw_to_min_shape(source_raw, self.crop_size)
        target_raw = _pad_hw_to_min_shape(target_raw, self.crop_size)
        local_index = int(index % self.samples_per_case)
        if self.deterministic:
            z = _deterministic_slice(source_raw.shape[0], local_index, self.samples_per_case)
            start_hw = _deterministic_crop_start_2d(source_raw.shape[1:], self.crop_size, index, self.seed)
        else:
            z = random.randint(0, int(source_raw.shape[0]) - 1)
            if random.random() < self.crop_foreground_prob:
                start_hw = _foreground_crop_start_2d(source_raw[z], target_raw[z], self.crop_size)
            else:
                start_hw = _uniform_crop_start_2d(source_raw.shape[1:], self.crop_size)

        source = _crop_2d(_slice_stack(source_raw, z, self.slice_offsets), start_hw, self.crop_size)
        target = _crop_2d(target_raw[z][None], start_hw, self.crop_size)
        source = self.normalizer.normalize(source, self.source_modality)
        target = self.normalizer.normalize(target, self.target_modality)
        return {
            "source": torch.from_numpy(source),
            "target": torch.from_numpy(target),
            "caseid": record.caseid,
            "z": torch.tensor(z, dtype=torch.long),
            "start": torch.tensor(start_hw, dtype=torch.long),
        }


def build_patch_loader(
    dataset,
    batch_size,
    shuffle,
    num_workers,
    distributed=False,
    seed=42,
    prefetch_factor=2,
    persistent_workers=True,
    drop_last=False,
    case_grouped=False,
):
    import torch
    import torch.distributed as dist
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler

    from .utils import seed_worker

    sampler = None
    if case_grouped:
        rank = dist.get_rank() if distributed and dist.is_available() and dist.is_initialized() else 0
        world_size = dist.get_world_size() if distributed and dist.is_available() and dist.is_initialized() else 1
        sampler = CaseGroupedSampler(
            dataset,
            shuffle=shuffle,
            seed=seed,
            distributed=distributed,
            rank=rank,
            world_size=world_size,
            drop_last=drop_last,
        )
        shuffle = False
    elif distributed:
        sampler = DistributedSampler(dataset, shuffle=shuffle, seed=int(seed))
        shuffle = False
    num_workers = int(num_workers)
    loader_kwargs = {
        "batch_size": int(batch_size),
        "shuffle": bool(shuffle),
        "sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "worker_init_fn": seed_worker,
        "drop_last": bool(drop_last),
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = int(prefetch_factor)
        loader_kwargs["persistent_workers"] = bool(persistent_workers)
    loader = DataLoader(dataset, **loader_kwargs)
    return loader, sampler
