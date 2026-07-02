from pathlib import Path

from .io import load_nifti_xyz
from .utils import load_json, save_json


class Normalizer:
    def __init__(self, method, stats, eps=1e-8):
        self.method = str(method).lower()
        self.stats = stats
        self.eps = float(eps)
        if self.method not in {"none", "zscore", "minmax"}:
            raise ValueError("normalization.method must be one of: none, zscore, minmax.")

    def normalize(self, arr, modality):
        import numpy as np

        if self.method == "none":
            return arr.astype(np.float32, copy=False)
        item = self.stats[modality]
        if self.method == "zscore":
            return ((arr - item["mean"]) / max(item["std"], self.eps)).astype(np.float32, copy=False)
        denom = max(item["max"] - item["min"], self.eps)
        return ((arr - item["min"]) / denom).astype(np.float32, copy=False)

    def inverse(self, arr, modality):
        import numpy as np

        if self.method == "none":
            return arr.astype(np.float32, copy=False)
        item = self.stats[modality]
        if self.method == "zscore":
            return (arr * max(item["std"], self.eps) + item["mean"]).astype(np.float32, copy=False)
        return (arr * max(item["max"] - item["min"], self.eps) + item["min"]).astype(np.float32, copy=False)

    def to_dict(self):
        return {"method": self.method, "eps": self.eps, "stats": self.stats}

    @classmethod
    def from_dict(cls, obj):
        return cls(method=obj["method"], stats=obj.get("stats", {}), eps=obj.get("eps", 1e-8))


def compute_modality_stats(records, modality):
    import numpy as np

    count = 0
    total = 0.0
    total_sq = 0.0
    vmin = None
    vmax = None
    for record in records:
        arr, _ = load_nifti_xyz(record.path(modality))
        arr64 = arr.astype(np.float64, copy=False)
        count += arr64.size
        total += float(arr64.sum())
        total_sq += float((arr64 * arr64).sum())
        amin = float(arr64.min())
        amax = float(arr64.max())
        vmin = amin if vmin is None else min(vmin, amin)
        vmax = amax if vmax is None else max(vmax, amax)
    mean = total / max(count, 1)
    var = max(total_sq / max(count, 1) - mean * mean, 0.0)
    return {
        "count": int(count),
        "min": float(vmin),
        "max": float(vmax),
        "mean": float(mean),
        "std": float(var**0.5),
    }


def build_normalizer(records, modalities, cfg, cache_path=None):
    method = str(cfg.get("method", "zscore")).lower()
    eps = float(cfg.get("eps", 1e-8))
    cache_path = Path(cache_path) if cache_path else None
    if cache_path and cache_path.exists() and not bool(cfg.get("recompute", False)):
        return Normalizer.from_dict(load_json(cache_path))

    if method == "none":
        stats = {modality: {} for modality in modalities}
    else:
        stats = {modality: compute_modality_stats(records, modality) for modality in modalities}
    normalizer = Normalizer(method=method, stats=stats, eps=eps)
    if cache_path:
        save_json(cache_path, normalizer.to_dict())
    return normalizer

