import tarfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.dataset import read_metadata
from src.models_rootstrap import ROOTSTRAP_LABELS, RootstrapDenseNet
from src.preprocess_common import PROJECT_ROOT
from src.preprocess_rootstrap import build_adni_rootstrap_metadata
from src.train_rootstrap_adni import rootstrap_transform
from src.utils import load_yaml, write_csv


DEFAULT_CONFIG = "configs/rootstrap_adni_finetune_data_aug_seed3.yaml"


class RootstrapInferenceDataset(Dataset):
    def __init__(self, metadata_csv, transform):
        rows = read_metadata(metadata_csv)
        self.rows = [row for row in rows if row.get("image_path") and not row.get("preprocessing_status", "").startswith("fail")]
        if not self.rows:
            raise ValueError(f"No usable image rows found in metadata CSV: {metadata_csv}")
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image_path = Path(row["image_path"])
        if not image_path.is_absolute():
            image_path = PROJECT_ROOT / image_path
        sample = self.transform({"image": str(image_path)})
        return {"id": row["ID"], "image": sample["image"]}


def locate_raw_root(path):
    path = Path(path).resolve()
    if any(path.glob("*.csv")):
        return path
    children = [p for p in path.iterdir() if p.is_dir()]
    if len(children) == 1 and any(children[0].glob("*.csv")):
        return children[0]
    return path


def prepare_input_metadata(input_path, dataset_name):
    input_path = Path(input_path)
    if input_path.is_file():
        extract_dir = PROJECT_ROOT / "dataset" / dataset_name
        extract_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(input_path, "r:*") as tar:
            tar.extractall(extract_dir)
        raw_dir = locate_raw_root(extract_dir)
    else:
        raw_dir = locate_raw_root(input_path)
    processed_dir = PROJECT_ROOT / "dataset" / "processed_rootstrap" / dataset_name
    return build_adni_rootstrap_metadata(raw_dir, processed_dir, require_labels=False, dataset_name=dataset_name)


def checkpoint_dir(config_path, config):
    return PROJECT_ROOT / str(config.get("output_dir", "outputs")) / Path(config_path).stem


def config_seeds(config):
    seeds = config.get("seeds") or [int(config.get("seed", 42))]
    return [int(seed) for seed in seeds]


def fold_checkpoint_path(config_path, config, seed, fold):
    base = checkpoint_dir(config_path, config)
    seeded = base / f"seed_{seed}_fold_{fold}.pt"
    if seeded.exists():
        return seeded
    return base / f"fold_{fold}.pt"


def load_fold_model(config_path, config, seed, fold, device):
    path = fold_checkpoint_path(config_path, config, seed, fold)
    if not path.exists():
        raise FileNotFoundError(f"Missing Rootstrap fold checkpoint: {path}")
    model = RootstrapDenseNet(
        config["checkpoint_path"],
        load_pretrained=False,
        dropout=float(config.get("dropout", 0)),
    ).to(device)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def predict(config_path, config, dataset, device):
    loader = DataLoader(dataset, batch_size=int(config.get("batch_size", 1)), shuffle=False, num_workers=int(config.get("num_workers", 2)))
    num_folds = int(config.get("num_folds", 5))
    seeds = config_seeds(config)
    logit_sums = {row["ID"]: np.zeros(len(ROOTSTRAP_LABELS), dtype=np.float64) for row in dataset.rows}
    with torch.no_grad():
        for seed in seeds:
            for fold in range(num_folds):
                model = load_fold_model(config_path, config, seed, fold, device)
                for batch in loader:
                    logits = model(batch["image"].to(device)).detach().cpu().numpy()
                    for case_id, values in zip(batch["id"], logits):
                        logit_sums[str(case_id)] += values
    return {case_id: ROOTSTRAP_LABELS[int(values.argmax())] for case_id, values in logit_sums.items()}


def run_rootstrap_eval(args, device):
    metadata_csv = prepare_input_metadata(args["dataset"], args["dataset_name"])
    config_path = args.get("config") or DEFAULT_CONFIG
    config = load_yaml(config_path)
    transform = rootstrap_transform(tuple(config.get("rootstrap_input_shape", [96, 96, 96])), train=False)
    dataset = RootstrapInferenceDataset(metadata_csv, transform)
    pred = predict(config_path, config, dataset, device)
    rows = [{"ID": row["ID"], "Pre": pred[row["ID"]]} for row in dataset.rows]

    output_dir = PROJECT_ROOT / "outputs" / args["dataset_name"]
    pred_csv = output_dir / "pred.csv"
    write_csv(pred_csv, sorted(rows, key=lambda row: row["ID"]), fieldnames=["ID", "Pre"])
    print(f"Saved prediction CSV: {pred_csv.relative_to(PROJECT_ROOT)}")
