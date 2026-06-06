import tarfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.dataset import get_sfcn_transform, read_metadata
from src.models_sfcn import SFCNWrapper
from src.preprocess_common import PROJECT_ROOT
from src.preprocess_sfcn import build_sfcn_metadata
from src.utils import load_yaml, write_csv


DEFAULT_AGE_CONFIG = "configs/sfcn_ukb_age_finetune_data_aug.yaml"
DEFAULT_SEX_CONFIG = "configs/sfcn_ukb_sex_finetune.yaml"


class SFCNInferenceDataset(Dataset):
    def __init__(self, metadata_csv, transform):
        self.metadata_csv = Path(metadata_csv)
        rows = read_metadata(self.metadata_csv)
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
    processed_dir = PROJECT_ROOT / "dataset" / "processed_sfcn" / dataset_name
    return build_sfcn_metadata(raw_dir, processed_dir, require_labels=False)


def checkpoint_dir(config_path, config):
    return PROJECT_ROOT / str(config.get("output_dir", "outputs")) / Path(config_path).stem


def load_fold_model(config_path, config, task, fold, device):
    path = checkpoint_dir(config_path, config) / f"fold_{fold}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing {task} fold checkpoint: {path}")
    model = SFCNWrapper(
        sfcn_repo=config["sfcn_repo"],
        task=task,
        checkpoint_path=config.get("checkpoint_path"),
        dropout=bool(config.get("dropout", True)),
        load_pretrained=False,
    ).to(device)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def predict_age(config_path, config, dataset, device):
    loader = DataLoader(dataset, batch_size=int(config.get("batch_size", 1)), shuffle=False, num_workers=int(config.get("num_workers", 2)))
    num_folds = int(config.get("num_folds", 5))
    sums = {row["ID"]: 0.0 for row in dataset.rows}
    with torch.no_grad():
        for fold in range(num_folds):
            model = load_fold_model(config_path, config, "age", fold, device)
            for batch in loader:
                image = batch["image"].to(device)
                pred = model.forward_age(image).detach().cpu().numpy()
                for case_id, value in zip(batch["id"], pred):
                    sums[str(case_id)] += float(value)
    return {case_id: value / num_folds for case_id, value in sums.items()}


def predict_sex(config_path, config, dataset, device, sex_classes):
    loader = DataLoader(dataset, batch_size=int(config.get("batch_size", 1)), shuffle=False, num_workers=int(config.get("num_workers", 2)))
    num_folds = int(config.get("num_folds", 5))
    prob_sums = {row["ID"]: np.zeros(len(sex_classes), dtype=np.float64) for row in dataset.rows}
    with torch.no_grad():
        for fold in range(num_folds):
            model = load_fold_model(config_path, config, "sex", fold, device)
            for batch in loader:
                image = batch["image"].to(device)
                prob = torch.exp(model.forward_log_prob(image)).detach().cpu().numpy()
                for case_id, values in zip(batch["id"], prob):
                    prob_sums[str(case_id)] += values[: len(sex_classes)]
    return {case_id: sex_classes[int(values.argmax())] for case_id, values in prob_sums.items()}


def run_sfcn_eval(args, device):
    metadata_csv = prepare_input_metadata(args["dataset"], args["dataset_name"])
    age_config_path = args.get("age_config") or DEFAULT_AGE_CONFIG
    sex_config_path = args.get("sex_config") or DEFAULT_SEX_CONFIG
    age_config = load_yaml(age_config_path)
    sex_config = load_yaml(sex_config_path)
    transform = get_sfcn_transform(tuple(age_config.get("sfcn_input_shape", [160, 192, 160])))
    dataset = SFCNInferenceDataset(metadata_csv, transform)
    sex_classes = [item.strip() for item in str(args.get("sex_classes", "0,1")).split(",") if item.strip()]

    age_pred = predict_age(age_config_path, age_config, dataset, device)
    sex_pred = predict_sex(sex_config_path, sex_config, dataset, device, sex_classes)
    rows = [{"ID": row["ID"], "Age": age_pred[row["ID"]], "Sex": sex_pred[row["ID"]]} for row in dataset.rows]

    output_dir = PROJECT_ROOT / "outputs" / args["dataset_name"]
    pred_csv = output_dir / "pred.csv"
    write_csv(pred_csv, sorted(rows, key=lambda row: row["ID"]), fieldnames=["ID", "Age", "Sex"])
    print(f"Saved prediction CSV: {pred_csv.relative_to(PROJECT_ROOT)}")
