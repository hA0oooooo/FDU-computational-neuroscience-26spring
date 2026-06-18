import sys
import warnings

from src.utils import configure_gpu_from_config, load_yaml, resolve_device, set_seed


def parse_config_path(argv):
    if len(argv) != 2 or argv[0] != "--config":
        raise SystemExit("Usage: python train.py --config configs/xxx.yaml")
    return argv[1]


def main():
    config_path = parse_config_path(sys.argv[1:])

    config = load_yaml(config_path)
    configure_gpu_from_config(config)
    warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API.*")

    set_seed(int(config["seed"]))
    device = resolve_device(str(config.get("device", "cuda")))
    dataset_name = str(config.get("dataset", "ukb")).lower()
    model_name = str(config.get("model", "brainiac")).lower()
    trainer = str(config.get("trainer", "dl")).lower()

    if dataset_name == "adni":
        if model_name == "brainiac":
            from src.train_adni import run_adni_dl_training, run_adni_sklearn_training

            if trainer == "sklearn":
                run_adni_sklearn_training(config, config_path, device)
            elif trainer == "dl":
                run_adni_dl_training(config, config_path, device)
            else:
                raise ValueError(f"Unknown ADNI trainer: {trainer}")
        elif model_name == "3dcnn":
            from src.train_3dcnn import run_3dcnn_dl_training, run_3dcnn_sklearn_training

            if trainer == "sklearn":
                run_3dcnn_sklearn_training(config, config_path, device)
            elif trainer == "dl":
                run_3dcnn_dl_training(config, config_path, device)
            else:
                raise ValueError(f"Unknown ADNI 3D-CNN trainer: {trainer}")
        elif model_name == "sfcn":
            from src.train_sfcn_adni import run_sfcn_adni_dl_training, run_sfcn_adni_sklearn_training

            if trainer == "sklearn":
                run_sfcn_adni_sklearn_training(config, config_path, device)
            elif trainer == "dl":
                run_sfcn_adni_dl_training(config, config_path, device)
            else:
                raise ValueError(f"Unknown ADNI SFCN trainer: {trainer}")
        elif model_name == "brainmvp":
            from src.train_brainmvp_adni import (
                run_brainmvp_adni_dl_training,
                run_brainmvp_adni_sklearn_training,
            )

            if trainer == "sklearn":
                run_brainmvp_adni_sklearn_training(config, config_path, device)
            elif trainer == "dl":
                run_brainmvp_adni_dl_training(config, config_path, device)
            else:
                raise ValueError(f"Unknown ADNI BrainMVP trainer: {trainer}")
        elif model_name == "rootstrap":
            from src.train_rootstrap_adni import run_rootstrap_baseline, run_rootstrap_finetune

            if trainer == "baseline":
                run_rootstrap_baseline(config, config_path, device)
            elif trainer == "dl":
                run_rootstrap_finetune(config, config_path, device)
            else:
                raise ValueError(f"Unknown ADNI Rootstrap trainer: {trainer}")
        else:
            raise ValueError(
                "ADNI currently supports only model=brainiac, model=3dcnn, model=sfcn, model=brainmvp, or model=rootstrap."
            )
    elif dataset_name != "ukb":
        raise NotImplementedError("Only UKB and ADNI are implemented.")
    elif model_name == "sfcn":
        from src.train_sfcn import run_sfcn_training

        run_sfcn_training(config, config_path, device)
    elif model_name == "brainiac" and trainer == "dl":
        from src.train_dl import run_dl_training

        run_dl_training(config, config_path, device)
    elif model_name == "brainiac" and trainer == "sklearn":
        from src.train_sklearn import run_sklearn_training

        run_sklearn_training(config, config_path, device)
    else:
        raise ValueError(f"Unknown model/trainer combination: model={model_name}, trainer={trainer}")


if __name__ == "__main__":
    main()
