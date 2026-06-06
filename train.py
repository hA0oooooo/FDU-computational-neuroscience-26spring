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

    if config["dataset"].lower() != "ukb":
        raise NotImplementedError("Only UKB is implemented in the current stage.")

    set_seed(int(config["seed"]))
    device = resolve_device(str(config.get("device", "cuda")))
    model_name = str(config.get("model", "brainiac")).lower()
    trainer = str(config.get("trainer", "dl")).lower()

    if model_name == "sfcn":
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
