import os
import sys


DEFAULT_GPU_ID = "0"
DEFAULT_DEVICE = "cuda"
DEFAULT_DATASET = "dataset/ADNI_test20_release"
DEFAULT_CONFIG = "configs/rootstrap_adni_finetune_data_aug_seed3.yaml"


def dataset_name_from_path(path):
    name = os.path.basename(os.path.normpath(path))
    for suffix in [".tar.gz", ".tgz", ".tar"]:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return os.path.splitext(name)[0]


def parse_args(argv):
    if len(argv) == 0:
        dataset_path = DEFAULT_DATASET
    elif len(argv) == 2 and argv[0] == "--dataset":
        dataset_path = argv[1]
    else:
        raise SystemExit("Usage: python eval2.py")
    return {
        "dataset": dataset_path,
        "dataset_name": dataset_name_from_path(dataset_path),
        "config": DEFAULT_CONFIG,
        "device": DEFAULT_DEVICE,
    }


def main():
    args = parse_args(sys.argv[1:])
    if DEFAULT_GPU_ID not in {None, "", "none", "None"}:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(DEFAULT_GPU_ID)
    os.environ.setdefault("MPLCONFIGDIR", "/tmp")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

    from src.eval_rootstrap import run_rootstrap_eval
    from src.utils import resolve_device, set_seed

    set_seed(42)
    device = resolve_device(str(args.get("device", "cuda")))
    run_rootstrap_eval(args, device)


if __name__ == "__main__":
    main()
