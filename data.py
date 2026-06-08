import sys

from src.preprocess_brinlac import build_adni_metadata, build_ukb_metadata
from src.preprocess_3dcnn import build_adni_3dcnn_metadata
from src.preprocess_brainmvp import build_adni_brainmvp_metadata
from src.preprocess_rootstrap import build_adni_rootstrap_metadata
from src.preprocess_sfcn import build_adni_sfcn_metadata, build_ukb_sfcn_metadata


def parse_args(argv):
    args = {"dataset": None, "model": None, "stage": "preprocess"}
    i = 0
    while i < len(argv):
        key = argv[i]
        if key not in {"--dataset", "--model", "--stage"} or i + 1 >= len(argv):
            raise SystemExit("Usage: python data.py --dataset ukb|adni --model brainiac|sfcn|3dcnn|brainmvp|rootstrap")
        args[key[2:]] = argv[i + 1].lower()
        i += 2

    if args["dataset"] not in {"ukb", "adni"}:
        raise SystemExit("--dataset must be ukb or adni.")
    if args["model"] is None:
        raise SystemExit("Usage: python data.py --dataset ukb|adni --model brainiac|sfcn|3dcnn|brainmvp|rootstrap")
    if args["model"] not in {"brainiac", "sfcn", "3dcnn", "brainmvp", "rootstrap"}:
        raise SystemExit("--model must be brainiac, sfcn, 3dcnn, brainmvp, or rootstrap.")
    if args["dataset"] == "ukb" and args["model"] in {"3dcnn", "brainmvp", "rootstrap"}:
        raise SystemExit("UKB currently supports only --model brainiac or sfcn.")
    if args["stage"] != "preprocess":
        raise SystemExit("Only --stage preprocess is supported.")
    return args


def main():
    args = parse_args(sys.argv[1:])
    if args["dataset"] == "adni" and args["model"] == "3dcnn":
        build_adni_3dcnn_metadata()
    elif args["dataset"] == "adni" and args["model"] == "brainmvp":
        build_adni_brainmvp_metadata()
    elif args["dataset"] == "adni" and args["model"] == "rootstrap":
        build_adni_rootstrap_metadata()
    elif args["dataset"] == "adni" and args["model"] == "sfcn":
        build_adni_sfcn_metadata()
    elif args["dataset"] == "adni":
        build_adni_metadata()
    elif args["model"] == "brainiac":
        build_ukb_metadata()
    else:
        build_ukb_sfcn_metadata()


if __name__ == "__main__":
    main()
