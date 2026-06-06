import sys

from src.preprocess import build_ukb_metadata
from src.preprocess_sfcn import build_ukb_sfcn_metadata


def parse_args(argv):
    args = {"dataset": None, "model": None}
    i = 0
    while i < len(argv):
        key = argv[i]
        if key not in {"--dataset", "--model"} or i + 1 >= len(argv):
            raise SystemExit("Usage: python data.py --dataset ukb --model brainiac|sfcn")
        args[key[2:]] = argv[i + 1].lower()
        i += 2

    if args["dataset"] == "adni":
        raise NotImplementedError("ADNI is intentionally not implemented in the current UKB stage.")
    if args["dataset"] != "ukb":
        raise SystemExit("Only --dataset ukb is supported in the current stage.")
    if args["model"] is None:
        raise SystemExit("Usage: python data.py --dataset ukb --model brainiac|sfcn")
    if args["model"] not in {"brainiac", "sfcn"}:
        raise SystemExit("--model must be brainiac or sfcn.")
    return args


def main():
    args = parse_args(sys.argv[1:])
    if args["model"] == "brainiac":
        build_ukb_metadata()
    else:
        build_ukb_sfcn_metadata()


if __name__ == "__main__":
    main()
