import csv
import os
from pathlib import Path

from src.config import (
    apply_cli_overrides,
    direction_spec,
    get_output_dir,
    list_int,
    load_config,
    model_family,
    parse_cli_args,
    split_run_id,
)
from src.utils import set_cuda_visible_devices


def parse_args():
    return parse_cli_args(
        [
            ("--config", "config", None, None),
            ("--checkpoint", "checkpoint", None, None),
            ("--direction", "direction", None, ("t1t2", "t2t1")),
            ("--eval-split", "eval_split", None, ("train", "val", "test", "all")),
            ("--device", "device", None, None),
        ]
    )


def select_records(split_records, eval_split):
    if eval_split is None:
        eval_split = "test"
    if eval_split == "all":
        return split_records["train"] + split_records["val"] + split_records["test"], eval_split
    return split_records[eval_split], eval_split


def default_checkpoint(output_paths, run_id):
    best = output_paths["weights"] / f"best_{run_id}.pt"
    if best.exists():
        return best
    raise FileNotFoundError(f"No checkpoint found for run_id={run_id}: tried {best}")


def resolve_checkpoint(pattern, output_paths, run_id):
    if pattern is None:
        return default_checkpoint(output_paths, run_id)
    return Path(str(pattern).format(run_id=run_id, output=str(output_paths["base"])))


def write_aggregate(output_paths, rows, summary):
    csv_path = output_paths["base"] / "eval.csv"
    if rows:
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    summary_path = output_paths["base"] / "summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        for key, value in summary.items():
            writer.writerow({"metric": key, "value": value})


def rank_slice(records, rank, world_size):
    return records[rank::world_size]


def gather_rows(rows):
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        return rows
    gathered = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, rows)
    return [row for part in gathered for row in part]


def sort_rows(rows):
    return sorted(rows, key=lambda row: str(row.get("caseid", "")))


def main():
    args = parse_args()
    cfg = apply_cli_overrides(load_config(args.config), args)
    set_cuda_visible_devices(cfg.get("runtime", {}).get("gpu_ids"))
    os.environ.setdefault("MPLCONFIGDIR", "/tmp")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

    from src.evaluation import evaluate_cases, load_model_checkpoint
    from src.metrics import summarize_metrics
    from src.model import build_model
    from src.splits import make_splits, read_manifest, read_split_csv
    from src.utils import cleanup_distributed, ensure_output_dirs, is_rank0, resolve_device, set_seed, setup_distributed

    try:
        local_rank, world_size = setup_distributed()
        rank = int(os.environ.get("RANK", "0"))
        runtime_cfg = cfg.get("runtime", {})
        split_cfg = cfg.get("split", {})
        seed = int(runtime_cfg.get("seed", split_cfg.get("seed", 42)))
        set_seed(seed, deterministic=False)
        device = resolve_device(runtime_cfg.get("device", "cuda"), local_rank=local_rank)
        spec = direction_spec(cfg)
        output_paths = ensure_output_dirs(get_output_dir(cfg))
        records = read_manifest(cfg["data"]["root"], cfg["data"]["manifest"])
        patch_cfg = cfg.get("patch", {})
        family = model_family(cfg)
        if family == "2dunet":
            model_input_size = list_int(patch_cfg.get("crop_size", [160, 160]), "patch.crop_size")
        else:
            model_input_size = list_int(patch_cfg.get("size", [64, 128, 128]), "patch.size")

        if is_rank0():
            print(
                f"[eval] start direction={spec['direction']} source={spec['source']} "
                f"target={spec['target']} world_size={world_size}",
            )

        split_source_csv = split_cfg.get("csv")
        if split_source_csv:
            split_records = read_split_csv(split_source_csv, records)
        else:
            split_records = make_splits(records, split_cfg)
        eval_records, eval_name = select_records(split_records, args.eval_split)
        if len(eval_records) == 0:
            raise RuntimeError(f"No cases selected for eval split: {eval_name}")
        local_records = rank_slice(eval_records, rank, world_size)
        run_id = split_run_id(cfg)
        if is_rank0():
            print(f"[eval:{run_id}_{eval_name}] cases={len(eval_records)} per_rank~={len(local_records)}")
        ckpt_path = resolve_checkpoint(args.checkpoint, output_paths, run_id)
        model = build_model(cfg, patch_size=model_input_size).to(device)
        _, normalizer = load_model_checkpoint(ckpt_path, model, device)
        rows, _ = evaluate_cases(
            cfg,
            model,
            normalizer,
            local_records,
            spec,
            output_paths,
            device,
            tag=f"{run_id}_{eval_name}_rank{rank}",
            write_outputs=False,
        )
        rows = gather_rows(rows)
        if is_rank0():
            rows = sort_rows(rows)
            summary = summarize_metrics(rows)
            write_aggregate(output_paths, rows, summary)
            print(f"[eval:{run_id}_{eval_name}] summary={summary}")
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
