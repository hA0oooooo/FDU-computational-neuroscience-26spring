import csv
import os
import re
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


def append_csv_row(path, row, fieldnames):
    path = Path(path)
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def write_split_csv(path, split_records):
    path = Path(path)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "caseid"])
        writer.writeheader()
        for split_name, records in split_records.items():
            for record in records:
                writer.writerow({"split": split_name, "caseid": record.caseid})


def reset_fresh_run(output_paths, run_id, paths):
    for path in paths:
        path = Path(path)
        if path.exists():
            path.unlink()
    weights_dir = output_paths["weights"]
    for path in [weights_dir / f"best_{run_id}.pt", *weights_dir.glob(f"epoch_*_{run_id}.pt")]:
        if path.exists():
            path.unlink()


def infer_start_epoch(checkpoint_path, ckpt):
    match = re.search(r"epoch_(\d+)_", Path(checkpoint_path).name)
    if match:
        return int(match.group(1))
    return int(ckpt.get("epoch", -1)) + 1


def read_epoch_csv_state(path):
    path = Path(path)
    state = {"global_step": None, "best_val_loss": None}
    if not path.exists():
        return state
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            try:
                step = int(float(row.get("global_step", "")))
                state["global_step"] = step if state["global_step"] is None else max(state["global_step"], step)
            except (TypeError, ValueError):
                pass
            try:
                val = float(row.get("val_loss", ""))
            except (TypeError, ValueError):
                continue
            if val == val:
                best = state["best_val_loss"]
                state["best_val_loss"] = val if best is None else min(best, val)
    return state


def read_step_csv_global_step(path):
    path = Path(path)
    best = None
    if not path.exists():
        return best
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            try:
                step = int(float(row.get("global_step", "")))
            except (TypeError, ValueError):
                continue
            best = step if best is None else max(best, step)
    return best


def parse_args():
    return parse_cli_args(
        [
            ("--config", "config", None, None),
            ("--direction", "direction", None, ("t1t2", "t2t1")),
            ("--resume", "resume", None, None),
            ("--device", "device", None, None),
        ]
    )


def main():
    args = parse_args()
    cfg = apply_cli_overrides(load_config(args.config), args)
    set_cuda_visible_devices(cfg.get("runtime", {}).get("gpu_ids"))
    os.environ.setdefault("MPLCONFIGDIR", "/tmp")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

    import torch
    from torch.nn.parallel import DistributedDataParallel
    from tqdm.auto import tqdm

    from src.data import PairedPatchDataset, PairedSliceDataset, build_patch_loader
    from src.losses import build_loss
    from src.model import build_model
    from src.normalization import build_normalizer
    from src.splits import check_required_files, make_splits, read_manifest, read_split_csv
    from src.training import (
        autocast_context,
        checkpoint_state,
        load_checkpoint,
        save_checkpoint,
        split_caseid_dict,
    )
    from src.utils import (
        cleanup_distributed,
        count_parameters,
        distributed_barrier,
        ensure_output_dirs,
        is_rank0,
        reduce_mean_tensor,
        resolve_device,
        set_seed,
        setup_distributed,
    )

    try:
        local_rank, world_size = setup_distributed()
        runtime_cfg = cfg.get("runtime", {})
        split_cfg = cfg.get("split", {})
        seed = int(runtime_cfg.get("seed", split_cfg.get("seed", 42)))
        deterministic = bool(runtime_cfg.get("deterministic", False))
        set_seed(seed, deterministic=deterministic)
        device = resolve_device(runtime_cfg.get("device", "cuda"), local_rank=local_rank)
        if device.type == "cuda" and not deterministic:
            torch.backends.cudnn.benchmark = True
            torch.set_float32_matmul_precision("high")

        spec = direction_spec(cfg)
        run_id = split_run_id(cfg)
        family = model_family(cfg)
        patch_cfg = cfg.get("patch", {})
        train_cfg = cfg.get("training", {})
        data_cache_size = int(train_cfg.get("data_cache_size", 0))
        use_channels_last = family == "2dunet" and bool(train_cfg.get("channels_last", False))
        output_paths = ensure_output_dirs(get_output_dir(cfg))
        step_csv = output_paths["base"] / "step.csv"
        epoch_csv = output_paths["base"] / "epoch.csv"
        split_csv = output_paths["base"] / "split.csv"
        resume_path = train_cfg.get("resume")
        init_checkpoint = cfg.get("pretrained", {}).get("checkpoint")
        split_source_csv = split_cfg.get("csv")
        normalizer_source_path = cfg.get("normalization", {}).get("cache_path")
        if resume_path and init_checkpoint:
            raise ValueError("Use either training.resume or pretrained.checkpoint, not both.")
        if resume_path and not split_csv.exists():
            raise FileNotFoundError(f"Resume requires existing split file: {split_csv}")
        if is_rank0() and train_cfg.get("fresh_start", False) and not resume_path:
            reset_fresh_run(output_paths, run_id, [step_csv, epoch_csv])
        distributed_barrier()

        records = read_manifest(cfg["data"]["root"], cfg["data"]["manifest"])
        missing = check_required_files(records)
        if missing:
            raise FileNotFoundError("Missing NIfTI files:\n" + "\n".join(missing[:20]))
        if split_source_csv:
            split_records = read_split_csv(split_source_csv, records)
        else:
            split_records = make_splits(records, split_cfg)
        if len(split_records["train"]) == 0:
            raise RuntimeError("Training split is empty.")

        if is_rank0():
            print(
                f"[train:{run_id}] start direction={spec['direction']} "
                f"source={spec['source']} target={spec['target']} world_size={world_size}",
            )
            print(
                f"[train:{run_id}] split sizes train={len(split_records['train'])} "
                f"val={len(split_records['val'])} test={len(split_records['test'])}",
            )
            if split_source_csv:
                print(f"[train:{run_id}] split_csv={split_source_csv}")
            elif resume_path:
                print(f"[train:{run_id}] reuse split={split_csv}")
            elif not split_csv.exists():
                write_split_csv(split_csv, split_records)
            else:
                print(f"[train:{run_id}] keep existing split={split_csv}")

        normalizer_path = output_paths["weights"] / f"normalizer_{run_id}.json"
        if resume_path and not normalizer_path.exists():
            raise FileNotFoundError(f"Resume requires existing normalizer: {normalizer_path}")
        normalizer_cache_path = Path(normalizer_source_path) if normalizer_source_path else normalizer_path
        if normalizer_source_path and not normalizer_cache_path.exists():
            raise FileNotFoundError(f"Normalizer reuse file not found: {normalizer_cache_path}")
        normalizer_cfg = dict(cfg.get("normalization", {}))
        if normalizer_source_path:
            normalizer_cfg["recompute"] = False
        if is_rank0():
            normalizer = build_normalizer(
                split_records["train"],
                modalities=["T1", "T2_FLAIR"],
                cfg=normalizer_cfg,
                cache_path=normalizer_cache_path,
            )
        distributed_barrier()
        if not is_rank0():
            normalizer = build_normalizer(
                split_records["train"],
                modalities=["T1", "T2_FLAIR"],
                cfg=normalizer_cfg,
                cache_path=normalizer_cache_path,
            )

        if family == "2dunet":
            model_input_size = list_int(patch_cfg.get("crop_size", [160, 160]), "patch.crop_size")
            slice_offsets = list_int(
                cfg.get("slice", {}).get("slice_offsets", [-3, -2, -1, 0, 1, 2, 3]),
                "slice.slice_offsets",
            )
            train_ds = PairedSliceDataset(
                split_records["train"],
                source_modality=spec["source"],
                target_modality=spec["target"],
                normalizer=normalizer,
                crop_size=model_input_size,
                slice_offsets=slice_offsets,
                samples_per_case=patch_cfg.get("samples_per_case_per_epoch", 16),
                crop_foreground_prob=patch_cfg.get("crop_foreground_prob", 0.8),
                deterministic=False,
                seed=seed,
                cache_size=data_cache_size,
            )
            if split_records["val"]:
                val_ds = PairedSliceDataset(
                    split_records["val"],
                    source_modality=spec["source"],
                    target_modality=spec["target"],
                    normalizer=normalizer,
                    crop_size=model_input_size,
                    slice_offsets=slice_offsets,
                    samples_per_case=patch_cfg.get("val_samples_per_case", 2),
                    crop_foreground_prob=0.0,
                    deterministic=True,
                    seed=seed + 17,
                    cache_size=data_cache_size,
                )
            else:
                val_ds = None
        else:
            model_input_size = list_int(patch_cfg.get("size", [64, 128, 128]), "patch.size")
            train_ds = PairedPatchDataset(
                split_records["train"],
                source_modality=spec["source"],
                target_modality=spec["target"],
                normalizer=normalizer,
                patch_size=model_input_size,
                patches_per_case=patch_cfg.get("patches_per_case_per_epoch", 8),
                foreground_prob=patch_cfg.get("foreground_prob", 0.7),
                deterministic=False,
                seed=seed,
                cache_size=data_cache_size,
            )
            if split_records["val"]:
                val_ds = PairedPatchDataset(
                    split_records["val"],
                    source_modality=spec["source"],
                    target_modality=spec["target"],
                    normalizer=normalizer,
                    patch_size=model_input_size,
                    patches_per_case=patch_cfg.get("val_patches_per_case", 1),
                    foreground_prob=0.0,
                    deterministic=True,
                    seed=seed + 17,
                    cache_size=data_cache_size,
                )
            else:
                val_ds = None

        val_loader = None

        train_loader, train_sampler = build_patch_loader(
            train_ds,
            batch_size=train_cfg.get("batch_size", 1),
            shuffle=True,
            num_workers=train_cfg.get("num_workers", 2),
            distributed=world_size > 1,
            seed=seed,
            prefetch_factor=train_cfg.get("prefetch_factor", 2),
            persistent_workers=train_cfg.get("persistent_workers", True),
            drop_last=train_cfg.get("drop_last", False),
            case_grouped=train_cfg.get("case_grouped_sampling", False),
        )
        if val_ds is not None:
            val_loader, _ = build_patch_loader(
                val_ds,
                batch_size=train_cfg.get("val_batch_size", 1 if family == "3dunet" else train_cfg.get("batch_size", 1)),
                shuffle=False,
                num_workers=train_cfg.get("num_workers", 2),
                distributed=False,
                seed=seed,
                prefetch_factor=train_cfg.get("prefetch_factor", 2),
                persistent_workers=train_cfg.get("persistent_workers", True),
                drop_last=False,
                case_grouped=False,
            )

        model = build_model(cfg, patch_size=model_input_size).to(device)
        if use_channels_last:
            model = model.to(memory_format=torch.channels_last)
        criterion = build_loss(cfg).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(train_cfg.get("lr", 1e-4)),
            weight_decay=float(train_cfg.get("weight_decay", 1e-5)),
        )
        amp_enabled = bool(train_cfg.get("amp", True)) and device.type == "cuda"
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        except TypeError:
            scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

        start_epoch = 0
        global_step = 0
        best_score = None
        if init_checkpoint:
            ckpt = load_checkpoint(init_checkpoint, model, optimizer=None, scaler=None, map_location=device)
            if is_rank0():
                print(
                    f"[train:{run_id}] initialized model from {init_checkpoint} "
                    f"checkpoint_epoch={ckpt.get('epoch')} checkpoint_global_step={ckpt.get('global_step')}",
                )
        if resume_path:
            history_state = read_epoch_csv_state(epoch_csv)
            history_global_step = history_state.get("global_step")
            if history_global_step is None:
                history_global_step = read_step_csv_global_step(step_csv)
            ckpt = load_checkpoint(resume_path, model, optimizer=optimizer, scaler=scaler, map_location=device)
            start_epoch = infer_start_epoch(resume_path, ckpt)
            global_step = max(int(ckpt.get("global_step") or 0), int(history_global_step or 0))
            best_score = ckpt.get("best_val_loss")
            if best_score is None:
                best_score = ckpt.get("best_score")
            if best_score is None:
                best_score = history_state.get("best_val_loss")
            if best_score is not None:
                best_score = float(best_score)
            if is_rank0():
                print(
                    f"[train:{run_id}] resumed checkpoint={resume_path} "
                    f"start_epoch={start_epoch} global_step={global_step} best_val_loss={best_score}",
                )

        if world_size > 1:
            model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

        if is_rank0():
            print(f"[train:{run_id}] parameters={count_parameters(model)}")
            print(f"[train:{run_id}] normalizer={normalizer.to_dict()}")

        epochs = int(train_cfg.get("epochs", 100))
        save_every = int(train_cfg.get("save_every", 10))
        val_every = int(train_cfg.get("val_every", 1))
        grad_clip_norm = train_cfg.get("grad_clip_norm", None)

        for epoch in range(start_epoch, epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            model.train()
            epoch_loss_sum = 0.0
            epoch_count = 0
            progress = tqdm(
                train_loader,
                total=len(train_loader),
                desc=f"epoch {epoch + 1}/{epochs}",
                dynamic_ncols=True,
                disable=not is_rank0(),
            )
            for batch in progress:
                source = batch["source"].to(device=device, dtype=torch.float32, non_blocking=True)
                target = batch["target"].to(device=device, dtype=torch.float32, non_blocking=True)
                if use_channels_last:
                    source = source.contiguous(memory_format=torch.channels_last)
                    target = target.contiguous(memory_format=torch.channels_last)
                optimizer.zero_grad(set_to_none=True)
                with autocast_context(device, amp_enabled):
                    pred = model(source)
                    parts = criterion(pred, target)
                    loss = parts["loss"]
                if not torch.isfinite(loss).all():
                    raise RuntimeError(f"Non-finite training loss at epoch={epoch} step={global_step}")
                scaler.scale(loss).backward()
                if grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
                scaler.step(optimizer)
                scaler.update()

                batch_size = int(source.shape[0])
                loss_value = float(loss.detach().cpu())
                epoch_loss_sum += loss_value * batch_size
                epoch_count += batch_size
                global_step += 1
                if is_rank0():
                    step_row = {
                        "epoch": epoch,
                        "global_step": global_step,
                        "train_loss": loss_value,
                        "l1": float(parts["l1"].detach().cpu()),
                        "ssim": float(parts["ssim"].detach().cpu()) if "ssim" in parts else None,
                        "grad": float(parts["grad"].detach().cpu()) if "grad" in parts else None,
                    }
                    append_csv_row(
                        step_csv,
                        step_row,
                        ["epoch", "global_step", "train_loss", "l1", "ssim", "grad"],
                    )
                    progress.set_postfix(loss=f"{loss_value:.4f}")

            stats = torch.tensor([epoch_loss_sum, epoch_count], device=device, dtype=torch.float64)
            if world_size > 1:
                stats = reduce_mean_tensor(stats)
            train_epoch_loss = float(stats[0].item() / max(stats[1].item(), 1.0))

            val_loss = None
            if is_rank0() and val_loader is not None and (epoch + 1) % val_every == 0:
                from src.training import validation_loss

                val_loss = validation_loss(model, val_loader, criterion, device, amp=amp_enabled)
            monitor = val_loss if val_loss is not None else train_epoch_loss
            saved = []
            if is_rank0():
                is_best = best_score is None or monitor < best_score
                if is_best:
                    best_score = monitor
                state = checkpoint_state(
                    model,
                    optimizer,
                    scaler,
                    epoch,
                    global_step,
                    best_score,
                    cfg,
                    normalizer,
                    split_caseid_dict(split_records),
                )
                if is_best:
                    best_path = output_paths["weights"] / f"best_{run_id}.pt"
                    save_checkpoint(best_path, state)
                    saved.append(str(best_path))
                if save_every > 0 and (epoch + 1) % save_every == 0:
                    epoch_path = output_paths["weights"] / f"epoch_{epoch + 1:03d}_{run_id}.pt"
                    save_checkpoint(epoch_path, state)
                    saved.append(str(epoch_path))
                epoch_row = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "train_loss": train_epoch_loss,
                    "val_loss": val_loss,
                    "checkpoint_path": ";".join(saved),
                }
                append_csv_row(
                    epoch_csv,
                    epoch_row,
                    ["epoch", "global_step", "train_loss", "val_loss", "checkpoint_path"],
                )
                message = (
                    f"[train:{run_id}] epoch={epoch} global_step={global_step} "
                    f"train_loss={train_epoch_loss:.6f} val_loss={val_loss} saved={saved}"
                )
                print(message)
            distributed_barrier()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
