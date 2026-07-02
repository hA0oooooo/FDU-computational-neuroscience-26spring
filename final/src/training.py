from pathlib import Path
from contextlib import nullcontext


def checkpoint_state(
    model,
    optimizer,
    scaler,
    epoch,
    global_step,
    best_score,
    cfg,
    normalizer,
    split_caseids,
):
    model_to_save = model.module if hasattr(model, "module") else model
    return {
        "model": model_to_save.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_score": None if best_score is None else float(best_score),
        "best_val_loss": None if best_score is None else float(best_score),
        "config": cfg,
        "normalizer": normalizer.to_dict(),
        "split_caseids": split_caseids,
    }


def save_checkpoint(path, state):
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, str(path))


def load_checkpoint(path, model, optimizer=None, scaler=None, map_location="cpu"):
    import torch

    ckpt = torch.load(str(path), map_location=map_location)
    model_to_load = model.module if hasattr(model, "module") else model
    model_to_load.load_state_dict(ckpt["model"])
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    return ckpt


def autocast_context(device, enabled):
    import torch

    if bool(enabled) and device.type == "cuda":
        return torch.autocast(device_type="cuda")
    return nullcontext()


def validation_loss(model, loader, criterion, device, amp=True):
    import torch

    if loader is None:
        return None
    model.eval()
    total = 0.0
    count = 0
    with torch.inference_mode():
        for batch in loader:
            source = batch["source"].to(device=device, dtype=torch.float32, non_blocking=True)
            target = batch["target"].to(device=device, dtype=torch.float32, non_blocking=True)
            with autocast_context(device, amp):
                pred = model(source)
                parts = criterion(pred, target)
            total += float(parts["loss"].detach().cpu()) * source.shape[0]
            count += source.shape[0]
    model.train()
    if count == 0:
        return None
    return total / count


def split_caseid_dict(split_records):
    return {name: [record.caseid for record in records] for name, records in split_records.items()}
