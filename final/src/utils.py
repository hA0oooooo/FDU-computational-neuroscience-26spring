import json
import os
import random
from pathlib import Path


def set_seed(seed, deterministic=False):
    seed = int(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def set_cuda_visible_devices(gpu_ids):
    if gpu_ids in {None, "", "none", "None"}:
        return
    if isinstance(gpu_ids, (list, tuple)):
        gpu_ids = ",".join(str(x) for x in gpu_ids)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(gpu_ids))


def setup_distributed():
    import torch
    import torch.distributed as dist

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        torch.cuda.set_device(local_rank)
        try:
            dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))
        except TypeError:
            dist.init_process_group(backend="nccl")
    return local_rank, world_size


def cleanup_distributed():
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    except ImportError:
        pass


def is_rank0():
    return int(os.environ.get("RANK", "0")) == 0


def distributed_barrier():
    try:
        import torch
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            if torch.cuda.is_available() and dist.get_backend() == "nccl":
                dist.barrier(device_ids=[torch.cuda.current_device()])
            else:
                dist.barrier()
    except ImportError:
        pass


def reduce_mean_tensor(tensor):
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        tensor = tensor.clone()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= dist.get_world_size()
    return tensor


def resolve_device(device, local_rank=0):
    import torch

    device = str(device)
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
        if ":" in device:
            return torch.device(device)
        return torch.device(f"cuda:{local_rank}")
    return torch.device(device)


def ensure_output_dirs(base_dir):
    base_dir = Path(base_dir)
    paths = {
        "base": base_dir,
        "weights": base_dir / "weights",
        "predictions": base_dir / "predictions",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path):
    with Path(path).open() as f:
        return json.load(f)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def seed_worker(worker_id):
    import numpy as np
    import torch

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
