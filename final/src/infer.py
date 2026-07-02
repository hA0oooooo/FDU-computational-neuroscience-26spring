def window_starts(size, patch, stride):
    size = int(size)
    patch = int(patch)
    stride = int(stride)
    if size <= patch:
        return [0]
    starts = list(range(0, size - patch + 1, stride))
    if starts[-1] != size - patch:
        starts.append(size - patch)
    return starts


def pad_volume_to_patch(volume, patch_size):
    import numpy as np

    pad_width = []
    for size, patch in zip(volume.shape, patch_size):
        pad_width.append((0, max(0, int(patch) - int(size))))
    if any(right > 0 for _, right in pad_width):
        volume = np.pad(volume, pad_width, mode="constant")
    return volume


def pad_hw_to_multiple(volume, multiple):
    import numpy as np

    multiple = int(multiple)
    h, w = int(volume.shape[-2]), int(volume.shape[-1])
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h > 0 or pad_w > 0:
        pad_width = [(0, 0) for _ in range(volume.ndim)]
        pad_width[-2] = (0, pad_h)
        pad_width[-1] = (0, pad_w)
        volume = np.pad(volume, pad_width, mode="constant")
    return volume


def sliding_window_predict(model, source_norm_dhw, patch_size, stride, device, amp=True, batch_size=1):
    from contextlib import nullcontext

    import numpy as np
    import torch

    model.eval()
    original_shape = tuple(source_norm_dhw.shape)
    volume = pad_volume_to_patch(source_norm_dhw, patch_size)
    patch_size = tuple(int(x) for x in patch_size)
    stride = tuple(int(x) for x in stride)
    starts_d = window_starts(volume.shape[0], patch_size[0], stride[0])
    starts_h = window_starts(volume.shape[1], patch_size[1], stride[1])
    starts_w = window_starts(volume.shape[2], patch_size[2], stride[2])
    windows = [(d, h, w) for d in starts_d for h in starts_h for w in starts_w]

    output = np.zeros(volume.shape, dtype=np.float32)
    counts = np.zeros(volume.shape, dtype=np.float32)
    batch_size = int(batch_size)
    enabled_amp = bool(amp) and getattr(device, "type", str(device)) == "cuda"

    with torch.inference_mode():
        for i in range(0, len(windows), batch_size):
            batch_windows = windows[i : i + batch_size]
            patches = []
            for d, h, w in batch_windows:
                patch = volume[d : d + patch_size[0], h : h + patch_size[1], w : w + patch_size[2]]
                patches.append(patch)
            tensor = torch.from_numpy(np.stack(patches)[:, None]).to(device=device, dtype=torch.float32)
            context = torch.autocast(device_type="cuda") if enabled_amp else nullcontext()
            with context:
                pred = model(tensor).detach().float().cpu().numpy()[:, 0]
            for pred_patch, (d, h, w) in zip(pred, batch_windows):
                output[d : d + patch_size[0], h : h + patch_size[1], w : w + patch_size[2]] += pred_patch
                counts[d : d + patch_size[0], h : h + patch_size[1], w : w + patch_size[2]] += 1.0

    output = output / np.maximum(counts, 1.0)
    d, h, w = original_shape
    return output[:d, :h, :w]


def slice_stack(volume, z, offsets):
    import numpy as np

    depth = int(volume.shape[0])
    indices = [min(max(int(z) + int(offset), 0), depth - 1) for offset in offsets]
    return np.stack([volume[idx] for idx in indices], axis=0)


def slice_stack_predict(model, source_norm_dhw, slice_offsets, device, amp=True, batch_size=16, pad_multiple=16):
    from contextlib import nullcontext

    import numpy as np
    import torch

    model.eval()
    d, h, w = source_norm_dhw.shape
    padded = pad_hw_to_multiple(source_norm_dhw, pad_multiple)
    output = np.zeros((d, padded.shape[1], padded.shape[2]), dtype=np.float32)
    batch_size = int(batch_size)
    offsets = tuple(int(x) for x in slice_offsets)
    enabled_amp = bool(amp) and getattr(device, "type", str(device)) == "cuda"

    with torch.inference_mode():
        for start in range(0, d, batch_size):
            z_values = list(range(start, min(start + batch_size, d)))
            stacks = [slice_stack(padded, z, offsets) for z in z_values]
            tensor = torch.from_numpy(np.stack(stacks)).to(device=device, dtype=torch.float32)
            context = torch.autocast(device_type="cuda") if enabled_amp else nullcontext()
            with context:
                pred = model(tensor).detach().float().cpu().numpy()[:, 0]
            for z, pred_slice in zip(z_values, pred):
                output[z] = pred_slice
    return output[:, :h, :w]
