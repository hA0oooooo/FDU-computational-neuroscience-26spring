import math


def _global_ssim_2d(pred, target, data_range):
    import numpy as np

    pred = pred.astype(np.float64, copy=False)
    target = target.astype(np.float64, copy=False)
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    mu_x = pred.mean()
    mu_y = target.mean()
    var_x = pred.var()
    var_y = target.var()
    cov = ((pred - mu_x) * (target - mu_y)).mean()
    return ((2 * mu_x * mu_y + c1) * (2 * cov + c2)) / ((mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2))


def slice_wise_ssim(pred_dhw, target_dhw, data_range):
    import numpy as np

    if data_range <= 0:
        return 1.0 if np.allclose(pred_dhw, target_dhw) else 0.0
    try:
        from skimage.metrics import structural_similarity

        values = [
            structural_similarity(target_dhw[z], pred_dhw[z], data_range=data_range)
            for z in range(target_dhw.shape[0])
        ]
    except ImportError:
        values = [_global_ssim_2d(pred_dhw[z], target_dhw[z], data_range) for z in range(target_dhw.shape[0])]
    return float(np.mean(values))


def case_metrics(pred_raw_dhw, target_raw_dhw):
    import numpy as np

    pred = pred_raw_dhw.astype(np.float64, copy=False)
    target = target_raw_dhw.astype(np.float64, copy=False)
    diff = pred - target
    mae = float(np.mean(np.abs(diff)))
    mse = float(np.mean(diff * diff))
    data_range = float(target.max() - target.min())
    if mse == 0:
        psnr = float("inf")
    elif data_range <= 0:
        psnr = float("-inf")
    else:
        psnr = float(10.0 * math.log10((data_range * data_range) / mse))
    ssim = slice_wise_ssim(pred, target, data_range)
    return {
        "mae": mae,
        "mse": mse,
        "psnr": psnr,
        "ssim": ssim,
        "data_range": data_range,
    }


def summarize_metrics(rows):
    import numpy as np

    keys = ["mae", "mse", "psnr", "ssim"]
    summary = {"num_cases": len(rows)}
    for key in keys:
        vals = np.asarray([row[key] for row in rows], dtype=np.float64)
        finite = vals[np.isfinite(vals)]
        if finite.size == 0:
            summary[f"{key}_mean"] = float("nan")
            summary[f"{key}_std"] = float("nan")
        else:
            summary[f"{key}_mean"] = float(finite.mean())
            summary[f"{key}_std"] = float(finite.std())
    return summary
