from pathlib import Path


def load_nifti_xyz(path):
    import nibabel as nib
    import numpy as np

    img = nib.load(str(path))
    arr = np.asanyarray(img.dataobj).astype(np.float32, copy=False)
    return arr, img


def xyz_to_dhw(arr_xyz):
    import numpy as np

    return np.transpose(arr_xyz, (2, 1, 0))


def dhw_to_xyz(arr_dhw):
    import numpy as np

    return np.transpose(arr_dhw, (2, 1, 0))


def load_nifti_dhw(path):
    arr_xyz, img = load_nifti_xyz(path)
    return xyz_to_dhw(arr_xyz), img


def save_nifti_dhw(volume_dhw, reference_img, path):
    import nibabel as nib
    import numpy as np

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr_xyz = dhw_to_xyz(volume_dhw).astype(np.float32, copy=False)
    header = reference_img.header.copy()
    header.set_data_dtype(np.float32)
    out = nib.Nifti1Image(arr_xyz, reference_img.affine, header=header)
    nib.save(out, str(path))


def pair_is_consistent(record, affine_atol=1e-3):
    import numpy as np

    t1_xyz, t1_img = load_nifti_xyz(record.path("T1"))
    t2_xyz, t2_img = load_nifti_xyz(record.path("T2_FLAIR"))
    same_shape = tuple(t1_xyz.shape) == tuple(t2_xyz.shape)
    same_affine = np.allclose(t1_img.affine, t2_img.affine, atol=affine_atol)
    return same_shape, same_affine, tuple(t1_xyz.shape), tuple(t2_xyz.shape)


def raw_stats(arr):
    return {
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
    }

