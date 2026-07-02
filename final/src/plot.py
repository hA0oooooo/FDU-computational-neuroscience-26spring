import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "report"


def read_rows(path):
    path = Path(path)
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def float_column(rows, name):
    values = []
    for row in rows:
        value = row.get(name)
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            values.append(float("nan"))
    return values


def plot_t1tot2_ssim_step():
    import matplotlib.pyplot as plt

    series = [
        ("2.5D U-Net", ROOT / "output/2dunet/t1tot2", "#1f77b4"),
        ("3D U-Net", ROOT / "output/3dunet/smooth/t1tot2", "#d62728"),
    ]
    plt.figure(figsize=(6.8, 4.2), dpi=180)
    for label, base, color in series:
        step_rows = read_rows(base / "step.csv")
        plt.plot(
            float_column(step_rows, "global_step"),
            float_column(step_rows, "ssim"),
            color=color,
            linewidth=0.55,
            alpha=0.9,
            label=f"{label}",
        )
    plt.xlabel("Step")
    plt.ylabel("SSIM loss")
    plt.grid(alpha=0.18, linewidth=0.5)
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "t1tot2_ssim_step.png", bbox_inches="tight")
    plt.close()


def plot_t1tot2_epoch():
    import matplotlib.pyplot as plt

    series = [
        ("2.5D U-Net", ROOT / "output/2dunet/t1tot2", "#1f77b4"),
        ("3D U-Net", ROOT / "output/3dunet/smooth/t1tot2", "#d62728"),
    ]
    plt.figure(figsize=(6.8, 4.2), dpi=180)
    for label, base, color in series:
        rows = read_rows(base / "epoch.csv")
        epochs = [v + 1 for v in float_column(rows, "epoch")]
        plt.plot(epochs, float_column(rows, "train_loss"), color=color, linewidth=0.95, label=f"{label} train")
        plt.plot(
            epochs,
            float_column(rows, "val_loss"),
            color=color,
            linestyle="--",
            linewidth=0.85,
            label=f"{label} val",
        )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(alpha=0.18, linewidth=0.5)
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "t1tot2_train_epoch.png", bbox_inches="tight")
    plt.close()


def plot_t2tot1_ssim_step():
    import matplotlib.pyplot as plt

    base = ROOT / "output/3dunet/smooth/t2tot1"
    step_rows = read_rows(base / "step.csv")
    plt.figure(figsize=(6.8, 4.2), dpi=180)
    plt.plot(
        float_column(step_rows, "global_step"),
        float_column(step_rows, "ssim"),
        color="#d62728",
        linewidth=0.55,
        alpha=0.9,
        label="3D U-Net",
    )
    plt.xlabel("Step")
    plt.ylabel("SSIM loss")
    plt.grid(alpha=0.18, linewidth=0.5)
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "t2tot1_ssim_step.png", bbox_inches="tight")
    plt.close()


def plot_t2tot1_epoch():
    import matplotlib.pyplot as plt

    base = ROOT / "output/3dunet/smooth/t2tot1"
    rows = read_rows(base / "epoch.csv")
    epochs = [v + 1 for v in float_column(rows, "epoch")]
    plt.figure(figsize=(6.8, 4.2), dpi=180)
    plt.plot(epochs, float_column(rows, "train_loss"), color="#d62728", linewidth=0.95, label="3D U-Net train")
    plt.plot(
        epochs,
        float_column(rows, "val_loss"),
        color="#d62728",
        linestyle="--",
        linewidth=0.85,
        label="3D U-Net val",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(alpha=0.18, linewidth=0.5)
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "t2tot1_train_epoch.png", bbox_inches="tight")
    plt.close()


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    plot_t1tot2_epoch()
    plot_t1tot2_ssim_step()
    plot_t2tot1_epoch()
    plot_t2tot1_ssim_step()


if __name__ == "__main__":
    main()
