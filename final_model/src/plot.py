"""Plot recorded metrics; does not train or read evaluation labels."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .utils import read_json


def plot(run_dir):
    root = Path(run_dir)
    rows = read_json(root/"metrics.json")
    epochs = [r["epoch"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for prefix, label, color in [("train", "Training", "#2474b5"), ("val", "Validation", "#d85b35")]:
        axes[0].plot(epochs, [r[f"{prefix}_loss"] for r in rows], label=label, color=color)
        axes[1].plot(epochs, [100*r[f"{prefix}_accuracy"] for r in rows], label=label, color=color)
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.2)
        ax.legend()
    axes[0].set_ylabel("Cross-entropy")
    axes[1].set_ylabel("Accuracy (%)")
    fig.suptitle(root.name)
    fig.tight_layout()
    fig.savefig(root/"curves.png", dpi=160)
    plt.close(fig)
    return root/"curves.png"


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    print(plot(p.parse_args().run_dir))
