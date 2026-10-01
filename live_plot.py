import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt

HISTORY_FILE = Path("results/davis/training_history.csv")
REFRESH_SECONDS = 2

plt.ion()
fig, ax = plt.subplots(figsize=(10, 6))


def read_history():
    if not HISTORY_FILE.exists():
        return [], [], []

    epochs = []
    training = []
    validation = []

    try:
        with HISTORY_FILE.open("r", newline="") as f:
            rows = list(csv.DictReader(f))

        for row in rows:
            try:
                epochs.append(int(row["epoch"]))
                training.append(float(row["training_mse"]))
                validation.append(float(row["validation_mse"]))
            except (ValueError, KeyError):
                continue

    except (PermissionError, OSError):
        pass

    return epochs, training, validation


while True:
    epochs, training, validation = read_history()

    ax.clear()

    if epochs:
        ax.plot(epochs, training, label="Training MSE", linewidth=2)
        ax.plot(epochs, validation, label="Validation MSE", linewidth=2)

        ax.set_title("MGraphDTA Live Training")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax.text(
            0.02,
            0.95,
            f"Latest epoch: {epochs[-1]}",
            transform=ax.transAxes,
            verticalalignment="top",
        )
    else:
        ax.set_title("MGraphDTA Live Training")
        ax.text(
            0.5,
            0.5,
            "Waiting for training data...",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    plt.tight_layout()
    plt.pause(0.1)
    time.sleep(REFRESH_SECONDS)