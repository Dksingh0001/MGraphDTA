import pandas as pd
import matplotlib.pyplot as plt

csv_path = r"results\davis\run_112_backup\training_history.csv"
output_path = r"results\davis\run_112_backup\training_curve_112_epochs.png"

df = pd.read_csv(csv_path)

best_idx = df["validation_mse"].idxmin()
best_epoch = int(df.loc[best_idx, "epoch"])
best_val = float(df.loc[best_idx, "validation_mse"])

plt.figure(figsize=(10, 6))

plt.plot(
    df["epoch"],
    df["training_mse"],
    label="Training MSE"
)

plt.plot(
    df["epoch"],
    df["validation_mse"],
    label="Validation MSE"
)

plt.scatter(
    best_epoch,
    best_val,
    s=60,
    label=f"Best validation MSE: epoch {best_epoch} = {best_val:.6f}"
)

plt.title("MGraphDTA Davis Training Curve - 112 Epochs")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(output_path, dpi=200)
plt.close()

print("Graph created successfully!")
print("Output:", output_path)
print("Epochs:", len(df))
print("Best epoch:", best_epoch)
print("Best validation MSE:", best_val)
