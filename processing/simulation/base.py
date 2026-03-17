import os
import json
import numpy as np
import matplotlib.pyplot as plt

class BaseSimulation:
    def __init__(self, output_dir, seed=42):
        self.output_dir = output_dir
        self.seed = seed
        np.random.seed(seed)
        os.makedirs(output_dir, exist_ok=True)

    def simulate(self, *args, **kwargs):
        raise NotImplementedError

    def generate_dataset(self, *args, **kwargs):
        raise NotImplementedError

    def save_data(self, train, test, metadata):
        path = os.path.join(self.output_dir, "data.npz")
        np.savez(path, train=train, test=test, **metadata)

    def save_report(self, train, test):
        report = {
            "train": {
                "max": float(np.max(train)),
                "min": float(np.min(train)),
                "mean": float(np.mean(train)),
                "std": float(np.std(train)),
            },
            "test": {
                "max": float(np.max(test)),
                "min": float(np.min(test)),
                "mean": float(np.mean(test)),
                "std": float(np.std(test)),
            },
            "difference_in_max": float(np.max(test) - np.max(train)),
            "seed": self.seed,
        }

        path = os.path.join(self.output_dir, "report.json")
        with open(path, "w") as f:
            json.dump(report, f, indent=4)

    def plot_spatiotemporal(self, train, test, metadata, filename="spacetime.png"):
        fig, axes = plt.subplots(
            2, 1,
            figsize=(10, 8),
            sharex=True,
            constrained_layout=True
        )

        im1 = axes[0].imshow(train.T, aspect='auto', origin='lower')
        axes[0].set_title(f"Train (F={metadata['F_train']})")
        axes[0].set_ylabel("Spatial index")

        im2 = axes[1].imshow(test.T, aspect='auto', origin='lower')
        axes[1].set_title(f"Test (F={metadata['F_test']})")
        axes[1].set_xlabel("Time")
        axes[1].set_ylabel("Spatial index")

        fig.colorbar(im1, ax=axes, label="State value")

        plt.savefig(os.path.join(self.output_dir, filename), dpi=300)
        plt.close()

    def plot_timeseries(self, train, test, metadata, filename="timeseries.png"):
        plt.figure(figsize=(10, 4))

        plt.plot(train[:, 0], label=f"Train (F={metadata['F_train']})")
        plt.plot(test[:, 0], label=f"Test (F={metadata['F_test']})", alpha=0.7)

        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.title("Time Series at Spatial Index 0")
        plt.legend()
        plt.tight_layout()

        plt.savefig(os.path.join(self.output_dir, filename), dpi=300)
        plt.close()

    def run(self):
        train, test, metadata = self.generate_dataset()
        self.save_data(train, test, metadata)
        self.save_report(train, test)
        self.plot_spatiotemporal(train, test, metadata)
        self.plot_timeseries(train, test, metadata)