import os
import json
import numpy as np
import matplotlib.pyplot as plt


def lorenz96_step(x, F):
    """
    One Euler step of Lorenz-96 dynamics.
    dx_i/dt = (x_{i+1} - x_{i-2}) * x_{i-1} - x_i + F
    """
    N = len(x)
    dx = np.zeros(N)
    for i in range(N):
        dx[i] = (
            (x[(i + 1) % N] - x[(i - 2) % N]) * x[(i - 1) % N]
            - x[i]
            + F
        )
    return dx


def simulate_lorenz96(N=40, F=8.0, T=2000, dt=0.01):
    """
    Simulate Lorenz-96 for T steps.
    """
    x = np.random.randn(N)
    trajectory = np.zeros((T, N))

    for t in range(T):
        dx = lorenz96_step(x, F)
        x = x + dt * dx
        trajectory[t] = x

    return trajectory


def generate_dataset(
    N=40,
    F_train=8.0,
    F_test=10.0,
    T_train=2000,
    T_test=2000,
    dt=0.01
):
    """
    Generate training data under F_train
    and test data under F_test.
    """
    train_data = simulate_lorenz96(N, F_train, T_train, dt)
    test_data = simulate_lorenz96(N, F_test, T_test, dt)
    return train_data, test_data


if __name__ == "__main__":
    # Reproducibility
    SEED = 42
    np.random.seed(SEED)

    # Output directory
    output_dir = "data/simulation/lorenz96"
    os.makedirs(output_dir, exist_ok=True)

    # Forcing parameters
    F_train = 8.0
    F_test = 10.0

    # Generate data
    train, test = generate_dataset(F_train=F_train, F_test=F_test)

    # Save dataset
    data_path = os.path.join(output_dir, "data.npz")
    np.savez(
        data_path,
        train=train,
        test=test,
        F_train=F_train,
        F_test=F_test,
        seed=SEED
    )

    # Compute extreme values
    report = {
        "train": {
            "max": float(np.max(train)),
            "min": float(np.min(train)),
            "mean": float(np.mean(train)),
            "std": float(np.std(train))
        },
        "test": {
            "max": float(np.max(test)),
            "min": float(np.min(test)),
            "mean": float(np.mean(test)),
            "std": float(np.std(test))
        },
        "difference_in_max": float(np.max(test) - np.max(train)),
        "seed": SEED
    }

    report_path = os.path.join(output_dir, "report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)

    # Save plot
    plt.figure(figsize=(10, 4))
    plt.plot(train[:, 0], label=f"Train (F={F_train})")
    plt.plot(test[:, 0], label=f"Test (F={F_test})", alpha=0.7)
    plt.legend()
    plt.title("Lorenz-96: Forcing Amplitude Shift")
    plt.tight_layout()

    plot_path = os.path.join(output_dir, "plot.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()