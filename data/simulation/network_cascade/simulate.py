import os
import json
import numpy as np
import matplotlib.pyplot as plt


def build_random_graph(N, connectivity=0.1):
    """
    Build a random directed graph adjacency matrix.
    Weights are normalized so spectral radius ≈ 1.
    """
    A = (np.random.rand(N, N) < connectivity).astype(float)
    np.fill_diagonal(A, 0.0)

    # Normalize by largest eigenvalue magnitude
    eigvals = np.linalg.eigvals(A)
    spectral_radius = np.max(np.abs(eigvals))
    if spectral_radius > 0:
        A = A / spectral_radius

    return A


def cascade_step(x, A, amp, shock_scale=0.5, decay=0.1):
    """
    One step of nonlinear cascade dynamics.

    x_{t+1} = (1 - decay) * x_t + amp * tanh(A @ x_t) + shock

    amp controls effective branching ratio.
    """
    shock = shock_scale * np.random.randn(len(x))
    transport = np.tanh(A @ x)
    x_next = (1 - decay) * x + amp * transport + shock
    return x_next


def simulate_cascade(N=40, amp=0.8, T=2000):
    """
    Simulate cascade system.
    amp < 1 → subcritical
    amp ≈ 1 → near-critical
    amp > 1 → supercritical
    """
    A = build_random_graph(N)
    x = np.zeros(N)
    trajectory = np.zeros((T, N))

    for t in range(T):
        x = cascade_step(x, A, amp)
        trajectory[t] = x

    return trajectory


def generate_dataset(
    N=40,
    amp_train=0.8,
    amp_test=0.98,
    T_train=2000,
    T_test=2000
):
    """
    Generate training data in subcritical regime
    and test data near criticality.
    """
    train_data = simulate_cascade(N, amp_train, T_train)
    test_data = simulate_cascade(N, amp_test, T_test)
    return train_data, test_data


if __name__ == "__main__":
    SEED = 42
    np.random.seed(SEED)

    output_dir = "data/simulation/network_cascade"
    os.makedirs(output_dir, exist_ok=True)

    # Amplification parameters
    amp_train = 0.8    # subcritical
    amp_test = 0.98    # near-critical

    train, test = generate_dataset(
        amp_train=amp_train,
        amp_test=amp_test
    )

    # Save dataset
    data_path = os.path.join(output_dir, "data.npz")
    np.savez(
        data_path,
        train=train,
        test=test,
        amp_train=amp_train,
        amp_test=amp_test,
        seed=SEED
    )

    # Compute extreme statistics
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
    plt.plot(train[:, 0], label=f"Train (amp={amp_train})")
    plt.plot(test[:, 0], label=f"Test (amp={amp_test})", alpha=0.7)
    plt.legend()
    plt.title("Network Cascade: Subcritical → Near-Critical Shift")
    plt.tight_layout()

    plot_path = os.path.join(output_dir, "plot.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
