import numpy as np

from processing.simulation.base import BaseSimulation


class NetworkCascadeSimulation(BaseSimulation):
    def __init__(self, N=40, connectivity=0.1, shock_scale=0.5, decay=0.1, **kwargs):
        super().__init__(**kwargs)
        self.N = N
        self.connectivity = connectivity
        self.shock_scale = shock_scale
        self.decay = decay

    def build_random_graph(self):
        A = (np.random.rand(self.N, self.N) < self.connectivity).astype(float)
        np.fill_diagonal(A, 0.0)

        eigvals = np.linalg.eigvals(A)
        spectral_radius = np.max(np.abs(eigvals))
        if spectral_radius > 0:
            A = A / spectral_radius

        return A

    def cascade_step(self, x, A, amp):
        shock = self.shock_scale * np.random.randn(len(x))
        transport = np.tanh(A @ x)
        x_next = (1 - self.decay) * x + amp * transport + shock
        return x_next

    def simulate(self, A, amp, T):
        x = np.zeros(self.N)
        trajectory = np.zeros((T, self.N))

        for t in range(T):
            x = self.cascade_step(x, A, amp)
            trajectory[t] = x

        return trajectory

    def generate_dataset(
        self,
        amp_train=0.8,
        amp_test=0.98,
        T=2000,
        burn_in=500
    ):
        A = self.build_random_graph()  # ← single shared graph

        train = self.simulate(A, amp_train, T)[burn_in:]
        test = self.simulate(A, amp_test, T)[burn_in:]

        metadata = {
            "amp_train": amp_train,
            "amp_test": amp_test,
            "train_label": f"amp={amp_train}",
            "test_label": f"amp={amp_test}",
            "N": self.N,
            "connectivity": self.connectivity,
            "shock_scale": self.shock_scale,
            "decay": self.decay,
            "T": T,
            "burn_in": burn_in,
            "seed": self.seed,
        }

        return train, test, metadata


if __name__ == "__main__":
    sim = NetworkCascadeSimulation(output_dir="data/network_cascade")
    sim.run()