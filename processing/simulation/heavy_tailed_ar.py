import numpy as np

from processing.simulation.base import BaseSimulation


class HeavyTailedARSimulation(BaseSimulation):
    def __init__(
        self,
        N=40,
        phi=0.5,
        connectivity=0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.N = N
        self.phi = phi
        self.connectivity = connectivity

    def build_spatial_matrix(self):
        A = (np.random.rand(self.N, self.N) < self.connectivity).astype(float)
        np.fill_diagonal(A, 0.0)

        eigvals = np.linalg.eigvals(A)
        spectral_radius = np.max(np.abs(eigvals))

        if spectral_radius > 0:
            A = A / (2.5 * spectral_radius)  # stronger scaling

        return A

    def sample_heavy_tailed_noise(self, df):
        """
        Student-t innovations with degrees of freedom df.
        Lower df → heavier tails.
        """
        return np.random.standard_t(df, size=self.N)

    def simulate(self, A, df, T):
        x = np.zeros(self.N)
        trajectory = np.zeros((T, self.N))

        for t in range(T):
            eps = self.sample_heavy_tailed_noise(df)

            x = self.phi * x + A @ x + eps
            trajectory[t] = x

        return trajectory

    def generate_dataset(
        self,
        T=2000,
        burn_in=500,
        df_train=10.0,
        df_test=3.0
    ):
        """
        Train: lighter tails (higher df)
        Test: heavier tails (lower df)
        """
        A = self.build_spatial_matrix()

        train = self.simulate(A, df_train, T)[burn_in:]
        test = self.simulate(A, df_test, T)[burn_in:]

        metadata = {
            "df_train": df_train,
            "df_test": df_test,
            "phi": self.phi,
            "N": self.N,
            "connectivity": self.connectivity,
            "T": T,
            "burn_in": burn_in,
            "seed": self.seed,
            "train_label": f"df={df_train}",
            "test_label": f"df={df_test}",
        }

        return train, test, metadata


if __name__ == "__main__":
    sim = HeavyTailedARSimulation(output_dir="data/heavy_tailed_ar")
    sim.run()