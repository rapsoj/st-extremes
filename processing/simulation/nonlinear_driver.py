import numpy as np

from processing.simulation.base import BaseSimulation


class NonlinearDriverSimulation(BaseSimulation):
    def __init__(
        self,
        N=40,
        coupling=0.5,
        lag=1,
        threshold=1.0,
        noise_scale=0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.N = N
        self.coupling = coupling
        self.lag = lag
        self.threshold = threshold
        self.noise_scale = noise_scale

    def generate_driver(self, T, scale=1.0):
        """
        Exogenous driver with controllable support.
        """
        return scale * np.random.randn(T, self.N)

    def nonlinear_response(self, driver_t):
        """
        Nonlinear, thresholded coupling.
        """
        return np.tanh(driver_t) * (np.abs(driver_t) > self.threshold)

    def simulate(self, driver, T):
        x = np.zeros(self.N)
        trajectory = np.zeros((T, self.N))

        for t in range(T):
            d_t = driver[max(0, t - self.lag)]

            response = self.nonlinear_response(d_t)

            noise = self.noise_scale * np.random.randn(self.N)

            x = (
                (1 - self.coupling) * x
                + self.coupling * response
                + noise
            )

            trajectory[t] = x

        return trajectory

    def generate_dataset(
        self,
        T=2000,
        burn_in=500,
        driver_scale_train=1.0,
        driver_scale_test=2.0
    ):
        # Same structure, different support
        driver = self.generate_driver(T, scale=1.0)

        train = self.simulate(driver_scale_train * driver, T)[burn_in:]
        test  = self.simulate(driver_scale_test  * driver, T)[burn_in:]

        metadata = {
            "driver_scale_train": driver_scale_train,
            "driver_scale_test": driver_scale_test,
            "N": self.N,
            "coupling": self.coupling,
            "lag": self.lag,
            "threshold": self.threshold,
            "noise_scale": self.noise_scale,
            "T": T,
            "burn_in": burn_in,
            "seed": self.seed,
            "train_label": f"driver_scale={driver_scale_train}",
            "test_label": f"driver_scale={driver_scale_test}",
        }

        return train, test, metadata


if __name__ == "__main__":
    sim = NonlinearDriverSimulation(output_dir="data/nonlinear_driver")
    sim.run()