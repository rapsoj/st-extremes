import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

from processing.simulation.base import BaseSimulation

class Lorenz96Simulation(BaseSimulation):
    def __init__(self, N=40, dt=0.01, **kwargs):
        super().__init__(**kwargs)
        self.N = N
        self.dt = dt

    def lorenz96(self, x, t, F):
        return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + F

    def simulate(self, F, T):
        t = np.arange(0.0, T * self.dt, self.dt)
        x0 = F * np.ones(self.N)
        x0[0] += 0.01
        return odeint(self.lorenz96, x0, t, args=(F,))

    def generate_dataset(self, F_train=8.0, F_test=10.0, T=2000, burn_in=500):
        train = self.simulate(F_train, T)[burn_in:]
        test = self.simulate(F_test, T)[burn_in:]

        metadata = {
            "F_train": F_train,
            "F_test": F_test,
            "train_label": f"F={F_train}",
            "test_label": f"F={F_test}",
            "N": self.N,
            "dt": self.dt,
            "T": T,
            "burn_in": burn_in,
            "seed": self.seed,
        }

        return train, test, metadata

if __name__ == "__main__":
    sim = Lorenz96Simulation(output_dir="data/lorenz96")
    sim.run()