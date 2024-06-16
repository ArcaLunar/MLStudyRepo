import numpy as np
import matplotlib.pyplot as plt


class LinearData:
    def __init__(self, k, b, scalar, rng):
        self.k = k
        self.b = b
        self.scalar = scalar
        self.rng = rng

    def linear_data(self, n: int):
        x = np.array(
            [
                np.random.randint(self.rng[0], self.rng[1]) + np.random.random()
                for i in range(n)
            ],
            dtype=float,
        )

        y = np.array([0] * n, dtype=float)
        for i in range(n):
            y[i] = self.k * x[i] + self.b + np.random.random() * self.scalar

        self.x = x
        self.y = y
        return x, y

    def plot_data(self):
        plt.plot(self.x, self.y, "o")
        plt.savefig("data.jpg")


if __name__ == "__main__":
    ld = LinearData(5, 10, 20, [0, 100])
    ld.linear_data(50)
    ld.plot_data()
