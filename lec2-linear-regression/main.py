import data
import linear_regr
import numpy as np
import matplotlib.pyplot as plt
import random


def main(cnt, alpha, epoch):
    ld = data.LinearData(5, 10, 20, [0, 100])
    lr = linear_regr.LinearRegression(*ld.linear_data(50))
    plt.plot(ld.x, ld.y, "o")

    # Test Least Square Method.
    (k, b), error = lr.least_square_method()
    x = [np.min(ld.x), np.max(ld.x)]
    y = [k * i + b for i in x]
    print(f"LSM: y={k:.2f}x+{b:.2f}, {error=:.2f}")
    plt.plot(x, y, label=f"LSM: {error=:.2f}")

    # Test Gradient Descent
    (k, b), error = lr.gradient_descent(alpha, epoch)
    y = [k * i + b for i in x]
    print(f"GD:  y={k:.2f}x+{b:.2f}, {error=:.2f}")
    plt.plot(x, y, "--", label=f"GD: {error=:.2f}")

    plt.title(f"{alpha=}, {epoch=}")
    plt.legend(loc=2)
    plt.savefig(f"results/test_result_{cnt}.jpg")
    plt.close()


if __name__ == "__main__":
    for i in range(100):
        alpha = random.random() / 1e4
        main(i + 1, alpha, 1000)
