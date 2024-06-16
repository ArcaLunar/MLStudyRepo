import numpy as np
from typing import Tuple


class LinearRegression:
    def __init__(self, x: np.ndarray, y: np.ndarray):
        # Load Data
        self.x = x
        self.y = y

    def gradient_descent(self, alpha: float, epoch: int = 100):
        """
        Learn the best fitting line, using gradient descent method

        Args:
                        alpha: Learning rate
                        epoch: number of iterations, default is 100

        Returns:
                        (k, b), error: parameters, error
        """
        k, b = 0.0, 0.0
        n = float(len(self.x))
        for i in range(epoch):
            y_pred = self.x * k + b
            d_k = -sum(self.x * (self.y - y_pred)) / n
            d_b = -sum(self.y - y_pred) / n
            k = k - alpha * d_k
            b = b - alpha * d_b

        return (k, b), self.measure_acc((k, b))

    def least_square_method(self):
        """
        Fit dataset with least square method.

        Return:
                        k, b, error: parameters, error
        """
        X = np.array([[self.x[i], 1] for i in range(len(self.x))], dtype=float)

        # beta = (X^T X)^{-1} X^T y
        beta = np.linalg.inv(X.transpose() @ X) @ X.transpose() @ self.y

        # Returns (k, b)
        return beta, self.measure_acc(beta)

    def measure_acc(self, beta):
        """
        Measures the (minimized) cost function.

        Args:
                        beta: fitting line coefficients. (K, B)
        """
        return sum((self.y - self.x * beta[0] - beta[1]) ** 2) * 0.5


# Test function.
if __name__ == "__main__":
    x = np.array([2, 4, 6], dtype=float)
    y = np.array([6, 8, 10], dtype=float)
    lr = LinearRegression(x, y)
    print(lr.least_square_method())
