"""
隶属函数库

常用隶属函数: 三角形、梯形、高斯型、S型、Z型。

Example:
    >>> from mathflow.fuzzy import MembershipFunction
    >>> mf = MembershipFunction()
    >>> y = mf.triangle(x, a=1, b=3, c=5)
    >>> mf.plot_all()
"""

import numpy as np


class MembershipFunction:
    """隶属函数库."""

    @staticmethod
    def triangle(x, a, b, c):
        """三角形隶属函数."""
        x = np.asarray(x, dtype=float)
        result = np.zeros_like(x)
        if abs(b - a) < 1e-10 and abs(c - b) < 1e-10:  # a==b==c
            result[np.abs(x - a) < 1e-10] = 1.0
        elif abs(b - a) < 1e-10:  # a==b, right triangle
            mask = (x >= b) & (x <= c)
            result[mask] = (c - x[mask]) / (c - b + 1e-10)
        elif abs(c - b) < 1e-10:  # b==c, left triangle
            mask = (x >= a) & (x <= b)
            result[mask] = (x[mask] - a) / (b - a + 1e-10)
        else:  # normal case
            mask = (x >= a) & (x <= c)
            result[mask] = np.maximum(0, np.minimum(
                (x[mask] - a) / (b - a),
                (c - x[mask]) / (c - b)
            ))
        return result

    @staticmethod
    def trapezoid(x, a, b, c, d):
        """梯形隶属函数."""
        x = np.asarray(x, dtype=float)
        return np.maximum(0, np.minimum(
            np.minimum((x - a) / (b - a + 1e-10), 1),
            (d - x) / (d - c + 1e-10)
        ))

    @staticmethod
    def gaussian(x, mean, sigma):
        """高斯型隶属函数."""
        x = np.asarray(x, dtype=float)
        return np.exp(-0.5 * ((x - mean) / sigma) ** 2)

    @staticmethod
    def sigmoid(x, a, c):
        """S型隶属函数."""
        x = np.asarray(x, dtype=float)
        return 1 / (1 + np.exp(-a * (x - c)))

    @staticmethod
    def z_shape(x, a, b):
        """Z型隶属函数."""
        x = np.asarray(x, dtype=float)
        result = np.ones_like(x)
        mask1 = (x >= a) & (x <= (a + b) / 2)
        mask2 = (x > (a + b) / 2) & (x <= b)
        result[mask1] = 1 - 2 * ((x[mask1] - a) / (b - a)) ** 2
        result[mask2] = 2 * ((x[mask2] - b) / (b - a)) ** 2
        result[x > b] = 0
        return result

    @staticmethod
    def s_shape(x, a, b):
        """S型隶属函数."""
        x = np.asarray(x, dtype=float)
        result = np.zeros_like(x)
        mask1 = (x >= a) & (x <= (a + b) / 2)
        mask2 = (x > (a + b) / 2) & (x <= b)
        result[mask1] = 2 * ((x[mask1] - a) / (b - a)) ** 2
        result[mask2] = 1 - 2 * ((x[mask2] - b) / (b - a)) ** 2
        result[x > b] = 1
        return result

    @staticmethod
    def bell(x, a, b, c):
        """广义钟形隶属函数."""
        x = np.asarray(x, dtype=float)
        return 1 / (1 + np.abs((x - c) / a) ** (2 * b))

    def plot_all(self, x_range=(0, 10), figsize=(12, 8)):
        """绘制所有常用隶属函数."""
        import matplotlib.pyplot as plt

        x = np.linspace(x_range[0], x_range[1], 200)
        fig, axes = plt.subplots(2, 3, figsize=figsize)

        ax = axes[0, 0]
        ax.plot(x, self.triangle(x, 2, 5, 8), "b-", linewidth=2)
        ax.set_title("三角形 (a=2, b=5, c=8)")
        ax.set_ylim(-0.1, 1.1)
        ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        ax.plot(x, self.trapezoid(x, 1, 3, 7, 9), "r-", linewidth=2)
        ax.set_title("梯形 (a=1, b=3, c=7, d=9)")
        ax.set_ylim(-0.1, 1.1)
        ax.grid(True, alpha=0.3)

        ax = axes[0, 2]
        ax.plot(x, self.gaussian(x, 5, 1), "g-", linewidth=2)
        ax.set_title("高斯型 (μ=5, σ=1)")
        ax.set_ylim(-0.1, 1.1)
        ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        ax.plot(x, self.sigmoid(x, 2, 5), "m-", linewidth=2)
        ax.set_title("S型 (a=2, c=5)")
        ax.set_ylim(-0.1, 1.1)
        ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        ax.plot(x, self.bell(x, 2, 4, 5), "c-", linewidth=2)
        ax.set_title("钟形 (a=2, b=4, c=5)")
        ax.set_ylim(-0.1, 1.1)
        ax.grid(True, alpha=0.3)

        ax = axes[1, 2]
        ax.plot(x, self.s_shape(x, 2, 8), "orange", linewidth=2)
        ax.plot(x, self.z_shape(x, 2, 8), "purple", linewidth=2, linestyle="--")
        ax.set_title("S型 / Z型")
        ax.set_ylim(-0.1, 1.1)
        ax.legend(["S型", "Z型"])
        ax.grid(True, alpha=0.3)

        plt.suptitle("常用隶属函数", fontsize=14, fontweight="bold")
        plt.tight_layout()
        return fig
