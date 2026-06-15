"""
插值方法

支持 Lagrange 插值、分段线性插值、三次样条插值。

Example:
    >>> from mathflow.interpolate import Interpolator
    >>> import numpy as np
    >>> x = np.array([0, 1, 2, 3, 4, 5])
    >>> y = np.array([0, 1, 4, 9, 16, 25])
    >>> interp = Interpolator(x, y)
    >>> val = interp.interpolate(2.5, method="cubic_spline")
"""

import numpy as np
from scipy.interpolate import lagrange, interp1d, CubicSpline


class Interpolator:
    """
    插值器.

    Parameters
    ----------
    x, y : array-like
        已知数据点 (x 需要单调递增)
    """

    def __init__(self, x, y):
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)

    def interpolate(self, x_new, method="cubic_spline"):
        """
        插值.

        Parameters
        ----------
        x_new : float or array-like
            待插值点
        method : str
            "lagrange", "linear", "cubic_spline"
        """
        x_new = np.atleast_1d(np.asarray(x_new, dtype=float))

        if method == "lagrange":
            poly = lagrange(self.x, self.y)
            return poly(x_new)

        elif method == "linear":
            f = interp1d(self.x, self.y, kind="linear")
            return f(x_new)

        elif method == "cubic_spline":
            cs = CubicSpline(self.x, self.y)
            return cs(x_new)

        else:
            raise ValueError(f"未知方法: {method}")

    def plot(self, n_points=200, figsize=(10, 6)):
        """绘制插值曲线."""
        import matplotlib.pyplot as plt

        x_smooth = np.linspace(self.x.min(), self.x.max(), n_points)

        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(self.x, self.y, c="red", s=60, zorder=5, label="已知数据点")

        for method, style in [("linear", "--"), ("cubic_spline", "-")]:
            y_smooth = self.interpolate(x_smooth, method=method)
            ax.plot(x_smooth, y_smooth, style, linewidth=1.5, label=method)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("插值方法对比")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig
