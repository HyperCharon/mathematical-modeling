"""
曲线拟合

支持多种常见函数形式的拟合: 线性、多项式、指数、对数、幂函数、Logistic。
自动选择最优拟合函数。

Example:
    >>> from mathflow.interpolate import CurveFitter
    >>> import numpy as np
    >>> x = np.array([1,2,3,4,5,6,7,8,9,10])
    >>> y = np.array([2.1, 3.9, 8.2, 15.8, 30.1, 58.3, 110.8, 210.2, 400.1, 760.5])
    >>> cf = CurveFitter(x, y)
    >>> result = cf.auto_fit()
    >>> print(cf.summary())
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy import stats
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class FitResult:
    """拟合结果."""
    func_name: str
    params: np.ndarray
    param_names: list
    r2: float
    rmse: float
    fitted_values: np.ndarray
    func: Callable
    equation: str


# 预定义函数
MODELS = {
    "linear": {
        "func": lambda x, a, b: a * x + b,
        "p0": [1, 0],
        "param_names": ["a", "b"],
        "equation": "y = a·x + b",
    },
    "polynomial2": {
        "func": lambda x, a, b, c: a * x**2 + b * x + c,
        "p0": [1, 1, 0],
        "param_names": ["a", "b", "c"],
        "equation": "y = a·x² + b·x + c",
    },
    "polynomial3": {
        "func": lambda x, a, b, c, d: a * x**3 + b * x**2 + c * x + d,
        "p0": [1, 1, 1, 0],
        "param_names": ["a", "b", "c", "d"],
        "equation": "y = a·x³ + b·x² + c·x + d",
    },
    "exponential": {
        "func": lambda x, a, b: a * np.exp(b * x),
        "p0": [1, 0.1],
        "param_names": ["a", "b"],
        "equation": "y = a·e^(b·x)",
    },
    "logarithmic": {
        "func": lambda x, a, b: a * np.log(x) + b,
        "p0": [1, 0],
        "param_names": ["a", "b"],
        "equation": "y = a·ln(x) + b",
    },
    "power": {
        "func": lambda x, a, b: a * x**b,
        "p0": [1, 1],
        "param_names": ["a", "b"],
        "equation": "y = a·x^b",
    },
    "logistic": {
        "func": lambda x, L, k, x0: L / (1 + np.exp(-k * (x - x0))),
        "p0": [None, 1, 0],  # 动态设置
        "param_names": ["L", "k", "x0"],
        "equation": "y = L / (1 + e^(-k·(x-x0)))",
    },
}


class CurveFitter:
    """
    曲线拟合器.

    Parameters
    ----------
    x, y : array-like
        数据点
    """

    def __init__(self, x, y):
        self.x = np.asarray(x, dtype=float).flatten()
        self.y = np.asarray(y, dtype=float).flatten()
        if len(self.x) != len(self.y):
            raise ValueError("x 和 y 长度必须一致")
        self._result = None

    def __repr__(self) -> str:
        if self._result is not None:
            return f"CurveFitter(n_points={len(self.x)}, best_model={self._result.func_name!r})"
        return f"CurveFitter(n_points={len(self.x)})"

    def fit(self, model_name="linear", **kwargs):
        """
        拟合指定模型.

        Parameters
        ----------
        model_name : str
            模型名称: "linear", "polynomial2", "polynomial3", "exponential",
            "logarithmic", "power", "logistic"
        """
        if model_name not in MODELS:
            raise ValueError(f"未知模型: {model_name}, 可选: {list(MODELS.keys())}")

        model = MODELS[model_name]
        func = model["func"]
        p0 = model["p0"]

        # 动态设置 logistic 初始值
        if model_name == "logistic" and p0[0] is None:
            p0 = [self.y.max(), 1, np.median(self.x)]

        try:
            popt, _ = curve_fit(func, self.x, self.y, p0=p0, maxfev=10000)
        except RuntimeError as e:
            raise RuntimeError(f"拟合失败: {e}")

        y_hat = func(self.x, *popt)
        ss_res = np.sum((self.y - y_hat)**2)
        ss_tot = np.sum((self.y - self.y.mean())**2)

        if ss_tot < 1e-10:
            # 常数数据，R2 无意义
            r2 = 0.0
        else:
            r2 = 1 - ss_res / ss_tot

        rmse = np.sqrt(np.mean((self.y - y_hat)**2))

        if r2 < -1:
            import warnings
            warnings.warn(f"{model_name} 拟合失败: R²={r2:.4f}，建议使用其他模型", UserWarning)

        # 构建方程字符串
        equation = model["equation"]
        for name, val in zip(model["param_names"], popt):
            if abs(val) < 1e-8:
                # skip near-zero coefficients in display
                equation = equation.replace(f"{name}·", "").replace(f"+ {name}", "").replace(f"- {name}", "").replace(name, f"{val:.4f}")
            else:
                equation = equation.replace(name, f"{val:.4f}")

        self._result = FitResult(
            func_name=model_name,
            params=popt,
            param_names=model["param_names"],
            r2=r2, rmse=rmse,
            fitted_values=y_hat,
            func=func,
            equation=equation,
        )
        return self

    def auto_fit(self, models=None):
        """
        自动选择最优拟合函数.

        尝试所有模型，选择 R² 最高的。
        """
        if models is None:
            models = ["linear", "polynomial2", "polynomial3", "exponential",
                       "logarithmic", "power"]

        best_r2 = -float("inf")
        best_result = None

        for name in models:
            try:
                self.fit(name)
                if self._result.r2 > best_r2:
                    best_r2 = self._result.r2
                    best_result = self._result
            except Exception:
                continue

        if best_result is None:
            raise RuntimeError("所有模型拟合失败")

        self._result = best_result
        return self

    def predict(self, x_new):
        """预测."""
        self._ensure_fitted()
        return self._result.func(np.asarray(x_new), *self._result.params)

    @property
    def result(self):
        self._ensure_fitted()
        return self._result

    def _ensure_fitted(self):
        if self._result is None:
            raise RuntimeError("请先调用 fit() 或 auto_fit()")

    def plot(self, figsize=(10, 6)):
        """绘制拟合曲线."""
        import matplotlib.pyplot as plt

        self._ensure_fitted()
        r = self._result

        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(self.x, self.y, c="blue", s=40, label="数据点", zorder=5)

        x_smooth = np.linspace(self.x.min(), self.x.max(), 200)
        y_smooth = r.func(x_smooth, *r.params)
        ax.plot(x_smooth, y_smooth, "r-", linewidth=2, label=f"拟合: {r.func_name}")

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"曲线拟合 ({r.func_name})\n{r.equation}\nR²={r.r2:.4f}, RMSE={r.rmse:.4f}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    def summary(self):
        self._ensure_fitted()
        r = self._result
        lines = [
            "=" * 60,
            f"  曲线拟合结果 ({r.func_name})",
            "=" * 60,
            f"  方程: {r.equation}",
            f"  R² = {r.r2:.6f}",
            f"  RMSE = {r.rmse:.6f}",
            "-" * 60,
            "  参数:",
        ]
        for name, val in zip(r.param_names, r.params):
            lines.append(f"    {name} = {val:.6f}")
        lines.append("=" * 60)
        return "\n".join(lines)
