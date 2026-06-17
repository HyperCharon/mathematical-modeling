"""
指数平滑法 (Exponential Smoothing)

包括一次、二次 (Holt)、三次 (Holt-Winters) 指数平滑。
适用于短期预测，对近期数据赋予更大权重。

Example:
    >>> from mathflow.predict import ExponentialSmoothing
    >>> data = [10, 12, 13, 15, 16, 18, 20, 22, 24, 26]
    >>> es = ExponentialSmoothing(data, method="holt", alpha=0.3, beta=0.1)
    >>> es.fit()
    >>> forecast = es.predict(steps=5)
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class SmoothResult:
    """指数平滑结果."""
    method: str
    level: np.ndarray        # 水平分量
    trend: np.ndarray        # 趋势分量 (Holt/Holt-Winters)
    seasonal: np.ndarray     # 季节分量 (Holt-Winters)
    fitted_values: np.ndarray
    residuals: np.ndarray


class ExponentialSmoothing:
    """
    指数平滑法.

    Parameters
    ----------
    data : array-like
        时间序列
    method : str
        "simple" (一次), "holt" (二次/Holt线性), "holt_winters" (三次)
    alpha : float
        水平平滑系数 (0, 1)
    beta : float
        趋势平滑系数 (0, 1)，仅 holt/holt_winters
    gamma : float
        季节平滑系数 (0, 1)，仅 holt_winters
    seasonal_periods : int
        季节周期，仅 holt_winters
    seasonal_type : str
        "additive" (加法) 或 "multiplicative" (乘法)
    """

    def __init__(self, data, method="holt", alpha=0.3, beta=0.1, gamma=0.1,
                 seasonal_periods=12, seasonal_type="additive"):
        self.data = np.asarray(data, dtype=float).flatten()
        self.method = method
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.seasonal_periods = seasonal_periods
        self.seasonal_type = seasonal_type
        self._result = None

    def __repr__(self) -> str:
        if self._result is not None:
            return f"ExponentialSmoothing(n={len(self.data)}, method={self.method!r})"
        return f"ExponentialSmoothing(n={len(self.data)}, method={self.method!r})"

    def fit(self):
        """拟合指数平滑模型."""
        if self.method == "simple":
            self._fit_simple()
        elif self.method == "holt":
            self._fit_holt()
        elif self.method == "holt_winters":
            self._fit_holt_winters()
        else:
            raise ValueError(f"未知方法: {self.method}")
        return self

    def _fit_simple(self):
        """一次指数平滑."""
        data = self.data
        n = len(data)
        alpha = self.alpha

        level = np.zeros(n)
        level[0] = data[0]
        for t in range(1, n):
            level[t] = alpha * data[t] + (1 - alpha) * level[t - 1]

        fitted = np.zeros(n)
        fitted[0] = data[0]
        fitted[1:] = level[:-1]

        self._result = SmoothResult(
            method="simple", level=level,
            trend=np.zeros(0), seasonal=np.zeros(0),
            fitted_values=fitted, residuals=data - fitted,
        )

    def _fit_holt(self):
        """Holt 线性趋势法 (二次指数平滑)."""
        data = self.data
        n = len(data)
        alpha, beta = self.alpha, self.beta

        level = np.zeros(n)
        trend = np.zeros(n)

        level[0] = data[0]
        trend[0] = data[1] - data[0] if n > 1 else 0

        for t in range(1, n):
            level[t] = alpha * data[t] + (1 - alpha) * (level[t - 1] + trend[t - 1])
            trend[t] = beta * (level[t] - level[t - 1]) + (1 - beta) * trend[t - 1]

        fitted = np.zeros(n)
        fitted[0] = data[0]
        fitted[1:] = level[:-1] + trend[:-1]

        self._result = SmoothResult(
            method="holt", level=level, trend=trend,
            seasonal=np.zeros(0),
            fitted_values=fitted, residuals=data - fitted,
        )

    def _fit_holt_winters(self):
        """Holt-Winters 季节法 (三次指数平滑)."""
        data = self.data
        n = len(data)
        alpha, beta, gamma = self.alpha, self.beta, self.gamma
        m = self.seasonal_periods

        if n < 2 * m:
            raise ValueError(f"数据长度 ({n}) 至少需要 2 × 季节周期 ({m})")

        level = np.zeros(n)
        trend = np.zeros(n)
        seasonal = np.zeros(n + m)  # 多预留一些用于预测

        # 初始化
        level[0] = np.mean(data[:m])
        trend[0] = (np.mean(data[m:2 * m]) - np.mean(data[:m])) / m

        if self.seasonal_type == "additive":
            for i in range(m):
                seasonal[i] = data[i] - level[0]
            for t in range(1, n):
                level[t] = alpha * (data[t] - seasonal[t - 1 + m]) + (1 - alpha) * (level[t - 1] + trend[t - 1])
                trend[t] = beta * (level[t] - level[t - 1]) + (1 - beta) * trend[t - 1]
                seasonal[t + m - 1] = gamma * (data[t] - level[t]) + (1 - gamma) * seasonal[t - 1]
        else:  # multiplicative
            for i in range(m):
                seasonal[i] = data[i] / level[0] if level[0] != 0 else 1
            for t in range(1, n):
                level[t] = alpha * (data[t] / seasonal[t - 1 + m]) + (1 - alpha) * (level[t - 1] + trend[t - 1])
                trend[t] = beta * (level[t] - level[t - 1]) + (1 - beta) * trend[t - 1]
                seasonal[t + m - 1] = gamma * (data[t] / level[t]) + (1 - gamma) * seasonal[t - 1]

        # 拟合值
        fitted = np.zeros(n)
        if self.seasonal_type == "additive":
            fitted[0] = data[0]
            fitted[1:] = level[:-1] + trend[:-1] + seasonal[m:n + m - 1]
        else:
            fitted[0] = data[0]
            fitted[1:] = (level[:-1] + trend[:-1]) * seasonal[m:n + m - 1]

        self._result = SmoothResult(
            method="holt_winters", level=level, trend=trend,
            seasonal=seasonal[:n + m],
            fitted_values=fitted, residuals=data - fitted,
        )

    def predict(self, steps=5):
        """预测未来值."""
        self._ensure_fitted()
        r = self._result
        n = len(self.data)

        if r.method == "simple":
            return np.full(steps, r.level[-1])

        elif r.method == "holt":
            forecasts = np.zeros(steps)
            for h in range(1, steps + 1):
                forecasts[h - 1] = r.level[-1] + h * r.trend[-1]
            return forecasts

        else:  # holt_winters
            m = self.seasonal_periods
            forecasts = np.zeros(steps)
            for h in range(1, steps + 1):
                seasonal_idx = (n + h - 1) % m
                if self.seasonal_type == "additive":
                    forecasts[h - 1] = r.level[-1] + h * r.trend[-1] + r.seasonal[n + seasonal_idx - 1]
                else:
                    forecasts[h - 1] = (r.level[-1] + h * r.trend[-1]) * r.seasonal[n + seasonal_idx - 1]
            return forecasts

    def _ensure_fitted(self):
        if self._result is None:
            raise RuntimeError("请先调用 fit() 进行计算")

    def plot(self, steps=5, figsize=(10, 5)):
        """绘制拟合和预测曲线."""
        import matplotlib.pyplot as plt

        self._ensure_fitted()
        r = self._result
        n = len(self.data)
        forecasts = self.predict(steps)

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(range(n), self.data, "b-o", label="实际值", markersize=4)
        ax.plot(range(n), r.fitted_values, "r--s", label="拟合值", markersize=3, alpha=0.7)
        ax.plot(range(n, n + steps), forecasts, "g--^", label=f"预测值 ({steps}步)", markersize=6)
        ax.set_title(f"{r.method.replace('_', ' ').title()} 指数平滑预测")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    def summary(self, steps=5):
        """打印结果摘要."""
        self._ensure_fitted()
        r = self._result
        forecasts = self.predict(steps)
        rmse = np.sqrt((r.residuals ** 2).mean())

        lines = [
            "=" * 50,
            f"  {r.method.replace('_', ' ').title()} 指数平滑结果",
            "=" * 50,
            f"  RMSE = {rmse:.4f}",
            f"  α = {self.alpha}",
        ]
        if r.method in ("holt", "holt_winters"):
            lines.append(f"  β = {self.beta}")
        if r.method == "holt_winters":
            lines.append(f"  γ = {self.gamma}")
            lines.append(f"  季节周期 = {self.seasonal_periods}")
        lines.append("-" * 50)
        lines.append(f"  预测 (未来{steps}步):")
        for i, f in enumerate(forecasts):
            lines.append(f"    t={len(self.data)+i+1}: {f:.2f}")
        lines.append("=" * 50)
        return "\n".join(lines)
