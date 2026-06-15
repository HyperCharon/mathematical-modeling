"""
时间序列分解

将时间序列分解为趋势、季节性和残差分量。

Example:
    >>> from mathflow.timeseries import TimeSeriesDecompose
    >>> import numpy as np
    >>> t = np.arange(100)
    >>> data = 2*t + 10*np.sin(2*np.pi*t/12) + np.random.randn(100)*2
    >>> ts = TimeSeriesDecompose(data, period=12)
    >>> result = ts.decompose()
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class DecomposeResult:
    """分解结果."""
    trend: np.ndarray
    seasonal: np.ndarray
    residual: np.ndarray
    observed: np.ndarray
    period: int
    method: str


class TimeSeriesDecompose:
    """
    时间序列分解.

    Parameters
    ----------
    data : array-like
        时间序列数据
    period : int
        季节周期 (如月度数据=12, 季度数据=4)
    model : str
        "additive" (加法) 或 "multiplicative" (乘法)
    """

    def __init__(self, data, period=12, model="additive"):
        self.data = np.asarray(data, dtype=float)
        self.period = period
        self.model = model
        self._result = None

    def decompose(self, method="moving_average") -> DecomposeResult:
        """
        执行分解.

        Parameters
        ----------
        method : str
            "moving_average" (移动平均), "stl" (简化STL)
        """
        if method == "moving_average":
            return self._moving_average_decompose()
        elif method == "stl":
            return self._stl_decompose()
        else:
            raise ValueError(f"未知方法: {method}")

    def _moving_average_decompose(self):
        """移动平均分解."""
        data = self.data
        n = len(data)
        p = self.period
        if p > n:
            raise ValueError(f"季节周期 ({p}) 不能大于数据长度 ({n})")

        # Step 1: 趋势 (中心化移动平均)
        if self.model == "additive":
            # 先去季节性
            detrended = data.copy()
        else:
            detrended = data.copy()

        # 移动平均提取趋势
        trend = np.full(n, np.nan)
        half_p = p // 2

        for i in range(half_p, n - half_p):
            if p % 2 == 0:
                trend[i] = (0.5 * data[i - half_p] + data[i - half_p + 1:i + half_p].sum() +
                            0.5 * data[i + half_p]) / p
            else:
                trend[i] = data[i - half_p:i + half_p + 1].mean()

        # 填充首尾
        for i in range(half_p):
            trend[i] = trend[half_p]
        for i in range(n - half_p, n):
            trend[i] = trend[n - half_p - 1]

        # Step 2: 去趋势
        if self.model == "additive":
            detrended = data - trend
        else:
            detrended = data / np.where(trend != 0, trend, 1)

        # Step 3: 季节性 (同期平均)
        seasonal = np.zeros(n)
        seasonal_means = np.zeros(p)
        for j in range(p):
            indices = np.arange(j, n, p)
            seasonal_means[j] = detrended[indices].mean()

        # 归一化 (加法模型均值为0, 乘法模型均值为1)
        if self.model == "additive":
            seasonal_means -= seasonal_means.mean()
        else:
            seasonal_means /= seasonal_means.mean()

        for i in range(n):
            seasonal[i] = seasonal_means[i % p]

        # Step 4: 残差
        if self.model == "additive":
            residual = data - trend - seasonal
        else:
            residual = data / (trend * seasonal)

        self._result = DecomposeResult(
            trend=trend, seasonal=seasonal, residual=residual,
            observed=data, period=p, method="moving_average"
        )
        return self._result

    def _stl_decompose(self):
        """简化 STL 分解 (迭代平滑)."""
        data = self.data
        n = len(data)
        p = self.period

        # 初始趋势估计
        trend = self._lowess_smooth(data, window=p + 1)

        for _ in range(3):  # 迭代3次
            # 去趋势
            if self.model == "additive":
                detrended = data - trend
            else:
                detrended = data / np.where(trend != 0, trend, 1)

            # 季节性
            seasonal_means = np.zeros(p)
            for j in range(p):
                indices = np.arange(j, n, p)
                seasonal_means[j] = detrended[indices].mean()

            if self.model == "additive":
                seasonal_means -= seasonal_means.mean()
            else:
                seasonal_means /= seasonal_means.mean()

            seasonal = np.array([seasonal_means[i % p] for i in range(n)])

            # 去季节性
            if self.model == "additive":
                deseason = data - seasonal
            else:
                deseason = data / np.where(seasonal != 0, seasonal, 1)

            # 更新趋势
            trend = self._lowess_smooth(deseason, window=p + 1)

        # 残差
        if self.model == "additive":
            residual = data - trend - seasonal
        else:
            residual = data / (trend * seasonal)

        self._result = DecomposeResult(
            trend=trend, seasonal=seasonal, residual=residual,
            observed=data, period=p, method="stl"
        )
        return self._result

    def _lowess_smooth(self, data, window=13):
        """简单移动平均平滑."""
        n = len(data)
        smoothed = np.zeros(n)
        half = window // 2
        for i in range(n):
            start = max(0, i - half)
            end = min(n, i + half + 1)
            smoothed[i] = data[start:end].mean()
        return smoothed

    @property
    def result(self):
        if self._result is None:
            raise RuntimeError("请先调用 decompose()")
        return self._result

    def plot(self, figsize=(12, 10)):
        """绘制分解结果."""
        import matplotlib.pyplot as plt

        r = self.result
        fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)

        axes[0].plot(r.observed, "b-", linewidth=1)
        axes[0].set_ylabel("观测值")
        axes[0].set_title("时间序列分解")

        axes[1].plot(r.trend, "r-", linewidth=1.5)
        axes[1].set_ylabel("趋势")

        axes[2].plot(r.seasonal, "g-", linewidth=1)
        axes[2].set_ylabel("季节性")

        axes[3].plot(r.residual, "gray", linewidth=0.8, alpha=0.7)
        axes[3].axhline(y=0, color="red", linestyle="--", linewidth=0.5)
        axes[3].set_ylabel("残差")
        axes[3].set_xlabel("时间")

        plt.tight_layout()
        return fig

    def summary(self):
        r = self.result
        lines = [
            "=" * 50,
            "  时间序列分解结果",
            "=" * 50,
            f"  方法: {r.method}",
            f"  模型: {self.model}",
            f"  周期: {r.period}",
            "-" * 50,
            f"  趋势范围: [{r.trend.min():.2f}, {r.trend.max():.2f}]",
            f"  季节性范围: [{r.seasonal.min():.2f}, {r.seasonal.max():.2f}]",
            f"  残差标准差: {r.residual.std():.4f}",
            f"  残差均值: {r.residual.mean():.4f}",
            "=" * 50,
        ]
        return "\n".join(lines)
