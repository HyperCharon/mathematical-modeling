"""
ARIMA 时间序列预测模型

自回归积分移动平均模型，适用于平稳或差分后平稳的时间序列。

Example:
    >>> from mathflow.predict import ARIMAModel
    >>> import numpy as np
    >>> data = np.random.randn(100).cumsum() + 50
    >>> model = ARIMAModel(data)
    >>> model.auto_fit(max_p=3, max_d=2, max_q=3)
    >>> forecast = model.predict(steps=10)
    >>> model.plot()
"""

import numpy as np
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

try:
    from statsmodels.tsa.arima.model import ARIMA as _ARIMA
    from statsmodels.tsa.stattools import adfuller
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False


@dataclass
class ARIMAResult:
    """ARIMA 模型结果."""
    order: Tuple[int, int, int]   # (p, d, q)
    aic: float
    bic: float
    fitted_values: np.ndarray
    residuals: np.ndarray
    summary_text: str


class ARIMAModel:
    """
    ARIMA 时间序列预测.

    Parameters
    ----------
    data : array-like
        时间序列数据
    """

    def __init__(self, data):
        if not HAS_STATSMODELS:
            raise ImportError("ARIMA 需要安装 statsmodels: pip install statsmodels")
        self.data = np.asarray(data, dtype=float).flatten()
        self._model = None
        self._result = None
        self._order = None

    def fit(self, order=(1, 1, 1)):
        """
        拟合 ARIMA(p, d, q) 模型.

        Parameters
        ----------
        order : tuple (p, d, q)
            p: 自回归阶数, d: 差分阶数, q: 移动平均阶数
        """
        self._order = order
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model = _ARIMA(self.data, order=order)
            self._result = self._model.fit()

        self._arima_result = ARIMAResult(
            order=order,
            aic=self._result.aic,
            bic=self._result.bic,
            fitted_values=self._result.fittedvalues,
            residuals=self._result.resid,
            summary_text=str(self._result.summary()),
        )
        return self

    def auto_fit(self, max_p=3, max_d=2, max_q=3, criterion="aic"):
        """
        自动选择最优 ARIMA 参数 (网格搜索).

        Parameters
        ----------
        max_p, max_d, max_q : int
            各阶数的搜索上界
        criterion : str
            准则: "aic" 或 "bic"
        """
        best_score = float("inf")
        best_order = (1, 1, 1)

        # 先做 ADF 检验确定差分阶数
        adf_d = self._suggest_d(max_d)

        for p in range(max_p + 1):
            for d in range(max(0, adf_d - 1), min(max_d + 1, adf_d + 2)):
                for q in range(max_q + 1):
                    if p == 0 and q == 0:
                        continue
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            model = _ARIMA(self.data, order=(p, d, q))
                            result = model.fit()
                            score = result.aic if criterion == "aic" else result.bic
                            if score < best_score:
                                best_score = score
                                best_order = (p, d, q)
                    except Exception:
                        continue

        self.fit(order=best_order)
        return self

    def _suggest_d(self, max_d):
        """通过 ADF 检验建议差分阶数."""
        series = self.data.copy()
        for d in range(max_d + 1):
            try:
                adf_result = adfuller(series, maxlag=min(20, (len(series) - 1) // 2 - 1), autolag="AIC")
                if adf_result[1] < 0.05:  # p-value < 0.05 → 平稳
                    return d
            except Exception:
                pass
            series = np.diff(series)
        return 1

    def predict(self, steps=10):
        """预测未来值."""
        self._ensure_fitted()
        forecast = self._result.forecast(steps=steps)
        return np.asarray(forecast)

    def get_confidence_interval(self, steps=10, alpha=0.05):
        """获取预测置信区间."""
        self._ensure_fitted()
        forecast = self._result.get_forecast(steps=steps)
        ci = forecast.conf_int(alpha=alpha)
        return ci

    @property
    def order(self):
        self._ensure_fitted()
        return self._arima_result.order

    @property
    def aic(self):
        self._ensure_fitted()
        return self._arima_result.aic

    def _ensure_fitted(self):
        if self._result is None:
            raise RuntimeError("请先调用 fit() 或 auto_fit() 进行建模")

    def plot(self, steps=10, figsize=(12, 6)):
        """绘制拟合和预测结果."""
        import matplotlib.pyplot as plt

        self._ensure_fitted()
        n = len(self.data)
        forecasts = self.predict(steps)
        ci = self.get_confidence_interval(steps)

        fig, axes = plt.subplots(2, 1, figsize=figsize, gridspec_kw={"height_ratios": [3, 1]})

        # 主图
        ax = axes[0]
        ax.plot(range(n), self.data, "b-", label="实际值", alpha=0.8)
        ax.plot(range(n), self._arima_result.fitted_values[:n], "r--", label="拟合值", alpha=0.7)
        ax.plot(range(n, n + steps), forecasts, "g-", label=f"预测值 ({steps}步)", linewidth=2)
        ax.fill_between(range(n, n + steps), ci.iloc[:, 0], ci.iloc[:, 1],
                        alpha=0.2, color="green", label="95% 置信区间")
        ax.set_title(f"ARIMA{self.order} 预测 (AIC={self.aic:.2f})")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 残差图
        ax2 = axes[1]
        residuals = self._arima_result.residuals[:n]
        ax2.bar(range(n), residuals, color="gray", alpha=0.6)
        ax2.axhline(y=0, color="red", linestyle="--")
        ax2.set_xlabel("时间")
        ax2.set_ylabel("残差")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def summary(self):
        """打印模型摘要."""
        self._ensure_fitted()
        return self._arima_result.summary_text
