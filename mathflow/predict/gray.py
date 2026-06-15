"""
灰色预测模型 GM(1,1)

适用于小样本 (4~15个数据点)、指数增长型序列的预测。
数模比赛中最高频的预测模型之一。

Example:
    >>> from mathflow.predict import GreyPrediction
    >>> data = [124, 130, 138, 146, 155, 165]
    >>> gp = GreyPrediction(data)
    >>> gp.fit()
    >>> forecast = gp.predict(steps=3)
    >>> gp.plot()
"""

import numpy as np
from scipy.optimize import minimize
from dataclasses import dataclass
from typing import Optional


@dataclass
class GreyResult:
    """灰色预测结果."""
    a: float            # 发展系数
    b: float            # 灰色作用量
    fitted_values: np.ndarray   # 拟合值 (还原后)
    residuals: np.ndarray       # 残差
    relative_errors: np.ndarray  # 相对误差
    posterior_ratio: float       # 后验差比值 C
    small_error_prob: float      # 小误差概率 P
    accuracy_level: str          # 精度等级


class GreyPrediction:
    """
    GM(1,1) 灰色预测模型.

    Parameters
    ----------
    data : array-like
        原始时间序列 (至少4个数据点)
    """

    def __init__(self, data):
        self.data = np.asarray(data, dtype=float).flatten()
        if len(self.data) < 4:
            raise ValueError("GM(1,1) 至少需要 4 个数据点")
        self._result = None

    def fit(self):
        """拟合 GM(1,1) 模型."""
        X0 = self.data
        n = len(X0)

        # Step 1: 累加生成序列 (AGO)
        X1 = np.cumsum(X0)

        # Step 2: 紧邻均值生成序列
        Z1 = 0.5 * (X1[:-1] + X1[1:])

        # Step 3: 构建数据矩阵 B 和常数向量 Y
        B = np.column_stack([-Z1, np.ones(n - 1)])
        Y = X0[1:]

        # Step 4: 最小二乘法求解参数 [a, b]
        params = np.linalg.lstsq(B, Y, rcond=None)[0]
        a, b = params[0], params[1]

        # Step 5: 求解时间响应函数
        # X1^(k+1) = (X0(1) - b/a) * exp(-a*k) + b/a
        if abs(a) < 1e-10:
            # a 接近 0 时退化为线性模型
            X1_hat = X1[0] + b * np.arange(n)
        else:
            c = X0[0] - b / a
            X1_hat = c * np.exp(-a * np.arange(n)) + b / a

        # Step 6: 累减还原
        X0_hat = np.zeros(n)
        X0_hat[0] = X0[0]
        X0_hat[1:] = np.diff(X1_hat)

        # Step 7: 计算残差和相对误差
        residuals = X0 - X0_hat
        relative_errors = np.abs(residuals) / np.where(X0 != 0, X0, 1)

        # Step 8: 后验差检验
        S1 = X0.std()  # 原始序列标准差
        S2 = residuals.std()  # 残差标准差
        C = S2 / S1 if S1 > 0 else 0

        # 小误差概率
        small_errors = np.abs(residuals - residuals.mean()) < 0.6745 * S1
        P = small_errors.mean()

        # 精度等级
        if C < 0.35 and P > 0.95:
            level = "好 (Good)"
        elif C < 0.50 and P > 0.80:
            level = "合格 (Qualified)"
        elif C < 0.65 and P > 0.70:
            level = "勉强 (Reluctant)"
        else:
            level = "不合格 (Unqualified)"

        self._result = GreyResult(
            a=a, b=b,
            fitted_values=X0_hat,
            residuals=residuals,
            relative_errors=relative_errors,
            posterior_ratio=C,
            small_error_prob=P,
            accuracy_level=level,
        )
        return self

    def predict(self, steps=1):
        """
        预测未来值.

        Parameters
        ----------
        steps : int
            预测步数
        """
        self._ensure_fitted()
        r = self._result
        n = len(self.data)

        if abs(r.a) < 1e-10:
            # 线性退化
            forecasts = self.data[0] + r.b * np.arange(n, n + steps)
            return forecasts

        c = self.data[0] - r.b / r.a
        # 累加序列预测
        X1_future = c * np.exp(-r.a * np.arange(n, n + steps)) + r.b / r.a
        X1_current = c * np.exp(-r.a * np.arange(n - 1, n + steps)) + r.b / r.a

        # 累减还原
        forecasts = np.diff(X1_current)[-steps:]
        return forecasts

    @property
    def result(self):
        self._ensure_fitted()
        return self._result

    def _ensure_fitted(self):
        if self._result is None:
            raise RuntimeError("请先调用 fit() 进行计算")

    def plot(self, steps=3, figsize=(10, 6)):
        """绘制拟合和预测曲线."""
        import matplotlib.pyplot as plt

        self._ensure_fitted()
        r = self._result
        n = len(self.data)
        forecasts = self.predict(steps)

        fig, axes = plt.subplots(2, 1, figsize=figsize, gridspec_kw={"height_ratios": [3, 1]})

        # 上图: 拟合和预测
        ax = axes[0]
        x_orig = np.arange(n)
        x_fore = np.arange(n, n + steps)

        ax.plot(x_orig, self.data, "bo-", label="原始数据", markersize=6)
        ax.plot(x_orig, r.fitted_values, "r--s", label="拟合值", markersize=5)
        ax.plot(x_fore, forecasts, "g--^", label=f"预测值 (未来{steps}步)", markersize=7)

        ax.fill_between(x_fore,
                        forecasts * (1 - r.relative_errors.mean()),
                        forecasts * (1 + r.relative_errors.mean()),
                        alpha=0.2, color="green", label="预测区间 (±平均相对误差)")

        ax.set_xlabel("时间")
        ax.set_ylabel("值")
        ax.set_title(f"GM(1,1) 灰色预测\na={r.a:.4f}, b={r.b:.4f}, C={r.posterior_ratio:.4f}, {r.accuracy_level}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 下图: 相对误差
        ax2 = axes[1]
        ax2.bar(x_orig[1:], r.relative_errors[1:] * 100, color="orange", alpha=0.7)
        ax2.axhline(y=r.relative_errors[1:].mean() * 100, color="red", linestyle="--",
                     label=f"平均相对误差: {r.relative_errors[1:].mean()*100:.2f}%")
        ax2.set_xlabel("时间")
        ax2.set_ylabel("相对误差 (%)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def summary(self, steps=3):
        """打印结果摘要."""
        self._ensure_fitted()
        r = self._result
        forecasts = self.predict(steps)
        n = len(self.data)

        lines = [
            "=" * 60,
            "  GM(1,1) 灰色预测结果",
            "=" * 60,
            f"  发展系数 a = {r.a:.6f}",
            f"  灰色作用量 b = {r.b:.6f}",
            f"  后验差比值 C = {r.posterior_ratio:.4f}",
            f"  小误差概率 P = {r.small_error_prob:.4f}",
            f"  精度等级: {r.accuracy_level}",
            "-" * 60,
            "  拟合结果:",
        ]
        for i in range(n):
            lines.append(f"    t={i+1}: 实际={self.data[i]:.2f}, 拟合={r.fitted_values[i]:.2f}, "
                         f"相对误差={r.relative_errors[i]*100:.2f}%")
        lines.append("-" * 60)
        lines.append(f"  预测结果 (未来{steps}步):")
        for i, f in enumerate(forecasts):
            lines.append(f"    t={n+i+1}: {f:.2f}")
        lines.append("=" * 60)
        return "\n".join(lines)
