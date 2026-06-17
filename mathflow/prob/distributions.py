"""
概率分布拟合

自动拟合常见概率分布 (正态、指数、泊松、伽马、威布尔等)。

Example:
    >>> from mathflow.prob import DistributionFitter
    >>> import numpy as np
    >>> data = np.random.normal(50, 10, 200)
    >>> df = DistributionFitter(data)
    >>> result = df.auto_fit()
"""

import numpy as np
from scipy import stats
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class FitResult:
    """分布拟合结果."""
    distribution: str
    params: tuple
    ks_statistic: float
    ks_p_value: float
    aic: float
    bic: float


class DistributionFitter:
    """
    概率分布拟合器.

    Parameters
    ----------
    data : array-like
        观测数据
    """

    DISTRIBUTIONS = [
        ("norm", stats.norm, "正态分布"),
        ("expon", stats.expon, "指数分布"),
        ("gamma", stats.gamma, "伽马分布"),
        ("weibull_min", stats.weibull_min, "威布尔分布"),
        ("lognorm", stats.lognorm, "对数正态分布"),
        ("beta", stats.beta, "贝塔分布"),
        ("uniform", stats.uniform, "均匀分布"),
        ("pareto", stats.pareto, "帕累托分布"),
    ]

    def __init__(self, data):
        self.data = np.asarray(data, dtype=float).flatten()
        self.n = len(self.data)
        if self.n < 2:
            raise ValueError("分布拟合至少需要 2 个数据点")
        if np.all(self.data == self.data[0]):
            import warnings
            warnings.warn("数据为常数，分布拟合结果可能无意义")
        self._results = []

    def __repr__(self) -> str:
        if self._results:
            best = self._results[0]
            return f"DistributionFitter(n={self.n}, best={best.distribution!r})"
        return f"DistributionFitter(n={self.n})"

    def fit(self, dist_name: str) -> FitResult:
        """拟合指定分布."""
        dist_info = next((d for d in self.DISTRIBUTIONS if d[0] == dist_name), None)
        if dist_info is None:
            raise ValueError(f"未知分布: {dist_name}")

        name, dist, cn_name = dist_info
        params = dist.fit(self.data)

        # K-S 检验
        ks_stat, ks_p = stats.kstest(self.data, name, args=params)

        # AIC, BIC
        try:
            log_lik = np.sum(dist.logpdf(self.data, *params))
            k = len(params)
            # 检查 log_lik 是否有效
            if not np.isfinite(log_lik):
                aic = float("inf")
                bic = float("inf")
            else:
                aic = 2 * k - 2 * log_lik
                bic = k * np.log(self.n) - 2 * log_lik
        except Exception:
            aic = float("inf")
            bic = float("inf")

        result = FitResult(
            distribution=cn_name, params=params,
            ks_statistic=ks_stat, ks_p_value=ks_p,
            aic=aic, bic=bic,
        )
        return result

    def auto_fit(self, top_n: int = 3) -> List[FitResult]:
        """自动拟合所有分布，返回最优的几个."""
        results = []
        for name, dist, cn_name in self.DISTRIBUTIONS:
            try:
                result = self.fit(name)
                # 过滤掉无效的 AIC/BIC 值
                if np.isfinite(result.aic) and np.isfinite(result.bic):
                    results.append(result)
            except Exception:
                continue

        # 按 AIC 排序
        results.sort(key=lambda r: r.aic)
        self._results = results[:top_n]
        return self._results

    def goodness_of_fit(self) -> dict:
        """拟合优度检验 (K-S, 卡方)."""
        results = {}
        for name, dist, cn_name in self.DISTRIBUTIONS:
            try:
                params = dist.fit(self.data)
                ks_stat, ks_p = stats.kstest(self.data, name, args=params)
                results[cn_name] = {
                    "K-S统计量": ks_stat,
                    "K-S p值": ks_p,
                    "通过(p>0.05)": ks_p > 0.05,
                }
            except Exception:
                continue
        return results

    def plot(self, figsize=(12, 5)):
        """绘制拟合结果."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # 直方图 + 拟合曲线
        ax = axes[0]
        ax.hist(self.data, bins=30, density=True, alpha=0.6, color="steelblue", edgecolor="white")

        x = np.linspace(self.data.min(), self.data.max(), 200)
        for result in self._results[:3]:
            dist_info = next(d for d in self.DISTRIBUTIONS if d[2] == result.distribution)
            dist = dist_info[1]
            y = dist.pdf(x, *result.params)
            ax.plot(x, y, linewidth=2, label=f"{result.distribution} (AIC={result.aic:.1f})")

        ax.set_xlabel("值")
        ax.set_ylabel("概率密度")
        ax.set_title("分布拟合")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Q-Q 图
        ax2 = axes[1]
        if self._results:
            best = self._results[0]
            dist_info = next(d for d in self.DISTRIBUTIONS if d[2] == best.distribution)
            dist = dist_info[1]
            theoretical = dist.ppf(np.linspace(0.01, 0.99, self.n), *best.params)
            observed = np.sort(self.data)
            ax2.scatter(theoretical, observed, alpha=0.5, s=15)
            lims = [min(theoretical.min(), observed.min()),
                    max(theoretical.max(), observed.max())]
            ax2.plot(lims, lims, "r--", linewidth=1.5)
            ax2.set_xlabel("理论分位数")
            ax2.set_ylabel("样本分位数")
            ax2.set_title(f"Q-Q 图 ({best.distribution})")
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def summary(self):
        if not self._results:
            self.auto_fit()

        lines = [
            "=" * 60,
            "  概率分布拟合结果",
            "=" * 60,
            f"  样本量: {self.n}",
            f"  均值: {self.data.mean():.4f}",
            f"  标准差: {self.data.std():.4f}",
            f"  偏度: {stats.skew(self.data):.4f}",
            f"  峰度: {stats.kurtosis(self.data):.4f}",
            "-" * 60,
            f"  {'分布':>12s}  {'K-S统计量':>10s}  {'K-S p值':>10s}  {'AIC':>12s}  {'BIC':>12s}",
            "-" * 60,
        ]
        for r in self._results:
            lines.append(
                f"  {r.distribution:>12s}  {r.ks_statistic:>10.4f}  {r.ks_p_value:>10.4f}  "
                f"{r.aic:>12.1f}  {r.bic:>12.1f}"
            )
        lines.append("=" * 60)
        return "\n".join(lines)
