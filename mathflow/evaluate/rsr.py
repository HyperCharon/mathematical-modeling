"""
秩和比法 (Rank-Sum Ratio, RSR)

融合了参数统计与非参数统计的优点，适用于多指标综合评价。

Example:
    >>> from mathflow.evaluate import RSR
    >>> import numpy as np
    >>> data = np.array([
    ...     [80, 90, 85],
    ...     [70, 80, 90],
    ...     [90, 85, 80],
    ...     [85, 75, 88],
    ... ])
    >>> rsr = RSR(data, weights=[0.4, 0.3, 0.3])
    >>> rsr.fit()
    >>> print(rsr.rankings)
"""

import numpy as np
from scipy import stats
from dataclasses import dataclass


@dataclass
class RSRResult:
    """RSR 结果."""
    rsr_values: np.ndarray     # 各方案的 RSR 值
    rankings: np.ndarray       # 排名
    probit: np.ndarray         # 概率单位 (用于回归分档)
    regression_params: tuple   # (slope, intercept, r_value) 回归参数


class RSR:
    """
    秩和比法 (RSR).

    Parameters
    ----------
    data : array-like, shape (n_samples, n_indicators)
        评价矩阵
    weights : array-like, optional
        各指标权重
    types : list of int, optional
        指标类型: 1=效益型, -1=成本型
    """

    def __init__(self, data, weights=None, types=None):
        self.data = np.asarray(data, dtype=float)
        if self.data.ndim != 2:
            raise ValueError("data 必须是二维矩阵")

        n, m = self.data.shape
        if weights is None:
            weights = np.ones(m) / m
        self.weights = np.asarray(weights, dtype=float)
        self.weights = self.weights / self.weights.sum()
        self.types = types
        self._result = None

    def fit(self):
        """计算 RSR."""
        data = self.data.copy()
        n, m = data.shape

        # Step 1: 编秩 (对每个指标分别排序)
        ranks = np.zeros_like(data)
        for j in range(m):
            if self.types and self.types[j] == -1:
                # 成本型: 降序编秩 (越小越好)
                ranks[:, j] = n + 1 - stats.rankdata(data[:, j])
            else:
                # 效益型: 升序编秩 (越大越好)
                ranks[:, j] = stats.rankdata(data[:, j])

        # Step 2: 计算加权 RSR
        rsr_values = (ranks * self.weights).sum(axis=1) / n

        # Step 3: 排名
        rankings = np.argsort(np.argsort(-rsr_values)) + 1

        # Step 4: 计算概率单位 (用于分档回归)
        sorted_rsr = np.sort(rsr_values)
        cumulative = np.arange(1, n + 1) / (n + 1)  # 累积频率
        probit = stats.norm.ppf(cumulative) * 10 + 5  # 概率单位

        # 线性回归: RSR = a * Probit + b
        slope, intercept, r_value, _, _ = stats.linregress(probit, sorted_rsr)

        self._result = RSRResult(
            rsr_values=rsr_values,
            rankings=rankings,
            probit=probit,
            regression_params=(slope, intercept, r_value),
        )
        return self

    @property
    def rankings(self):
        self._ensure_fitted()
        return self._result.rankings

    @property
    def rsr_values(self):
        self._ensure_fitted()
        return self._result.rsr_values

    def _ensure_fitted(self):
        if self._result is None:
            raise RuntimeError("请先调用 fit() 进行计算")

    def summary(self, labels=None):
        """打印结果摘要."""
        self._ensure_fitted()
        r = self._result
        n = len(r.rsr_values)
        if labels is None:
            labels = [f"方案{i+1}" for i in range(n)]

        lines = [
            "=" * 50,
            "  秩和比法 (RSR) 结果",
            "=" * 50,
        ]
        for i in np.argsort(-r.rsr_values):
            lines.append(f"  {labels[i]:>10s}  RSR={r.rsr_values[i]:.4f}  排名={r.rankings[i]}")
        lines.append("-" * 50)
        slope, intercept, r_value = r.regression_params
        lines.append(f"  回归方程: RSR = {slope:.4f} × Probit + ({intercept:.4f})")
        lines.append(f"  R² = {r_value**2:.4f}")
        lines.append("=" * 50)
        return "\n".join(lines)
