"""
CRITIC 法 (Criteria Importance Through Intercriteria Correlation)

基于指标对比强度和冲突性客观赋权。
考虑了指标间的相关性：与其他指标相关性越低的指标，权重越高。

Example:
    >>> from mathflow.evaluate import CRITIC
    >>> critic = CRITIC(data)
    >>> critic.fit()
    >>> print(critic.weights)
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class CRITICResult:
    """CRITIC 法结果."""
    weights: np.ndarray
    contrast: np.ndarray     # 对比强度 (标准差)
    conflict: np.ndarray     # 冲突性 (1 - 相关系数之和)
    information: np.ndarray  # 信息量 = 对比强度 × 冲突性


class CRITIC:
    """
    CRITIC 法客观赋权.

    Parameters
    ----------
    data : array-like, shape (n_samples, n_indicators)
        评价矩阵
    types : list of int, optional
        指标类型: 1=效益型, -1=成本型
    """

    def __init__(self, data, types=None):
        self.data = np.asarray(data, dtype=float)
        if self.data.ndim != 2:
            raise ValueError("data 必须是二维矩阵")
        self.types = types
        self._result = None

    def fit(self):
        """计算 CRITIC 权重."""
        data = self.data.copy()
        n, m = data.shape

        # Step 1: 指标正向化
        if self.types is not None:
            for j in range(m):
                if self.types[j] == -1:
                    data[:, j] = data[:, j].max() - data[:, j]

        # Step 2: 归一化 (min-max)
        mins = data.min(axis=0)
        maxs = data.max(axis=0)
        ranges = maxs - mins
        ranges[ranges == 0] = 1e-10
        normalized = (data - mins) / ranges

        # Step 3: 对比强度 (标准差)
        contrast = normalized.std(axis=0)

        # Step 4: 冲突性 (相关系数矩阵)
        corr = np.corrcoef(normalized.T)
        np.fill_diagonal(corr, 0)
        conflict = 1 - corr.sum(axis=0) / (m - 1) if m > 1 else np.ones(m)

        # Step 5: 信息量
        information = contrast * conflict

        # Step 6: 权重
        info_sum = information.sum()
        weights = information / info_sum if info_sum > 0 else np.ones(m) / m

        self._result = CRITICResult(
            weights=weights,
            contrast=contrast,
            conflict=conflict,
            information=information,
        )
        return self

    @property
    def weights(self):
        self._ensure_fitted()
        return self._result.weights

    @property
    def result(self):
        self._ensure_fitted()
        return self._result

    def _ensure_fitted(self):
        if self._result is None:
            raise RuntimeError("请先调用 fit() 进行计算")

    def summary(self, labels=None):
        """打印结果摘要."""
        self._ensure_fitted()
        r = self._result
        m = len(r.weights)
        if labels is None:
            labels = [f"指标{j+1}" for j in range(m)]

        lines = [
            "=" * 70,
            "  CRITIC 法客观赋权结果",
            "=" * 70,
            f"  {'指标':>10s}  {'对比强度':>8s}  {'冲突性':>8s}  {'信息量':>8s}  {'权重':>8s}",
            "-" * 70,
        ]
        for j in range(m):
            lines.append(
                f"  {labels[j]:>10s}  {r.contrast[j]:>8.4f}  "
                f"{r.conflict[j]:>8.4f}  {r.information[j]:>8.4f}  {r.weights[j]:>8.4f}"
            )
        lines.append("=" * 70)
        return "\n".join(lines)
