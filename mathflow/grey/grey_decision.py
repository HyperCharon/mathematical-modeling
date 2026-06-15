"""
灰色决策

灰色聚类决策、灰色关联决策。

Example:
    >>> from mathflow.grey import GreyDecision
    >>> import numpy as np
    >>> # 评价矩阵 (5个方案, 4个指标)
    >>> data = np.array([
    ...     [80, 90, 70, 85],
    ...     [75, 85, 80, 90],
    ...     [90, 80, 75, 80],
    ...     [85, 88, 82, 78],
    ...     [88, 92, 68, 88],
    ... ])
    >>> gd = GreyDecision(data)
    >>> result = gd.evaluate()
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class GreyDecisionResult:
    """灰色决策结果."""
    scores: np.ndarray
    rankings: np.ndarray
    method: str


class GreyDecision:
    """
    灰色决策.

    Parameters
    ----------
    data : array-like, shape (n_samples, n_indicators)
        评价矩阵
    weights : array-like, optional
        指标权重
    types : list of int, optional
        指标类型: 1=效益型, -1=成本型
    """

    def __init__(self, data, weights=None, types=None):
        self.data = np.asarray(data, dtype=float)
        n, m = self.data.shape
        self.weights = weights if weights is not None else np.ones(m) / m
        self.types = types if types is not None else [1] * m
        self._result = None

    def evaluate(self, method="grey_relational") -> GreyDecisionResult:
        """
        灰色评价.

        Parameters
        ----------
        method : str
            "grey_relational" (灰色关联评价), "grey_clustering" (灰色聚类)
        """
        if method == "grey_relational":
            return self._grey_relational_eval()
        elif method == "grey_clustering":
            return self._grey_clustering_eval()
        else:
            raise ValueError(f"未知方法: {method}")

    def _grey_relational_eval(self):
        """灰色关联评价."""
        data = self.data.copy()
        n, m = data.shape

        # 正向化
        for j in range(m):
            if self.types[j] == -1:
                data[:, j] = data[:, j].max() - data[:, j]

        # 确定参考序列 (各指标最优值)
        reference = data.max(axis=0)

        # 计算关联系数
        rho = 0.5
        diff = np.abs(data - reference)
        delta_min = diff.min()
        delta_max = diff.max()

        if delta_max == 0:
            xi = np.ones_like(diff)
        else:
            xi = (delta_min + rho * delta_max) / (diff + rho * delta_max)

        # 加权关联度
        scores = (xi * self.weights).sum(axis=1)
        rankings = np.argsort(-scores) + 1

        self._result = GreyDecisionResult(
            scores=scores, rankings=rankings, method="灰色关联评价"
        )
        return self._result

    def _grey_clustering_eval(self):
        """灰色聚类评价 (白化权函数)."""
        data = self.data.copy()
        n, m = data.shape

        # 定义灰类 (高/中/低)
        thresholds = np.percentile(data, [33, 67], axis=0)

        # 白化权函数
        scores = np.zeros(n)
        for i in range(n):
            score = 0
            for j in range(m):
                val = data[i, j]
                low, high = thresholds[0, j], thresholds[1, j]
                if val >= high:
                    membership = 1.0
                elif val >= low:
                    membership = (val - low) / (high - low)
                else:
                    membership = 0.0
                score += membership * self.weights[j]
            scores[i] = score

        rankings = np.argsort(-scores) + 1

        self._result = GreyDecisionResult(
            scores=scores, rankings=rankings, method="灰色聚类评价"
        )
        return self._result

    @property
    def result(self):
        if self._result is None:
            raise RuntimeError("请先调用 evaluate()")
        return self._result

    def summary(self, labels=None):
        r = self.result
        n = len(r.scores)
        if labels is None:
            labels = [f"方案{i+1}" for i in range(n)]

        lines = [
            "=" * 50,
            f"  {r.method}结果",
            "=" * 50,
        ]
        for i in np.argsort(-r.scores):
            lines.append(f"  {labels[i]:>10s}  得分={r.scores[i]:.4f}  排名={r.rankings[i]}")
        lines.append("=" * 50)
        return "\n".join(lines)
