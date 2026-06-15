"""
TOPSIS 法 (Technique for Order Preference by Similarity to Ideal Solution)

逼近理想解排序法，用于多指标综合评价。

Example:
    >>> from mathflow.evaluate import TOPSIS
    >>> import numpy as np
    >>> data = np.array([
    ...     [80, 90, 85, 70],
    ...     [70, 80, 90, 80],
    ...     [90, 85, 80, 75],
    ...     [85, 75, 88, 85],
    ... ])
    >>> topsis = TOPSIS(data, weights=[0.3, 0.25, 0.25, 0.2])
    >>> result = topsis.fit()
    >>> print(topsis.rankings)
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class TOPSISResult:
    """TOPSIS 计算结果."""
    scores: np.ndarray        # 综合得分
    rankings: np.ndarray      # 排名 (1=最优)
    ideal_best: np.ndarray    # 正理想解
    ideal_worst: np.ndarray   # 负理想解
    dist_best: np.ndarray     # 到正理想解的距离
    dist_worst: np.ndarray    # 到负理想解的距离
    normalized_data: np.ndarray  # 归一化后的数据


class TOPSIS:
    """
    TOPSIS 逼近理想解排序法.

    Parameters
    ----------
    data : array-like, shape (n_samples, n_indicators)
        评价矩阵，每行是一个方案，每列是一个指标
    weights : array-like, shape (n_indicators,)
        各指标权重，自动归一化
    types : list of int, optional
        指标类型: 1=效益型(越大越好), -1=成本型(越小越好)
        默认全部为效益型
    """

    def __init__(self, data, weights=None, types=None):
        self.data = np.asarray(data, dtype=float)
        if self.data.ndim != 2:
            raise ValueError("data 必须是二维矩阵")

        n_indicators = self.data.shape[1]

        # 权重处理
        if weights is None:
            weights = np.ones(n_indicators) / n_indicators
        self.weights = np.asarray(weights, dtype=float)
        self.weights = self.weights / self.weights.sum()  # 归一化

        # 指标类型
        if types is None:
            types = [1] * n_indicators
        self.types = np.asarray(types)
        self._result = None

    def fit(self):
        """执行 TOPSIS 计算."""
        data = self.data.copy()
        n, m = data.shape
        weights = self.weights
        types = self.types

        # Step 1: 向量归一化
        norms = np.sqrt((data ** 2).sum(axis=0))
        norms[norms == 0] = 1e-10
        normalized = data / norms

        # Step 2: 加权归一化矩阵
        weighted = normalized * weights

        # Step 3: 确定正理想解和负理想解
        ideal_best = np.zeros(m)
        ideal_worst = np.zeros(m)
        for j in range(m):
            if types[j] == 1:  # 效益型
                ideal_best[j] = weighted[:, j].max()
                ideal_worst[j] = weighted[:, j].min()
            else:  # 成本型
                ideal_best[j] = weighted[:, j].min()
                ideal_worst[j] = weighted[:, j].max()

        # Step 4: 计算距离
        dist_best = np.sqrt(((weighted - ideal_best) ** 2).sum(axis=1))
        dist_worst = np.sqrt(((weighted - ideal_worst) ** 2).sum(axis=1))

        # Step 5: 计算综合得分
        scores = dist_worst / (dist_best + dist_worst)
        scores = np.nan_to_num(scores, nan=0.5)  # All identical → score 0.5

        # Step 6: 排名 (得分越高排名越靠前)
        rankings = np.argsort(np.argsort(-scores)) + 1

        self._result = TOPSISResult(
            scores=scores,
            rankings=rankings,
            ideal_best=ideal_best,
            ideal_worst=ideal_worst,
            dist_best=dist_best,
            dist_worst=dist_worst,
            normalized_data=weighted,
        )
        return self

    @property
    def scores(self):
        self._ensure_fitted()
        return self._result.scores

    @property
    def rankings(self):
        self._ensure_fitted()
        return self._result.rankings

    @property
    def result(self):
        self._ensure_fitted()
        return self._result

    def _ensure_fitted(self):
        if self._result is None:
            raise RuntimeError("请先调用 fit() 进行计算")

    def plot_scores(self, labels=None, figsize=(8, 5)):
        """绘制综合得分柱状图."""
        import matplotlib.pyplot as plt

        self._ensure_fitted()
        scores = self.scores
        n = len(scores)
        if labels is None:
            labels = [f"方案{i+1}" for i in range(n)]

        # 按得分排序
        order = np.argsort(-scores)

        fig, ax = plt.subplots(figsize=figsize)
        colors = plt.cm.RdYlGn(np.linspace(0.8, 0.3, n))
        bars = ax.barh(range(n), scores[order], color=colors)
        ax.set_yticks(range(n))
        ax.set_yticklabels([labels[i] for i in order], fontsize=11)
        ax.set_xlabel("综合得分", fontsize=12)
        ax.set_title("TOPSIS 综合评价得分", fontsize=14)

        for bar, s in zip(bars, scores[order]):
            ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                    f"{s:.4f} (排名{self.rankings[order[list(bars).index(bar)]]})",
                    va="center", fontsize=10)

        ax.invert_yaxis()
        plt.tight_layout()
        return fig

    def summary(self, labels=None):
        """打印结果摘要."""
        self._ensure_fitted()
        r = self._result
        n = len(r.scores)
        if labels is None:
            labels = [f"方案{i+1}" for i in range(n)]

        lines = [
            "=" * 60,
            "  TOPSIS 逼近理想解排序法结果",
            "=" * 60,
        ]
        for i in np.argsort(-r.scores):
            lines.append(f"  {labels[i]:>10s}  得分={r.scores[i]:.4f}  排名={r.rankings[i]}")
        lines.append("=" * 60)
        return "\n".join(lines)

    def __repr__(self):
        if self._result:
            return f"TOPSIS(n_samples={self.data.shape[0]}, n_indicators={self.data.shape[1]})"
        return f"TOPSIS(shape={self.data.shape})"
