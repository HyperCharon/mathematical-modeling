"""
灰色关联分析 (Grey Relational Analysis, GRA)

衡量序列之间的几何相似程度，关联度越大说明越相关。

Example:
    >>> from mathflow.evaluate import GreyRelationalAnalysis
    >>> import numpy as np
    >>> reference = np.array([1.0, 1.1, 1.2, 1.3, 1.4])
    >>> compare = np.array([
    ...     [1.0, 0.9, 1.1, 1.2, 1.3],
    ...     [0.8, 1.0, 1.0, 1.1, 1.2],
    ...     [1.2, 1.3, 1.1, 1.0, 0.9],
    ... ])
    >>> gra = GreyRelationalAnalysis(reference, compare)
    >>> gra.fit()
    >>> print(gra.correlations)
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class GRAResult:
    """灰色关联分析结果."""
    correlations: np.ndarray   # 各序列的关联度
    relation_matrix: np.ndarray  # 关联系数矩阵


class GreyRelationalAnalysis:
    """
    灰色关联分析.

    Parameters
    ----------
    reference : array-like, shape (n_points,)
        参考序列 (母序列)
    compare : array-like, shape (n_series, n_points)
        比较序列 (子序列矩阵，每行是一个序列)
    rho : float
        分辨系数，取值范围 [0, 1]，默认 0.5
    """

    def __init__(self, reference, compare, rho=0.5):
        self.reference = np.asarray(reference, dtype=float).flatten()
        self.compare = np.asarray(compare, dtype=float)
        if self.compare.ndim == 1:
            self.compare = self.compare.reshape(1, -1)
        if self.compare.shape[1] != len(self.reference):
            raise ValueError("参考序列和比较序列长度必须一致")
        self.rho = rho
        self._result = None

    def fit(self):
        """计算灰色关联度."""
        ref = self.reference
        comp = self.compare
        n_series, n_points = comp.shape
        rho = self.rho

        # Step 1: 无量纲化 (初值化)
        epsilon = 1e-10
        if abs(ref[0]) < epsilon:
            import warnings
            warnings.warn("参考序列首值为零，初值化可能无意义")
            ref_norm = ref
        else:
            ref_norm = ref / ref[0]

        # 对比较序列初值化，防止除零
        first_vals = comp[:, 0:1]
        comp_norm = np.where(np.abs(first_vals) > epsilon,
                            comp / first_vals,
                            comp)
        comp_norm = np.where(np.isfinite(comp_norm), comp_norm, 0)

        # Step 2: 计算差序列
        diff = np.abs(comp_norm - ref_norm)

        # Step 3: 计算最大差和最小差
        delta_min = diff.min()
        delta_max = diff.max()

        # Step 4: 计算关联系数
        if delta_max == 0:
            xi = np.ones_like(diff)
        else:
            xi = (delta_min + rho * delta_max) / (diff + rho * delta_max)

        # Step 5: 计算关联度 (关联系数的平均值)
        correlations = xi.mean(axis=1)

        self._result = GRAResult(
            correlations=correlations,
            relation_matrix=xi,
        )
        return self

    @property
    def correlations(self):
        self._ensure_fitted()
        return self._result.correlations

    @property
    def result(self):
        self._ensure_fitted()
        return self._result

    def _ensure_fitted(self):
        if self._result is None:
            raise RuntimeError("请先调用 fit() 进行计算")

    def plot(self, labels=None, figsize=(10, 5)):
        """绘制关联度和关联系数热力图."""
        import matplotlib.pyplot as plt

        self._ensure_fitted()
        r = self._result
        n_series = len(r.correlations)
        if labels is None:
            labels = [f"序列{i+1}" for i in range(n_series)]

        fig, axes = plt.subplots(1, 2, figsize=(figsize[0] * 1.2, figsize[1]))

        # 左: 关联度柱状图
        ax = axes[0]
        colors = plt.cm.YlOrRd(np.linspace(0.4, 0.9, n_series))
        order = np.argsort(-r.correlations)
        ax.barh(range(n_series), r.correlations[order], color=colors)
        ax.set_yticks(range(n_series))
        ax.set_yticklabels([labels[i] for i in order])
        ax.set_xlabel("关联度")
        ax.set_title("灰色关联度排序")
        ax.invert_yaxis()

        # 右: 关联系数热力图
        ax2 = axes[1]
        im = ax2.imshow(r.relation_matrix, cmap="YlOrRd", aspect="auto")
        ax2.set_ylabel("比较序列")
        ax2.set_xlabel("时刻")
        ax2.set_title("关联系数矩阵")
        plt.colorbar(im, ax=ax2, label="关联系数")

        plt.tight_layout()
        return fig

    def summary(self, labels=None):
        """打印结果摘要."""
        self._ensure_fitted()
        r = self._result
        n_series = len(r.correlations)
        if labels is None:
            labels = [f"序列{i+1}" for i in range(n_series)]

        lines = [
            "=" * 50,
            "  灰色关联分析 (GRA) 结果",
            "=" * 50,
            f"  分辨系数 ρ = {self.rho}",
            "-" * 50,
        ]
        for i in np.argsort(-r.correlations):
            lines.append(f"  {labels[i]:>10s}  关联度 = {r.correlations[i]:.4f}")
        lines.append("=" * 50)
        return "\n".join(lines)
