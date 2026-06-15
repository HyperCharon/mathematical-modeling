"""
熵权法 (Entropy Weight Method)

基于信息熵客观确定指标权重，数据离散程度越大，该指标权重越高。

Example:
    >>> from mathflow.evaluate import EntropyWeight
    >>> import numpy as np
    >>> data = np.array([
    ...     [80, 90, 85, 70],
    ...     [70, 80, 90, 80],
    ...     [90, 85, 80, 75],
    ...     [85, 75, 88, 85],
    ... ])
    >>> ew = EntropyWeight(data)
    >>> ew.fit()
    >>> print(ew.weights)
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class EntropyWeightResult:
    """熵权法结果."""
    weights: np.ndarray
    entropy_values: np.ndarray  # 各指标的信息熵
    redundancy: np.ndarray      # 信息效用值 (1 - e)
    variation: np.ndarray       # 变异系数


class EntropyWeight:
    """
    熵权法确定客观权重.

    Parameters
    ----------
    data : array-like, shape (n_samples, n_indicators)
        评价矩阵
    types : list of int, optional
        指标类型: 1=效益型(越大越好), -1=成本型(越小越好)
    """

    def __init__(self, data, types=None):
        self.data = np.asarray(data, dtype=float)
        if self.data.ndim != 2:
            raise ValueError("data 必须是二维矩阵")
        self.types = types
        self._result = None

    def fit(self):
        """计算熵权."""
        data = self.data.copy()
        n, m = data.shape

        # Step 1: 指标正向化 (成本型取倒数)
        if self.types is not None:
            for j in range(m):
                if self.types[j] == -1:
                    # 避免除零
                    min_val = data[:, j].min()
                    if min_val > 0:
                        data[:, j] = 1.0 / data[:, j]
                    else:
                        data[:, j] = data[:, j].max() - data[:, j]

        # Step 2: 归一化 (比重法)
        col_sums = data.sum(axis=0)
        col_sums[col_sums == 0] = 1e-10
        P = data / col_sums  # 比重矩阵

        # Step 3: 计算信息熵
        # 避免 log(0)
        P_log = np.where(P > 0, P * np.log(P), 0)
        k = 1.0 / np.log(n)
        e = -k * P_log.sum(axis=0)  # 信息熵

        # Step 4: 信息效用值
        d = 1 - e

        # Step 5: 归一化得到权重
        d_sum = d.sum()
        if d_sum == 0:
            weights = np.ones(m) / m
        else:
            weights = d / d_sum

        # 变异系数
        variation = data.std(axis=0) / (data.mean(axis=0) + 1e-10)

        self._result = EntropyWeightResult(
            weights=weights,
            entropy_values=e,
            redundancy=d,
            variation=variation,
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

    def plot_weights(self, labels=None, figsize=(8, 5)):
        """绘制权重分布."""
        import matplotlib.pyplot as plt

        self._ensure_fitted()
        weights = self.weights
        m = len(weights)
        if labels is None:
            labels = [f"指标{j+1}" for j in range(m)]

        fig, axes = plt.subplots(1, 2, figsize=(figsize[0] * 1.5, figsize[1]))

        # 左: 权重柱状图
        ax = axes[0]
        colors = plt.cm.Oranges(np.linspace(0.4, 0.8, m))
        ax.bar(range(m), weights, color=colors)
        ax.set_xticks(range(m))
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel("权重")
        ax.set_title("熵权法 - 指标权重")

        # 右: 信息熵
        ax2 = axes[1]
        ax2.bar(range(m), self._result.entropy_values, color=plt.cm.Blues(np.linspace(0.4, 0.8, m)))
        ax2.set_xticks(range(m))
        ax2.set_xticklabels(labels, fontsize=10)
        ax2.set_ylabel("信息熵")
        ax2.set_title("各指标信息熵")

        plt.tight_layout()
        return fig

    def summary(self, labels=None):
        """打印结果摘要."""
        self._ensure_fitted()
        r = self._result
        m = len(r.weights)
        if labels is None:
            labels = [f"指标{j+1}" for j in range(m)]

        lines = [
            "=" * 60,
            "  熵权法 (Entropy Weight Method) 结果",
            "=" * 60,
            f"  {'指标':>10s}  {'信息熵':>8s}  {'效用值':>8s}  {'权重':>8s}",
            "-" * 60,
        ]
        for j in range(m):
            lines.append(
                f"  {labels[j]:>10s}  {r.entropy_values[j]:>8.4f}  "
                f"{r.redundancy[j]:>8.4f}  {r.weights[j]:>8.4f}"
            )
        lines.append("=" * 60)
        return "\n".join(lines)

    def __repr__(self):
        if self._result:
            return f"EntropyWeight(n_samples={self.data.shape[0]}, n_indicators={self.data.shape[1]})"
        return f"EntropyWeight(shape={self.data.shape})"
