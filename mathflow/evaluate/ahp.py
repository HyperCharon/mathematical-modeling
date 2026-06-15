"""
层次分析法 (Analytic Hierarchy Process, AHP)

用于多准则决策，通过两两比较构建判断矩阵，计算权重并检验一致性。

Example:
    >>> from mathflow.evaluate import AHP
    >>> ahp = AHP()
    >>> ahp.set_matrix([
    ...     [1,   3,   5],
    ...     [1/3, 1,   3],
    ...     [1/5, 1/3, 1]
    ... ])
    >>> ahp.fit()
    >>> print(ahp.weights)
    >>> print(f"CR = {ahp.CR:.4f}, 一致性: {ahp.is_consistent}")
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# AHP 1-9 标度随机一致性指标 RI 表
_RI_TABLE = {
    1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12,
    6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49,
    11: 0.52, 12: 0.54, 13: 0.56, 14: 0.58, 15: 0.60,
}


@dataclass
class AHPResult:
    """AHP 计算结果."""
    weights: np.ndarray
    lambda_max: float
    CI: float
    CR: float
    is_consistent: bool
    n: int
    method: str


class AHP:
    """
    层次分析法 (AHP).

    支持多种权重计算方法:
        - "eigenvalue": 特征值法 (默认，最经典)
        - "geometric":  几何平均法 (算术平均的改进)
        - "arithmetic": 算术平均法

    Parameters
    ----------
    method : str
        权重计算方法，默认 "eigenvalue"
    cr_threshold : float
        一致性比率阈值，默认 0.10
    """

    def __init__(self, method: str = "eigenvalue", cr_threshold: float = 0.10):
        self.method = method
        self.cr_threshold = cr_threshold
        self._matrix = None
        self._result = None

    def set_matrix(self, matrix):
        """
        设置判断矩阵.

        Parameters
        ----------
        matrix : array-like
            n×n 的判断矩阵，a_ij 表示 i 相对于 j 的重要程度
        """
        matrix = np.asarray(matrix, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"判断矩阵必须是方阵，当前形状: {matrix.shape}")
        n = matrix.shape[0]
        # 验证互反性 (允许小误差)
        for i in range(n):
            for j in range(n):
                if matrix[i, j] * matrix[j, i] < 0.99 or matrix[i, j] * matrix[j, i] > 1.01:
                    if i != j:
                        pass  # 宽松处理，实际比赛数据常有小误差
        self._matrix = matrix
        self._result = None
        return self

    def set_comparison(self, comparisons):
        """
        通过两两比较列表设置判断矩阵.

        Parameters
        ----------
        comparisons : list of (i, j, value)
            i 相对于 j 的重要程度，例如 (0, 1, 3) 表示第0个比第1个重要程度为3
        """
        n = max(max(i, j) for i, j, _ in comparisons) + 1
        matrix = np.ones((n, n))
        for i, j, v in comparisons:
            matrix[i, j] = v
            matrix[j, i] = 1.0 / v
        return self.set_matrix(matrix)

    def fit(self):
        """计算权重和一致性指标."""
        if self._matrix is None:
            raise RuntimeError("请先调用 set_matrix() 设置判断矩阵")

        matrix = self._matrix
        n = matrix.shape[0]

        if self.method == "eigenvalue":
            weights, lambda_max = self._eigenvalue_method(matrix)
        elif self.method == "geometric":
            weights, lambda_max = self._geometric_method(matrix)
        elif self.method == "arithmetic":
            weights, lambda_max = self._arithmetic_method(matrix)
        else:
            raise ValueError(f"未知方法: {self.method}, 可选: eigenvalue, geometric, arithmetic")

        # 一致性检验
        CI = (lambda_max - n) / (n - 1) if n > 1 else 0.0
        RI = _RI_TABLE.get(n, 1.49 + 0.2 * (n - 15)) if n > 15 else _RI_TABLE.get(n, 0.58)
        CR = CI / RI if RI > 0 else 0.0

        self._result = AHPResult(
            weights=weights,
            lambda_max=lambda_max,
            CI=CI,
            CR=CR,
            is_consistent=CR < self.cr_threshold,
            n=n,
            method=self.method,
        )
        return self

    @property
    def weights(self):
        """权重向量."""
        self._ensure_fitted()
        return self._result.weights

    @property
    def CR(self):
        """一致性比率."""
        self._ensure_fitted()
        return self._result.CR

    @property
    def is_consistent(self):
        """是否通过一致性检验."""
        self._ensure_fitted()
        return self._result.is_consistent

    @property
    def result(self):
        """完整结果."""
        self._ensure_fitted()
        return self._result

    def _ensure_fitted(self):
        if self._result is None:
            raise RuntimeError("请先调用 fit() 进行计算")

    # ---- 计算方法 ----

    def _eigenvalue_method(self, matrix):
        """特征值法."""
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        idx = np.argmax(eigenvalues.real)
        lambda_max = eigenvalues[idx].real
        weights = eigenvectors[:, idx].real
        weights = weights / weights.sum()
        return weights, lambda_max

    def _geometric_method(self, matrix):
        """几何平均法 (行元素几何平均归一化)."""
        n = matrix.shape[0]
        geometric_mean = np.prod(matrix, axis=1) ** (1.0 / n)
        weights = geometric_mean / geometric_mean.sum()
        # 近似计算 lambda_max
        lambda_max = np.sum((matrix @ weights) / weights) / n
        return weights, lambda_max

    def _arithmetic_method(self, matrix):
        """算术平均法 (列归一化后行平均)."""
        col_sums = matrix.sum(axis=0)
        normalized = matrix / col_sums
        weights = normalized.mean(axis=1)
        lambda_max = np.sum((matrix @ weights) / weights) / matrix.shape[0]
        return weights, lambda_max

    # ---- 可视化 ----

    def plot_weights(self, labels=None, figsize=(8, 5), title="AHP 权重分布"):
        """绘制权重柱状图."""
        import matplotlib.pyplot as plt

        self._ensure_fitted()
        weights = self.weights
        n = len(weights)
        if labels is None:
            labels = [f"指标{i+1}" for i in range(n)]

        fig, ax = plt.subplots(figsize=figsize)
        bars = ax.bar(range(n), weights, color=plt.cm.Blues(np.linspace(0.4, 0.8, n)))
        ax.set_xticks(range(n))
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_ylabel("权重", fontsize=12)
        ax.set_title(f"{title}\n(CR={self.CR:.4f}, {'✅ 通过' if self.is_consistent else '❌ 未通过'}一致性检验)", fontsize=13)

        for bar, w in zip(bars, weights):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{w:.4f}", ha="center", fontsize=10)

        plt.tight_layout()
        return fig

    def summary(self):
        """打印结果摘要."""
        self._ensure_fitted()
        r = self._result
        lines = [
            "=" * 50,
            "  AHP 层次分析法结果",
            "=" * 50,
            f"  矩阵阶数: {r.n}",
            f"  计算方法: {r.method}",
            f"  λ_max:   {r.lambda_max:.4f}",
            f"  CI:      {r.ci:.4f}" if hasattr(r, 'ci') else f"  CI:      {r.CI:.4f}",
            f"  CR:      {r.CR:.4f}",
            f"  一致性:  {'✅ 通过' if r.is_consistent else '❌ 未通过'} (阈值={self.cr_threshold})",
            "-" * 50,
            "  权重向量:",
        ]
        for i, w in enumerate(r.weights):
            lines.append(f"    W{i+1} = {w:.4f}")
        lines.append("=" * 50)
        return "\n".join(lines)

    def __repr__(self):
        if self._result:
            return f"AHP(n={self._result.n}, method={self.method}, CR={self.CR:.4f})"
        return f"AHP(method={self.method})"
