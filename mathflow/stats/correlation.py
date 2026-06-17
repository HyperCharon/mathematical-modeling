"""
相关性分析

支持 Pearson、Spearman、Kendall 相关系数计算，含显著性检验和可视化。

Example:
    >>> from mathflow.stats import CorrelationAnalysis
    >>> import numpy as np
    >>> X = np.random.randn(100, 4)
    >>> ca = CorrelationAnalysis(X, var_names=["温度", "压力", "浓度", "时间"])
    >>> result = ca.fit(method="pearson")
    >>> print(ca.summary())
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
from scipy import stats


@dataclass
class CorrelationResult:
    """相关性分析结果."""
    corr_matrix: np.ndarray       # 相关系数矩阵
    p_matrix: np.ndarray          # p 值矩阵
    method: str
    n_samples: int
    n_vars: int
    var_names: List[str]


class CorrelationAnalysis:
    """
    相关性分析.

    Parameters
    ----------
    data : array-like, shape (n_samples, n_vars)
        数据矩阵
    var_names : list of str, optional
        变量名
    """

    def __init__(self, data, var_names=None):
        self.data = np.asarray(data, dtype=float)
        if self.data.ndim == 1:
            self.data = self.data.reshape(-1, 1)

        self.n_samples, self.n_vars = self.data.shape

        if var_names is None:
            self.var_names = [f"X{i+1}" for i in range(self.n_vars)]
        else:
            if len(var_names) != self.n_vars:
                raise ValueError(f"var_names 长度 ({len(var_names)}) 必须等于变量数 ({self.n_vars})")
            self.var_names = var_names

        self._result = None

    def __repr__(self) -> str:
        return f"CorrelationAnalysis(n_samples={self.n_samples}, n_vars={self.n_vars})"

    def fit(self, method: str = "pearson") -> CorrelationResult:
        """
        计算相关系数.

        Parameters
        ----------
        method : str
            "pearson" (皮尔逊), "spearman" (斯皮尔曼), "kendall" (肯德尔)
        """
        n = self.n_vars
        corr_matrix = np.zeros((n, n))
        p_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i == j:
                    corr_matrix[i, j] = 1.0
                    p_matrix[i, j] = 0.0
                elif i < j:
                    if method == "pearson":
                        corr, p = stats.pearsonr(self.data[:, i], self.data[:, j])
                    elif method == "spearman":
                        corr, p = stats.spearmanr(self.data[:, i], self.data[:, j])
                    elif method == "kendall":
                        corr, p = stats.kendalltau(self.data[:, i], self.data[:, j])
                    else:
                        raise ValueError(f"未知方法: {method}")

                    corr_matrix[i, j] = corr
                    corr_matrix[j, i] = corr
                    p_matrix[i, j] = p
                    p_matrix[j, i] = p

        self._result = CorrelationResult(
            corr_matrix=corr_matrix,
            p_matrix=p_matrix,
            method=method,
            n_samples=self.n_samples,
            n_vars=self.n_vars,
            var_names=self.var_names,
        )
        return self._result

    def get_significant_pairs(self, alpha: float = 0.05,
                              min_corr: float = 0.0) -> List[Tuple]:
        """
        获取显著相关的变量对.

        Parameters
        ----------
        alpha : float
            显著性水平
        min_corr : float
            最小相关系数绝对值
        """
        self._ensure_fitted()
        r = self._result
        pairs = []

        for i in range(r.n_vars):
            for j in range(i + 1, r.n_vars):
                if r.p_matrix[i, j] < alpha and abs(r.corr_matrix[i, j]) >= min_corr:
                    pairs.append((
                        r.var_names[i],
                        r.var_names[j],
                        r.corr_matrix[i, j],
                        r.p_matrix[i, j],
                    ))

        # 按相关系数绝对值排序
        pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        return pairs

    @property
    def corr_matrix(self):
        self._ensure_fitted()
        return self._result.corr_matrix

    def _ensure_fitted(self):
        if self._result is None:
            raise RuntimeError("请先调用 fit()")

    def plot(self, figsize=(8, 6), annot=True):
        """绘制相关系数热力图."""
        import matplotlib.pyplot as plt

        self._ensure_fitted()
        r = self._result

        fig, ax = plt.subplots(figsize=figsize)

        # 绘制热力图
        im = ax.imshow(r.corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1)

        # 设置刻度
        ax.set_xticks(range(r.n_vars))
        ax.set_yticks(range(r.n_vars))
        ax.set_xticklabels(r.var_names, rotation=45, ha="right")
        ax.set_yticklabels(r.var_names)

        # 添加数值标注
        if annot:
            for i in range(r.n_vars):
                for j in range(r.n_vars):
                    text = f"{r.corr_matrix[i, j]:.2f}"
                    if r.p_matrix[i, j] < 0.001:
                        text += "***"
                    elif r.p_matrix[i, j] < 0.01:
                        text += "**"
                    elif r.p_matrix[i, j] < 0.05:
                        text += "*"
                    ax.text(j, i, text, ha="center", va="center",
                           fontsize=9, color="black")

        plt.colorbar(im, ax=ax, label="相关系数")
        ax.set_title(f"{r.method.title()} 相关系数矩阵")

        plt.tight_layout()
        return fig

    def summary(self) -> str:
        """打印摘要."""
        self._ensure_fitted()
        r = self._result

        lines = [
            "=" * 70,
            "  相关性分析结果",
            "=" * 70,
            f"  方法: {r.method.title()}",
            f"  样本量: {r.n_samples}",
            f"  变量数: {r.n_vars}",
            "-" * 70,
        ]

        # 打印相关系数矩阵
        header = f"  {'':>12s}"
        for name in r.var_names:
            header += f"  {name:>10s}"
        lines.append(header)
        lines.append("-" * 70)

        for i in range(r.n_vars):
            row = f"  {r.var_names[i]:>12s}"
            for j in range(r.n_vars):
                val = r.corr_matrix[i, j]
                row += f"  {val:>10.4f}"
            lines.append(row)

        lines.append("-" * 70)

        # 显著性标注说明
        lines.append("  显著性: *** p<0.001  ** p<0.01  * p<0.05")

        # 打印显著相关对
        pairs = self.get_significant_pairs(alpha=0.05)
        if pairs:
            lines.append("-" * 70)
            lines.append("  显著相关的变量对 (p<0.05):")
            for name1, name2, corr, p in pairs[:10]:
                lines.append(f"    {name1} - {name2}: r={corr:.4f}, p={p:.4f}")

        lines.append("=" * 70)
        return "\n".join(lines)
