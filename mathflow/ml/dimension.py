"""
降维分析: PCA 主成分分析

Example:
    >>> from mathflow.ml import DimensionReduction
    >>> import numpy as np
    >>> data = np.random.randn(100, 10)
    >>> dr = DimensionReduction(data)
    >>> result = dr.fit(n_components=3)
    >>> dr.plot_variance()
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass


@dataclass
class PCAResult:
    """PCA 结果."""
    components: np.ndarray        # 主成分
    explained_variance: np.ndarray
    explained_variance_ratio: np.ndarray
    cumulative_variance: np.ndarray
    transformed: np.ndarray       # 降维后的数据
    n_components: int


class DimensionReduction:
    """
    降维分析.

    Parameters
    ----------
    data : array-like, shape (n_samples, n_features)
        数据矩阵
    standardize : bool
        是否标准化
    """

    def __init__(self, data, standardize=True):
        self.data = np.asarray(data, dtype=float)
        self.standardize = standardize
        self._result = None

    def fit(self, n_components=None, method="pca"):
        """
        执行降维.

        Parameters
        ----------
        n_components : int or float
            主成分数量或累计方差贡献率阈值
        method : str
            "pca" (目前只支持 PCA)
        """
        data = self.data.copy()
        if self.standardize:
            scaler = StandardScaler()
            data = scaler.fit_transform(data)

        if n_components is None:
            n_components = min(data.shape)

        if isinstance(n_components, float) and 0 < n_components < 1:
            # 按累计方差贡献率选主成分数
            pca_full = PCA()
            pca_full.fit(data)
            cumulative = np.cumsum(pca_full.explained_variance_ratio_)
            n_components = np.argmax(cumulative >= n_components) + 1

        pca = PCA(n_components=n_components)
        transformed = pca.fit_transform(data)

        cumulative = np.cumsum(pca.explained_variance_ratio_)

        self._pca = pca
        self._data_scaled = data
        self._result = PCAResult(
            components=pca.components_,
            explained_variance=pca.explained_variance_,
            explained_variance_ratio=pca.explained_variance_ratio_,
            cumulative_variance=cumulative,
            transformed=transformed,
            n_components=n_components,
        )
        return self._result

    @property
    def transformed(self):
        self._ensure_fitted()
        return self._result.transformed

    def _ensure_fitted(self):
        if self._result is None:
            raise RuntimeError("请先调用 fit()")

    def plot_variance(self, figsize=(10, 5)):
        """绘制方差贡献率."""
        import matplotlib.pyplot as plt

        self._ensure_fitted()
        r = self._result
        n = len(r.explained_variance_ratio)

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        ax = axes[0]
        ax.bar(range(1, n + 1), r.explained_variance_ratio * 100, color="steelblue")
        ax.set_xlabel("主成分")
        ax.set_ylabel("方差贡献率 (%)")
        ax.set_title("各主成分方差贡献率")
        ax.grid(True, alpha=0.3)

        ax2 = axes[1]
        ax2.plot(range(1, n + 1), r.cumulative_variance * 100, "ro-")
        ax2.axhline(y=95, linestyle="--", color="green", label="95% 阈值")
        ax2.set_xlabel("主成分数")
        ax2.set_ylabel("累计方差贡献率 (%)")
        ax2.set_title("累计方差贡献率")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def biplot(self, labels=None, figsize=(10, 8)):
        """双标图 (样本 + 特征向量)."""
        import matplotlib.pyplot as plt

        self._ensure_fitted()
        r = self._result
        if r.n_components < 2:
            raise ValueError("双标图至少需要 2 个主成分")

        fig, ax = plt.subplots(figsize=figsize)

        # 绘制样本点
        ax.scatter(r.transformed[:, 0], r.transformed[:, 1], alpha=0.5, s=20, c="steelblue")

        # 绘制特征向量
        scale = max(r.transformed[:, 0].max(), r.transformed[:, 1].max()) * 0.8
        for i, (comp0, comp1) in enumerate(zip(r.components[0], r.components[1])):
            ax.annotate("",
                        xy=(comp0 * scale, comp1 * scale),
                        xytext=(0, 0),
                        arrowprops=dict(arrowstyle="->", color="red", lw=1.5))
            name = labels[i] if labels else f"x{i+1}"
            ax.text(comp0 * scale * 1.1, comp1 * scale * 1.1, name, fontsize=9, color="red")

        ax.set_xlabel(f"PC1 ({r.explained_variance_ratio[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({r.explained_variance_ratio[1]*100:.1f}%)")
        ax.set_title("PCA 双标图")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    def summary(self, feature_names=None):
        self._ensure_fitted()
        r = self._result
        n = r.n_components
        if feature_names is None:
            feature_names = [f"x{i+1}" for i in range(r.components.shape[1])]

        lines = ["=" * 60, "  PCA 主成分分析结果", "=" * 60]
        for i in range(n):
            lines.append(f"  PC{i+1}: 方差贡献率={r.explained_variance_ratio[i]*100:.2f}%, "
                         f"累计={r.cumulative_variance[i]*100:.2f}%")
            # 主成分载荷
            loadings = sorted(zip(feature_names, r.components[i]),
                              key=lambda x: abs(x[1]), reverse=True)
            top3 = [f"{name}({val:.3f})" for name, val in loadings[:3]]
            lines.append(f"         主要载荷: {', '.join(top3)}")
        lines.append("=" * 60)
        return "\n".join(lines)
