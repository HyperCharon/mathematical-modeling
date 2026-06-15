"""
聚类分析

支持 KMeans、DBSCAN、层次聚类，自动选 K (肘部法 + 轮廓系数)。

Example:
    >>> from mathflow.ml import ClusterAnalysis
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> data = np.vstack([
    ...     np.random.randn(50, 2) + [0, 0],
    ...     np.random.randn(50, 2) + [5, 5],
    ...     np.random.randn(50, 2) + [10, 0],
    ... ])
    >>> ca = ClusterAnalysis(data)
    >>> result = ca.auto_k(max_k=8)
    >>> labels = ca.fit(method="kmeans")
"""

import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass
from typing import Optional


@dataclass
class ClusterResult:
    """聚类结果."""
    labels: np.ndarray
    n_clusters: int
    method: str
    silhouette: float
    calinski_harabasz: float
    centers: Optional[np.ndarray]


class ClusterAnalysis:
    """
    聚类分析.

    Parameters
    ----------
    data : array-like, shape (n_samples, n_features)
        数据矩阵
    standardize : bool
        是否标准化 (默认 True)
    """

    def __init__(self, data, standardize=True):
        self.data = np.asarray(data, dtype=float)
        self.standardize = standardize
        self._scaler = None
        self._data_scaled = None
        self._result = None
        self._k_scores = None

        if standardize:
            self._scaler = StandardScaler()
            self._data_scaled = self._scaler.fit_transform(self.data)
        else:
            self._data_scaled = self.data.copy()

    def auto_k(self, max_k=10, method="elbow_silhouette"):
        """
        自动选择最优 K.

        Parameters
        ----------
        max_k : int
            最大 K 值
        method : str
            "elbow_silhouette" (肘部法 + 轮廓系数)
        """
        k_range = range(2, min(max_k + 1, len(self.data)))
        inertias = []
        silhouettes = []

        for k in k_range:
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            labels = km.fit_predict(self._data_scaled)
            inertias.append(km.inertia_)
            silhouettes.append(silhouette_score(self._data_scaled, labels))

        self._k_scores = {
            "k_range": list(k_range),
            "inertias": inertias,
            "silhouettes": silhouettes,
        }

        # 选择轮廓系数最大的 K
        best_k = list(k_range)[np.argmax(silhouettes)]
        self._auto_k = best_k
        return best_k

    def fit(self, method="kmeans", n_clusters=None):
        """
        执行聚类.

        Parameters
        ----------
        method : str
            "kmeans", "dbscan", "hierarchical"
        n_clusters : int, optional
            聚类数 (默认使用 auto_k 的结果)
        """
        if n_clusters is None:
            n_clusters = getattr(self, '_auto_k', 3)

        data = self._data_scaled

        if method == "kmeans":
            model = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
            labels = model.fit_predict(data)
            centers = model.cluster_centers_
        elif method == "dbscan":
            model = DBSCAN(eps=0.5, min_samples=5)
            labels = model.fit_predict(data)
            centers = None
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        elif method == "hierarchical":
            model = AgglomerativeClustering(n_clusters=n_clusters)
            labels = model.fit_predict(data)
            centers = None
        else:
            raise ValueError(f"未知方法: {method}")

        # 排除噪声点计算指标
        mask = labels >= 0
        if mask.sum() > n_clusters and n_clusters > 1:
            sil = silhouette_score(data[mask], labels[mask])
            ch = calinski_harabasz_score(data[mask], labels[mask])
        else:
            sil, ch = -1, 0

        self._result = ClusterResult(
            labels=labels,
            n_clusters=n_clusters,
            method=method,
            silhouette=sil,
            calinski_harabasz=ch,
            centers=centers,
        )
        return labels

    def plot_elbow(self, figsize=(12, 5)):
        """绘制肘部图和轮廓系数图."""
        import matplotlib.pyplot as plt

        if self._k_scores is None:
            raise RuntimeError("请先调用 auto_k()")

        ks = self._k_scores["k_range"]
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        ax = axes[0]
        ax.plot(ks, self._k_scores["inertias"], "bo-")
        ax.set_xlabel("K")
        ax.set_ylabel("惯性 (Inertia)")
        ax.set_title("肘部法")
        ax.grid(True, alpha=0.3)

        ax2 = axes[1]
        ax2.plot(ks, self._k_scores["silhouettes"], "ro-")
        best_k = ks[np.argmax(self._k_scores["silhouettes"])]
        ax2.axvline(x=best_k, linestyle="--", color="green", label=f"最优 K={best_k}")
        ax2.set_xlabel("K")
        ax2.set_ylabel("轮廓系数")
        ax2.set_title("轮廓系数法")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_clusters(self, figsize=(8, 8)):
        """绘制聚类结果 (2D)."""
        import matplotlib.pyplot as plt

        self._ensure_fitted()
        r = self._result
        data = self.data
        labels = r.labels

        fig, ax = plt.subplots(figsize=figsize)
        unique_labels = sorted(set(labels))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

        for label, color in zip(unique_labels, colors):
            mask = labels == label
            name = f"簇 {label}" if label >= 0 else "噪声"
            ax.scatter(data[mask, 0], data[mask, 1], c=[color], label=name, alpha=0.6, s=30)

        if r.centers is not None and self._scaler is not None:
            centers_orig = self._scaler.inverse_transform(r.centers)
            ax.scatter(centers_orig[:, 0], centers_orig[:, 1], c="red", marker="X", s=200, label="聚类中心")

        ax.set_title(f"{r.method.title()} 聚类 (K={r.n_clusters}, 轮廓系数={r.silhouette:.3f})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    def evaluate_external(self, true_labels):
        """
        外部评估: 与真实标签对比.

        Parameters
        ----------
        true_labels : array-like
            真实标签

        Returns
        -------
        dict
            ARI (调整兰德指数), NMI (标准化互信息), AMI (调整互信息)
        """
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score

        self._ensure_fitted()
        pred = self._result.labels

        # 排除噪声点
        mask = pred >= 0
        if mask.sum() == 0:
            return {"ARI": 0, "NMI": 0, "AMI": 0}

        true = np.asarray(true_labels)[mask]
        pred_clean = pred[mask]

        ari = adjusted_rand_score(true, pred_clean)
        nmi = normalized_mutual_info_score(true, pred_clean)
        ami = adjusted_mutual_info_score(true, pred_clean)

        return {"ARI": ari, "NMI": nmi, "AMI": ami}

    def _ensure_fitted(self):
        if self._result is None:
            raise RuntimeError("请先调用 fit()")

    def summary(self):
        self._ensure_fitted()
        r = self._result
        lines = ["=" * 50, f"  {r.method.title()} 聚类结果", "=" * 50,
                  f"  聚类数: {r.n_clusters}",
                  f"  轮廓系数: {r.silhouette:.4f}",
                  f"  CH 指数: {r.calinski_harabasz:.2f}",
                  "-" * 50]
        for i in range(r.n_clusters):
            count = (r.labels == i).sum()
            lines.append(f"  簇 {i}: {count} 个样本")
        if (r.labels == -1).any():
            lines.append(f"  噪声: {(r.labels == -1).sum()} 个样本")
        lines.append("=" * 50)
        return "\n".join(lines)
