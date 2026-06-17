"""
网络分析模块

提供网络中心性指标和社区发现算法。

Example:
    >>> from mathflow.graph import NetworkAnalysis
    >>> import numpy as np
    >>> # 邻接矩阵
    >>> adj = np.array([
    ...     [0, 1, 1, 0, 0],
    ...     [1, 0, 1, 1, 0],
    ...     [1, 1, 0, 0, 1],
    ...     [0, 1, 0, 0, 1],
    ...     [0, 0, 1, 1, 0],
    ... ])
    >>> na = NetworkAnalysis(adj)
    >>> centrality = na.degree_centrality()
    >>> communities = na.community_detection()
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from collections import defaultdict, deque


@dataclass
class CentralityResult:
    """中心性指标结果."""
    degree: np.ndarray          # 度中心性
    betweenness: np.ndarray     # 介数中心性
    closeness: np.ndarray       # 接近中心性
    eigenvector: np.ndarray     # 特征向量中心性
    pagerank: np.ndarray        # PageRank值
    n_nodes: int


@dataclass
class CommunityResult:
    """社区发现结果."""
    communities: List[List[int]]  # 社区列表
    n_communities: int            # 社区数量
    modularity: float             # 模块度
    membership: np.ndarray        # 节点所属社区


class NetworkAnalysis:
    """
    网络分析工具.

    Parameters
    ----------
    adjacency : array-like, shape (n_nodes, n_nodes)
        邻接矩阵 (对称矩阵表示无向图)
    directed : bool
        是否为有向图
    """

    def __init__(self, adjacency, directed=False):
        self.adjacency = np.asarray(adjacency, dtype=float)
        if self.adjacency.ndim != 2:
            raise ValueError("邻接矩阵必须是二维的")
        if self.adjacency.shape[0] != self.adjacency.shape[1]:
            raise ValueError("邻接矩阵必须是方阵")

        self.n_nodes = self.adjacency.shape[0]
        self.directed = directed

        # 构建邻接列表
        self.adj_list = defaultdict(list)
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if self.adjacency[i, j] > 0:
                    self.adj_list[i].append(j)

    def __repr__(self) -> str:
        return f"NetworkAnalysis(n_nodes={self.n_nodes}, directed={self.directed})"

    def degree_centrality(self) -> np.ndarray:
        """
        计算度中心性.

        度中心性 = 节点的度 / (n-1)
        """
        if self.directed:
            # 有向图: 入度 + 出度
            in_degree = np.sum(self.adjacency > 0, axis=0)
            out_degree = np.sum(self.adjacency > 0, axis=1)
            degree = (in_degree + out_degree) / (2 * (self.n_nodes - 1))
        else:
            # 无向图
            degree = np.sum(self.adjacency > 0, axis=1) / (self.n_nodes - 1)

        return degree

    def betweenness_centrality(self) -> np.ndarray:
        """
        计算介数中心性.

        介数中心性衡量节点作为中介的程度。
        """
        betweenness = np.zeros(self.n_nodes)

        for s in range(self.n_nodes):
            # BFS 找最短路径
            stack = []
            pred = defaultdict(list)
            sigma = np.zeros(self.n_nodes)
            sigma[s] = 1
            dist = np.full(self.n_nodes, -1)
            dist[s] = 0
            queue = deque([s])

            while queue:
                v = queue.popleft()
                stack.append(v)
                for w in self.adj_list[v]:
                    # 第一次发现w
                    if dist[w] < 0:
                        dist[w] = dist[v] + 1
                        queue.append(w)
                    # 最短路径经过v
                    if dist[w] == dist[v] + 1:
                        sigma[w] += sigma[v]
                        pred[w].append(v)

            # 回溯计算依赖
            delta = np.zeros(self.n_nodes)
            while stack:
                w = stack.pop()
                for v in pred[w]:
                    delta[v] += (sigma[v] / sigma[w]) * (1 + delta[w])
                if w != s:
                    betweenness[w] += delta[w]

        # 归一化
        if not self.directed:
            betweenness /= 2
        betweenness /= (self.n_nodes - 1) * (self.n_nodes - 2)

        return betweenness

    def closeness_centrality(self) -> np.ndarray:
        """
        计算接近中心性.

        接近中心性 = (n-1) / 节点到其他所有节点的最短路径之和
        """
        closeness = np.zeros(self.n_nodes)

        for s in range(self.n_nodes):
            # BFS 计算最短路径
            dist = np.full(self.n_nodes, -1)
            dist[s] = 0
            queue = deque([s])

            while queue:
                v = queue.popleft()
                for w in self.adj_list[v]:
                    if dist[w] < 0:
                        dist[w] = dist[v] + 1
                        queue.append(w)

            # 计算接近中心性
            reachable = np.sum(dist > 0)
            if reachable > 0:
                closeness[s] = reachable / np.sum(dist[dist > 0])

        return closeness

    def eigenvector_centrality(self, max_iter=100, tol=1e-6) -> np.ndarray:
        """
        计算特征向量中心性.

        特征向量中心性 = 邻接矩阵主特征向量
        """
        # 幂迭代法
        x = np.ones(self.n_nodes) / self.n_nodes

        for _ in range(max_iter):
            x_new = self.adjacency @ x
            norm = np.linalg.norm(x_new)
            if norm > 0:
                x_new = x_new / norm
            if np.linalg.norm(x_new - x) < tol:
                break
            x = x_new

        return x

    def pagerank(self, damping=0.85, max_iter=100, tol=1e-6) -> np.ndarray:
        """
        计算 PageRank 值.

        Parameters
        ----------
        damping : float
            阻尼系数 (通常为0.85)
        max_iter : int
            最大迭代次数
        tol : float
            收敛精度
        """
        n = self.n_nodes
        # 转移矩阵
        M = self.adjacency.copy()
        for j in range(n):
            col_sum = M[:, j].sum()
            if col_sum > 0:
                M[:, j] /= col_sum

        # 迭代
        pr = np.ones(n) / n
        for _ in range(max_iter):
            pr_new = (1 - damping) / n + damping * M @ pr
            if np.linalg.norm(pr_new - pr) < tol:
                break
            pr = pr_new

        return pr / pr.sum()

    def compute_all_centrality(self) -> CentralityResult:
        """计算所有中心性指标."""
        return CentralityResult(
            degree=self.degree_centrality(),
            betweenness=self.betweenness_centrality(),
            closeness=self.closeness_centrality(),
            eigenvector=self.eigenvector_centrality(),
            pagerank=self.pagerank(),
            n_nodes=self.n_nodes,
        )

    def community_detection(self, method="louvain") -> CommunityResult:
        """
        社区发现.

        Parameters
        ----------
        method : str
            "louvain" (Louvain算法) 或 "label_propagation" (标签传播)
        """
        if method == "louvain":
            return self._louvain()
        elif method == "label_propagation":
            return self._label_propagation()
        else:
            raise ValueError(f"未知方法: {method}")

    def _louvain(self, resolution=1.0) -> CommunityResult:
        """Louvain 社区发现算法."""
        n = self.n_nodes
        # 初始化: 每个节点是一个社区
        membership = np.arange(n)

        # 计算模块度增益
        m = np.sum(self.adjacency) / 2  # 总边数
        if m == 0:
            return CommunityResult(
                communities=[[i] for i in range(n)],
                n_communities=n,
                modularity=0.0,
                membership=membership,
            )

        # 度数
        k = np.sum(self.adjacency, axis=1)

        # 迭代优化
        improved = True
        while improved:
            improved = False
            for i in range(n):
                # 计算将节点i移动到每个邻居社区的模块度增益
                best_gain = 0
                best_community = membership[i]

                # 获取邻居社区
                neighbor_communities = set()
                for j in self.adj_list[i]:
                    neighbor_communities.add(membership[j])

                # 当前社区的内部边权重
                current_community = membership[i]
                ki_in = 0
                for j in self.adj_list[i]:
                    if membership[j] == current_community:
                        ki_in += self.adjacency[i, j]

                # 计算当前社区的总度数
                sigma_tot = 0
                for j in range(n):
                    if membership[j] == current_community:
                        sigma_tot += k[j]

                # 尝试移动到每个邻居社区
                for c in neighbor_communities:
                    if c == current_community:
                        continue

                    # 计算移动到社区c的内部边权重
                    ki_in_new = 0
                    for j in self.adj_list[i]:
                        if membership[j] == c:
                            ki_in_new += self.adjacency[i, j]

                    # 计算社区c的总度数
                    sigma_tot_new = 0
                    for j in range(n):
                        if membership[j] == c:
                            sigma_tot_new += k[j]

                    # 模块度增益
                    gain = (ki_in_new - ki_in) / (2 * m) - \
                           resolution * k[i] * (sigma_tot_new - sigma_tot + k[i]) / (2 * m)**2

                    if gain > best_gain:
                        best_gain = gain
                        best_community = c

                if best_community != membership[i]:
                    membership[i] = best_community
                    improved = True

        # 提取社区
        community_dict = defaultdict(list)
        for i, c in enumerate(membership):
            community_dict[c].append(i)
        communities = list(community_dict.values())

        # 计算模块度
        modularity = self._compute_modularity(membership)

        return CommunityResult(
            communities=communities,
            n_communities=len(communities),
            modularity=modularity,
            membership=membership,
        )

    def _label_propagation(self, max_iter=100) -> CommunityResult:
        """标签传播算法."""
        n = self.n_nodes
        labels = np.arange(n)

        for _ in range(max_iter):
            changed = False
            nodes = np.random.permutation(n)

            for i in nodes:
                # 统计邻居标签
                label_counts = defaultdict(int)
                for j in self.adj_list[i]:
                    label_counts[labels[j]] += self.adjacency[i, j]

                if label_counts:
                    # 选择出现最多的标签
                    max_count = max(label_counts.values())
                    candidates = [l for l, c in label_counts.items() if c == max_count]
                    new_label = np.random.choice(candidates)

                    if new_label != labels[i]:
                        labels[i] = new_label
                        changed = True

            if not changed:
                break

        # 提取社区
        community_dict = defaultdict(list)
        for i, c in enumerate(labels):
            community_dict[c].append(i)
        communities = list(community_dict.values())

        modularity = self._compute_modularity(labels)

        return CommunityResult(
            communities=communities,
            n_communities=len(communities),
            modularity=modularity,
            membership=labels,
        )

    def _compute_modularity(self, membership) -> float:
        """计算模块度."""
        m = np.sum(self.adjacency) / 2
        if m == 0:
            return 0.0

        Q = 0.0
        k = np.sum(self.adjacency, axis=1)

        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if membership[i] == membership[j]:
                    Q += self.adjacency[i, j] - k[i] * k[j] / (2 * m)

        return Q / (2 * m)

    def summary(self) -> str:
        """打印网络摘要."""
        n_edges = np.sum(self.adjacency > 0)
        if not self.directed:
            n_edges //= 2

        lines = [
            "=" * 50,
            "  网络分析摘要",
            "=" * 50,
            f"  节点数: {self.n_nodes}",
            f"  边数: {int(n_edges)}",
            f"  类型: {'有向图' if self.directed else '无向图'}",
            f"  密度: {n_edges / (self.n_nodes * (self.n_nodes - 1)):.4f}",
        ]

        # 计算中心性
        centrality = self.compute_all_centrality()

        lines.append("-" * 50)
        lines.append("  中心性指标 (Top 5):")
        lines.append(f"  {'节点':>6s}  {'度中心性':>10s}  {'介数中心性':>10s}  {'PageRank':>10s}")
        lines.append("-" * 50)

        # 按PageRank排序
        top_indices = np.argsort(-centrality.pagerank)[:5]
        for i in top_indices:
            lines.append(
                f"  Node {i:>3d}  {centrality.degree[i]:>10.4f}  "
                f"{centrality.betweenness[i]:>10.4f}  {centrality.pagerank[i]:>10.4f}"
            )

        # 社区发现
        community = self.community_detection()
        lines.append("-" * 50)
        lines.append(f"  社区数量: {community.n_communities}")
        lines.append(f"  模块度: {community.modularity:.4f}")

        lines.append("=" * 50)
        return "\n".join(lines)
