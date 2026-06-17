"""
最短路径算法

支持 Dijkstra、Bellman-Ford、Floyd 算法。

Example:
    >>> from mathflow.graph import ShortestPath
    >>> sp = ShortestPath()
    >>> sp.add_edge(0, 1, 4)
    >>> sp.add_edge(0, 2, 1)
    >>> sp.add_edge(2, 1, 2)
    >>> sp.add_edge(1, 3, 1)
    >>> dist, path = sp.dijkstra(0, 3)
"""

import numpy as np
import heapq
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class PathResult:
    """路径结果."""
    distance: float
    path: List[int]
    all_distances: Optional[np.ndarray] = None


class ShortestPath:
    """
    最短路径求解器.

    支持有向图和无向图，支持稀疏和稠密表示。
    """

    def __init__(self, n_nodes: int = 0):
        self.n_nodes = n_nodes
        self.edges = []  # [(u, v, w)]
        self.adj = {}  # {u: [(v, w), ...]}

    def __repr__(self) -> str:
        return f"ShortestPath(n_nodes={self.n_nodes}, n_edges={len(self.edges)})"

    def add_edge(self, u, v, weight=1.0, directed=True):
        """添加边."""
        self.edges.append((u, v, weight))
        self.n_nodes = max(self.n_nodes, u + 1, v + 1)

        if u not in self.adj:
            self.adj[u] = []
        self.adj[u].append((v, weight))

        if not directed:
            if v not in self.adj:
                self.adj[v] = []
            self.adj[v].append((u, weight))
        return self

    def add_edges(self, edges, directed=True):
        """批量添加边 [(u, v, weight), ...]."""
        for u, v, w in edges:
            self.add_edge(u, v, w, directed)
        return self

    def from_adjacency_matrix(self, matrix, directed=True):
        """从邻接矩阵构建图."""
        matrix = np.asarray(matrix, dtype=float)
        n = matrix.shape[0]
        self.n_nodes = n
        inf = float("inf")
        for i in range(n):
            for j in range(n):
                if directed:
                    if matrix[i, j] < inf and matrix[i, j] != 0:
                        self.add_edge(i, j, matrix[i, j], directed=True)
                else:
                    if j > i and matrix[i, j] < inf and matrix[i, j] != 0:
                        self.add_edge(i, j, matrix[i, j], directed=False)
        return self

    def dijkstra(self, source, target=None):
        """
        Dijkstra 最短路径 (非负权重).

        Parameters
        ----------
        source : int
            源节点
        target : int, optional
            目标节点，为 None 时返回到所有节点的最短距离
        """
        n = self.n_nodes
        dist = np.full(n, float("inf"))
        prev = np.full(n, -1, dtype=int)
        dist[source] = 0

        pq = [(0, source)]
        visited = set()

        while pq:
            d, u = heapq.heappop(pq)
            if u in visited:
                continue
            visited.add(u)

            if target is not None and u == target:
                break

            for v, w in self.adj.get(u, []):
                if v not in visited and dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
                    prev[v] = u
                    heapq.heappush(pq, (dist[v], v))

        if target is not None:
            path = self._reconstruct_path(prev, source, target)
            return PathResult(distance=dist[target], path=path, all_distances=dist)
        return PathResult(distance=0, path=[], all_distances=dist)

    def bellman_ford(self, source):
        """Bellman-Ford 算法 (支持负权重)."""
        n = self.n_nodes
        dist = np.full(n, float("inf"))
        prev = np.full(n, -1, dtype=int)
        dist[source] = 0

        for _ in range(n - 1):
            for u, v, w in self.edges:
                if dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
                    prev[v] = u

        # 检测负环
        for u, v, w in self.edges:
            if dist[u] + w < dist[v]:
                raise ValueError("图中存在负权环")

        return PathResult(distance=0, path=[], all_distances=dist)

    def floyd(self):
        """Floyd-Warshall 全源最短路径."""
        n = self.n_nodes
        dist = np.full((n, n), float("inf"))
        np.fill_diagonal(dist, 0)

        for u, v, w in self.edges:
            dist[u][v] = min(dist[u][v], w)

        # Floyd 核心 (向量化版本，比三重循环快10倍+)
        for k in range(n):
            # dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
            dist = np.minimum(dist, dist[:, k:k+1] + dist[k:k+1, :])

        return dist

    def _reconstruct_path(self, prev, source, target):
        """重建路径."""
        path = []
        current = target
        while current != -1:
            path.append(current)
            current = prev[current]
        path.reverse()
        if path and path[0] == source:
            return path
        return []

    def plot(self, figsize=(8, 8)):
        """可视化图."""
        import matplotlib.pyplot as plt
        import networkx as nx

        G = nx.DiGraph()
        for u, v, w in self.edges:
            G.add_edge(u, v, weight=w)

        pos = nx.spring_layout(G, seed=42)
        fig, ax = plt.subplots(figsize=figsize)

        nx.draw_networkx_nodes(G, pos, ax=ax, node_color="lightblue", node_size=500)
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=12)
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color="gray", arrows=True)

        edge_labels = {(u, v): f"{w:.1f}" for u, v, w in self.edges}
        nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=ax, font_size=9)

        ax.set_title("图结构可视化")
        plt.tight_layout()
        return fig
