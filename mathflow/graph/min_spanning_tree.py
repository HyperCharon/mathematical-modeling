"""
最小生成树算法 (Kruskal, Prim)

Example:
    >>> from mathflow.graph import MinSpanningTree
    >>> mst = MinSpanningTree()
    >>> mst.add_edges([(0,1,4), (0,2,1), (1,2,2), (1,3,5), (2,3,8)])
    >>> result = mst.kruskal()
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class MSTResult:
    """最小生成树结果."""
    edges: List[Tuple[int, int, float]]
    total_weight: float
    n_components: int


class MinSpanningTree:
    """最小生成树求解器."""

    def __init__(self):
        self.edges = []
        self.n_nodes = 0

    def add_edge(self, u, v, weight=1.0):
        self.edges.append((u, v, weight))
        self.n_nodes = max(self.n_nodes, u + 1, v + 1)
        return self

    def add_edges(self, edges):
        for u, v, w in edges:
            self.add_edge(u, v, w)
        return self

    def kruskal(self):
        """Kruskal 算法."""
        parent = list(range(self.n_nodes))
        rank = [0] * self.n_nodes

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            px, py = find(x), find(y)
            if px == py:
                return False
            if rank[px] < rank[py]:
                px, py = py, px
            parent[py] = px
            if rank[px] == rank[py]:
                rank[px] += 1
            return True

        sorted_edges = sorted(self.edges, key=lambda e: e[2])
        mst_edges = []
        total_weight = 0

        for u, v, w in sorted_edges:
            if union(u, v):
                mst_edges.append((u, v, w))
                total_weight += w

        # 计算连通分量数
        roots = set(find(i) for i in range(self.n_nodes))

        self._result = MSTResult(
            edges=mst_edges,
            total_weight=total_weight,
            n_components=len(roots),
        )
        return self._result

    def prim(self):
        """Prim 算法."""
        import heapq

        n = self.n_nodes
        adj = {i: [] for i in range(n)}
        for u, v, w in self.edges:
            adj[u].append((w, v, u))
            adj[v].append((w, u, v))

        visited = set()
        mst_edges = []
        total_weight = 0

        # 从节点 0 开始
        pq = adj[0][:]
        heapq.heapify(pq)
        visited.add(0)

        while pq and len(visited) < n:
            w, v, u = heapq.heappop(pq)
            if v in visited:
                continue
            visited.add(v)
            mst_edges.append((u, v, w))
            total_weight += w

            for next_w, next_v, _ in adj[v]:
                if next_v not in visited:
                    heapq.heappush(pq, (next_w, next_v, v))

        self._result = MSTResult(
            edges=mst_edges,
            total_weight=total_weight,
            n_components=n - len(visited) + 1,
        )
        return self._result

    def summary(self):
        if not hasattr(self, '_result') or self._result is None:
            raise RuntimeError("请先调用 kruskal() 或 prim()")
        r = self._result
        lines = ["=" * 50, "  最小生成树结果", "=" * 50,
                  f"  总权重: {r.total_weight:.2f}", f"  连通分量: {r.n_components}",
                  "-" * 50, "  MST 边:"]
        for u, v, w in r.edges:
            lines.append(f"    ({u}, {v})  权重={w:.2f}")
        lines.append("=" * 50)
        return "\n".join(lines)
