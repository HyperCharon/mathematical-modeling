"""
网络流算法 (最大流 - Ford-Fulkerson / Edmonds-Karp)

Example:
    >>> from mathflow.graph import NetworkFlow
    >>> nf = NetworkFlow()
    >>> nf.add_edge(0, 1, 16)
    >>> nf.add_edge(0, 2, 13)
    >>> nf.add_edge(1, 2, 10)
    >>> nf.add_edge(1, 3, 12)
    >>> nf.add_edge(2, 1, 4)
    >>> nf.add_edge(2, 4, 14)
    >>> nf.add_edge(3, 2, 9)
    >>> nf.add_edge(3, 5, 20)
    >>> nf.add_edge(4, 3, 7)
    >>> nf.add_edge(4, 5, 4)
    >>> result = nf.max_flow(0, 5)
"""

import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class FlowResult:
    """网络流结果."""
    max_flow_value: float
    flow_matrix: np.ndarray
    min_cut: Tuple[List[int], List[int]]


class NetworkFlow:
    """网络流求解器."""

    def __init__(self, n_nodes=0):
        self.n_nodes = n_nodes
        self.capacity = {}

    def add_edge(self, u, v, capacity):
        """添加边 (u, v) 及其容量."""
        self.n_nodes = max(self.n_nodes, u + 1, v + 1)
        self.capacity[(u, v)] = self.capacity.get((u, v), 0) + capacity
        return self

    def max_flow(self, source, sink):
        """
        Edmonds-Karp 算法求最大流.
        """
        n = self.n_nodes
        # 构建残量图
        residual = {}
        for (u, v), cap in self.capacity.items():
            residual[(u, v)] = cap
            if (v, u) not in residual:
                residual[(v, u)] = 0

        flow = {}
        for (u, v) in self.capacity:
            flow[(u, v)] = 0

        total_flow = 0

        while True:
            # BFS 找增广路径
            parent = {source: None}
            queue = deque([source])
            found = False

            while queue:
                u = queue.popleft()
                if u == sink:
                    found = True
                    break
                for v in range(n):
                    if v not in parent and residual.get((u, v), 0) > 0:
                        parent[v] = u
                        queue.append(v)

            if not found:
                break

            # 找瓶颈容量
            path_flow = float("inf")
            v = sink
            while parent[v] is not None:
                u = parent[v]
                path_flow = min(path_flow, residual[(u, v)])
                v = u

            # 更新残量图
            v = sink
            while parent[v] is not None:
                u = parent[v]
                residual[(u, v)] -= path_flow
                residual[(v, u)] = residual.get((v, u), 0) + path_flow
                if (u, v) in flow:
                    flow[(u, v)] += path_flow
                v = u

            total_flow += path_flow

        # 构建流量矩阵
        flow_matrix = np.zeros((n, n))
        for (u, v), f in flow.items():
            flow_matrix[u][v] = f

        # 最小割 (从 source 可达的节点集合)
        visited = set()
        queue = deque([source])
        visited.add(source)
        while queue:
            u = queue.popleft()
            for v in range(n):
                if v not in visited and residual.get((u, v), 0) > 0:
                    visited.add(v)
                    queue.append(v)

        S = list(visited)
        T = [i for i in range(n) if i not in visited]

        self._result = FlowResult(
            max_flow_value=total_flow,
            flow_matrix=flow_matrix,
            min_cut=(S, T),
        )
        return self._result

    def summary(self):
        if not hasattr(self, '_result') or self._result is None:
            raise RuntimeError("请先调用 max_flow()")
        r = self._result
        lines = ["=" * 50, "  最大流结果", "=" * 50,
                  f"  最大流值: {r.max_flow_value:.2f}",
                  "-" * 50,
                  f"  最小割 S: {r.min_cut[0]}",
                  f"  最小割 T: {r.min_cut[1]}",
                  "=" * 50]
        return "\n".join(lines)
