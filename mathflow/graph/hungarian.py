"""
匈牙利算法 (Hungarian Algorithm)

求解指派问题 (Assignment Problem): n 个人分配 n 个任务，最小化总成本。

Example:
    >>> from mathflow.graph import Hungarian
    >>> cost = np.array([
    ...     [4, 1, 3],
    ...     [2, 0, 5],
    ...     [3, 2, 2],
    ... ])
    >>> h = Hungarian(cost)
    >>> result = h.solve()
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
from scipy.optimize import linear_sum_assignment


@dataclass
class AssignmentResult:
    """指派结果."""
    assignment: List[Tuple[int, int]]  # (row, col) 指派对
    total_cost: float
    cost_matrix: np.ndarray


class Hungarian:
    """
    匈牙利算法求解指派问题.

    Parameters
    ----------
    cost_matrix : array-like, shape (n, n)
        成本矩阵
    maximize : bool
        True 则求最大收益指派
    """

    def __init__(self, cost_matrix, maximize=False):
        self.cost_matrix = np.asarray(cost_matrix, dtype=float)
        self.maximize = maximize
        if self.cost_matrix.ndim != 2:
            raise ValueError("成本矩阵必须是二维的")

    def solve(self) -> AssignmentResult:
        """求解指派问题."""
        cost = self.cost_matrix.copy()

        if self.maximize:
            cost = cost.max() - cost

        row_ind, col_ind = linear_sum_assignment(cost)
        assignment = list(zip(row_ind.tolist(), col_ind.tolist()))
        total_cost = sum(self.cost_matrix[r, c] for r, c in assignment)

        self._result = AssignmentResult(
            assignment=assignment,
            total_cost=total_cost,
            cost_matrix=self.cost_matrix,
        )
        return self._result

    def plot(self, figsize=(8, 6)):
        """可视化指派结果."""
        import matplotlib.pyplot as plt

        self._ensure_result()
        r = self._result
        n = r.cost_matrix.shape[0]

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(r.cost_matrix, cmap="YlOrRd", aspect="auto")
        plt.colorbar(im, ax=ax, label="成本")

        for row, col in r.assignment:
            ax.add_patch(plt.Rectangle((col - 0.5, row - 0.5), 1, 1,
                                       fill=False, edgecolor="blue", linewidth=3))

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xlabel("任务")
        ax.set_ylabel("人员")
        ax.set_title(f"指派结果 (总成本: {r.total_cost:.2f})")
        plt.tight_layout()
        return fig

    def _ensure_result(self):
        if not hasattr(self, '_result') or self._result is None:
            raise RuntimeError("请先调用 solve()")

    def summary(self):
        self._ensure_result()
        r = self._result
        lines = [
            "=" * 50,
            "  匈牙利算法 - 指派问题结果",
            "=" * 50,
            f"  总成本: {r.total_cost:.2f}",
            "-" * 50,
            "  指派方案:",
        ]
        for row, col in r.assignment:
            lines.append(f"    人员{row+1} → 任务{col+1} (成本={r.cost_matrix[row, col]:.2f})")
        lines.append("=" * 50)
        return "\n".join(lines)
