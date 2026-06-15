"""
动态规划: 资源分配问题

将有限资源分配给多个活动，使总收益最大。

Example:
    >>> from mathflow.dp import ResourceAllocation
    >>> # 将5台设备分配给3个车间
    >>> # profit[i][j] = 将j台设备分配给车间i的收益
    >>> profit = [
    ...     [0, 3, 7, 9, 12, 13],  # 车间A
    ...     [0, 5, 10, 12, 14, 15], # 车间B
    ...     [0, 4, 6, 11, 12, 13],  # 车间C
    ... ]
    >>> ra = ResourceAllocation(profit, total_resource=5)
    >>> result = ra.solve()
"""

import numpy as np
from dataclasses import dataclass
from typing import List


@dataclass
class AllocationResult:
    """资源分配结果."""
    max_profit: float
    allocation: List[int]   # 每个活动分配的资源量
    dp_table: np.ndarray


class ResourceAllocation:
    """
    资源分配问题 (动态规划).

    Parameters
    ----------
    profit : list of list
        profit[i][j] = 将 j 单位资源分配给活动 i 的收益
    total_resource : int
        总资源量
    """

    def __init__(self, profit: List[List[float]], total_resource: int):
        self.profit = [np.array(p) for p in profit]
        self.n_activities = len(profit)
        self.total_resource = total_resource

    def solve(self) -> AllocationResult:
        """求解资源分配问题."""
        n = self.n_activities
        R = self.total_resource

        # dp[k][r] = 前k个活动分配r个资源的最大收益
        dp = np.zeros((n + 1, R + 1))
        alloc = np.zeros((n + 1, R + 1), dtype=int)

        for k in range(1, n + 1):
            for r in range(R + 1):
                best = 0
                best_j = 0
                max_assign = min(r, len(self.profit[k - 1]) - 1)
                for j in range(max_assign + 1):
                    val = dp[k - 1][r - j] + self.profit[k - 1][j]
                    if val > best:
                        best = val
                        best_j = j
                dp[k][r] = best
                alloc[k][r] = best_j

        # 回溯
        allocation = []
        r = R
        for k in range(n, 0, -1):
            j = alloc[k][r]
            allocation.append(j)
            r -= j
        allocation.reverse()

        self._result = AllocationResult(
            max_profit=dp[n][R],
            allocation=allocation,
            dp_table=dp,
        )
        return self._result

    @property
    def result(self):
        if not hasattr(self, '_result') or self._result is None:
            raise RuntimeError("请先调用 solve()")
        return self._result

    def summary(self):
        r = self.result
        lines = [
            "=" * 50,
            "  资源分配问题结果 (动态规划)",
            "=" * 50,
            f"  总资源: {self.total_resource}",
            f"  最大收益: {r.max_profit}",
            "-" * 50,
            "  分配方案:",
        ]
        for i, a in enumerate(r.allocation):
            lines.append(f"    活动{i+1}: {a} 单位资源, 收益={self.profit[i][a]}")
        lines.append("=" * 50)
        return "\n".join(lines)
