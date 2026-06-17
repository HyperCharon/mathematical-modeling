"""
动态规划: 背包问题

支持 0-1 背包、完全背包、多重背包。

Example:
    >>> from mathflow.dp import Knapsack
    >>> # 0-1 背包: 4个物品, 容量8
    >>> knap = Knapsack(capacity=8)
    >>> knap.add_item(weight=2, value=3, name="物品A")
    >>> knap.add_item(weight=3, value=4, name="物品B")
    >>> knap.add_item(weight=4, value=5, name="物品C")
    >>> knap.add_item(weight=5, value=6, name="物品D")
    >>> result = knap.solve_01()
"""

import numpy as np
from dataclasses import dataclass
from typing import List


@dataclass
class KnapsackResult:
    """背包问题结果."""
    max_value: float
    selected_items: List[int]   # 被选物品索引
    total_weight: float
    dp_table: np.ndarray


class Knapsack:
    """
    背包问题求解器.

    Parameters
    ----------
    capacity : float
        背包容量
    """

    def __init__(self, capacity: float):
        self.capacity = capacity
        self.items = []  # [(weight, value, name)]
        self._result_01 = None
        self._result_complete = None

    def __repr__(self) -> str:
        return f"Knapsack(capacity={self.capacity}, n_items={len(self.items)})"

    def _ensure_result(self) -> None:
        """确保至少求解了一种问题."""
        if self._result_01 is None and self._result_complete is None:
            raise RuntimeError("请先调用 solve_01() 或 solve_complete()")

    def add_item(self, weight: float, value: float, name: str = ""):
        """添加物品."""
        self.items.append((weight, value, name or f"物品{len(self.items)}"))
        return self

    def solve_01(self) -> KnapsackResult:
        """0-1 背包 (每个物品最多选一次)."""
        n = len(self.items)
        W = int(self.capacity)
        weights = [int(w) for w, _, _ in self.items]
        values = [v for _, v, _ in self.items]

        # DP 表
        dp = np.zeros((n + 1, W + 1), dtype=float)

        for i in range(1, n + 1):
            for w in range(W + 1):
                dp[i][w] = dp[i - 1][w]
                if weights[i - 1] <= w:
                    dp[i][w] = max(dp[i][w], dp[i - 1][w - weights[i - 1]] + values[i - 1])

        # 回溯找选中物品
        selected = []
        w = W
        for i in range(n, 0, -1):
            if dp[i][w] != dp[i - 1][w]:
                selected.append(i - 1)
                w -= weights[i - 1]
        selected.reverse()

        total_weight = sum(weights[i] for i in selected)

        self._result_01 = KnapsackResult(
            max_value=dp[n][W],
            selected_items=selected,
            total_weight=total_weight,
            dp_table=dp,
        )
        return self._result_01

    def solve_complete(self) -> KnapsackResult:
        """完全背包 (每个物品可选无限次)."""
        n = len(self.items)
        W = int(self.capacity)
        weights = [int(w) for w, _, _ in self.items]
        values = [v for _, v, _ in self.items]

        dp = np.zeros(W + 1)
        choice = np.zeros(W + 1, dtype=int)

        for w in range(1, W + 1):
            for i in range(n):
                if weights[i] <= w:
                    if dp[w - weights[i]] + values[i] > dp[w]:
                        dp[w] = dp[w - weights[i]] + values[i]
                        choice[w] = i

        # 回溯
        selected = []
        w = W
        while w > 0:
            i = choice[w]
            selected.append(i)
            w -= weights[i]

        self._result_complete = KnapsackResult(
            max_value=dp[W],
            selected_items=selected,
            total_weight=sum(weights[i] for i in selected),
            dp_table=dp.reshape(1, -1),
        )
        return self._result_complete

    def plot_dp(self, result_type="01", figsize=(10, 6)):
        """绘制 DP 表."""
        self._ensure_result()
        import matplotlib.pyplot as plt

        if result_type == "01":
            r = self._result_01
        else:
            r = self._result_complete

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(r.dp_table, cmap="YlOrRd", aspect="auto")
        plt.colorbar(im, ax=ax, label="最大价值")
        ax.set_xlabel("背包容量")
        ax.set_ylabel("物品数")
        ax.set_title(f"背包问题 DP 表 (最优值: {r.max_value})")
        plt.tight_layout()
        return fig

    def summary(self, result_type="01"):
        self._ensure_result()
        if result_type == "01":
            r = self._result_01
        else:
            r = self._result_complete

        lines = [
            "=" * 50,
            f"  背包问题结果 ({'0-1' if result_type == '01' else '完全背包'})",
            "=" * 50,
            f"  背包容量: {self.capacity}",
            f"  最大价值: {r.max_value}",
            f"  总重量: {r.total_weight}",
            "-" * 50,
            "  选中物品:",
        ]
        for i in r.selected_items:
            w, v, name = self.items[i]
            lines.append(f"    {name}: 重量={w}, 价值={v}")
        lines.append("=" * 50)
        return "\n".join(lines)
