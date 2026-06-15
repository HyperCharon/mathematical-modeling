"""动态规划模块测试."""

import numpy as np
import pytest
from mathflow.dp import Knapsack, ResourceAllocation


class TestKnapsack:
    def test_01_basic(self):
        knap = Knapsack(capacity=8)
        knap.add_item(weight=2, value=3)
        knap.add_item(weight=3, value=4)
        knap.add_item(weight=4, value=5)
        knap.add_item(weight=5, value=6)
        result = knap.solve_01()
        assert result.max_value > 0
        assert result.total_weight <= 8

    def test_01_simple(self):
        knap = Knapsack(capacity=10)
        knap.add_item(weight=5, value=10)
        knap.add_item(weight=5, value=10)
        result = knap.solve_01()
        assert result.max_value == 20  # 两个都选

    def test_complete(self):
        knap = Knapsack(capacity=10)
        knap.add_item(weight=3, value=4)
        knap.add_item(weight=4, value=5)
        result = knap.solve_complete()
        assert result.max_value > 0


class TestResourceAllocation:
    def test_basic(self):
        profit = [
            [0, 3, 7, 9, 12],
            [0, 5, 10, 12, 14],
            [0, 4, 6, 11, 12],
        ]
        ra = ResourceAllocation(profit, total_resource=4)
        result = ra.solve()
        assert result.max_profit > 0
        assert sum(result.allocation) == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
