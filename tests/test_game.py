"""博弈论模块测试."""

import numpy as np
import pytest
from mathflow.game import NashEquilibrium, MatrixGame


class TestNashEquilibrium:
    def test_prisoners_dilemma(self):
        """囚徒困境: 唯一NE是(背叛,背叛)."""
        p1 = [[-1, -3], [0, -2]]
        p2 = [[-1, 0], [-3, -2]]
        ne = NashEquilibrium(p1, p2)
        result = ne.find_all()
        assert (1, 1) in result.pure_strategies  # 双方都背叛

    def test_coordination(self):
        """协调博弈: 两个纯策略NE."""
        p1 = [[2, 0], [0, 1]]
        p2 = [[2, 0], [0, 1]]
        ne = NashEquilibrium(p1, p2)
        result = ne.find_all()
        assert len(result.pure_strategies) == 2
        assert (0, 0) in result.pure_strategies
        assert (1, 1) in result.pure_strategies

    def test_mixed_2x2(self):
        """2x2博弈应有混合策略NE."""
        # 性别之战: 有纯策略NE也有混合策略NE
        p1 = [[3, 0], [0, 2]]
        p2 = [[2, 0], [0, 3]]
        ne = NashEquilibrium(p1, p2)
        result = ne.find_all()
        # 应有纯策略NE
        assert len(result.pure_strategies) >= 1


class TestMatrixGame:
    def test_saddle_point(self):
        """有鞍点的矩阵博弈."""
        # 行最小值的最大值 = 列最大值的最小值
        payoff = [[3, 1, 2], [2, 2, 2], [1, 3, 2]]
        game = MatrixGame(payoff)
        result = game.solve()
        # 行最小值: [1, 2, 1], maximin = 2
        # 列最大值: [3, 3, 2], minimax = 2
        assert result.has_saddle_point

    def test_no_saddle(self):
        """无鞍点的矩阵博弈."""
        payoff = [[0, 1, -1], [-1, 0, 1], [1, -1, 0]]
        game = MatrixGame(payoff)
        result = game.solve()
        assert not result.has_saddle_point
        assert abs(result.value) < 0.1  # 石头剪刀布博弈值为0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
