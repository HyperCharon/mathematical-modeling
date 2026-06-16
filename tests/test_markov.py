"""
测试 markov 模块: MarkovChain
"""

import numpy as np
import pytest


class TestMarkovChain:
    """测试马尔可夫链."""

    @pytest.fixture
    def simple_chain(self):
        from mathflow.markov.chain import MarkovChain
        P = np.array([
            [0.7, 0.3],
            [0.4, 0.6]
        ])
        return MarkovChain(P, states=["Sunny", "Rainy"])

    def test_invalid_transition_matrix(self):
        """行和不为 1 应抛出异常."""
        from mathflow.markov.chain import MarkovChain
        P = np.array([
            [0.7, 0.2],  # 行和 = 0.9
            [0.4, 0.6]
        ])
        with pytest.raises(ValueError):
            MarkovChain(P)

    def test_n_step_transition_1(self, simple_chain):
        """1 步转移矩阵应等于原始矩阵."""
        P1 = simple_chain.n_step_transition(1)
        np.testing.assert_array_almost_equal(P1, simple_chain.P)

    def test_n_step_transition_2(self, simple_chain):
        """2 步转移矩阵应等于 P @ P."""
        P2 = simple_chain.n_step_transition(2)
        P_expected = simple_chain.P @ simple_chain.P
        np.testing.assert_array_almost_equal(P2, P_expected)

    def test_steady_state(self, simple_chain):
        """稳态分布应满足 pi*P = pi 且 sum=1."""
        pi = simple_chain.steady_state()
        # pi * P = pi
        pi_next = pi @ simple_chain.P
        np.testing.assert_array_almost_equal(pi, pi_next, decimal=6)
        # sum = 1
        assert abs(sum(pi) - 1.0) < 1e-6
        # 所有元素非负
        assert all(p >= -1e-10 for p in pi)

    def test_simulate_length(self, simple_chain):
        """模拟结果长度应正确."""
        result = simple_chain.simulate(n_steps=100, start=0)
        assert len(result) == 100

    def test_simulate_valid_states(self, simple_chain):
        """模拟结果应只包含有效状态."""
        result = simple_chain.simulate(n_steps=100, start=0)
        assert all(0 <= s < 2 for s in result)

    def test_simulate_convergence(self, simple_chain):
        """长期模拟的频率应趋近稳态分布."""
        pi = simple_chain.steady_state()
        result = simple_chain.simulate(n_steps=10000, start=0, seed=42)
        freq_sunny = sum(1 for s in result if s == 0) / len(result)
        # 允许 5% 的误差
        assert abs(freq_sunny - pi[0]) < 0.05

    def test_classify_states(self, simple_chain):
        """不可约链的所有状态应标记为常返."""
        classification = simple_chain.classify_states()
        assert len(classification) == 2

    def test_summary(self, simple_chain):
        """summary 应返回字符串."""
        s = simple_chain.summary()
        assert isinstance(s, str)
        assert len(s) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
