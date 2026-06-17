"""
测试 simulation 模块: MonteCarlo, QueueModel
"""

import numpy as np
import pytest


class TestMonteCarlo:
    """测试蒙特卡洛模拟."""

    def test_estimate_pi(self):
        """估算 pi 应接近 3.14159."""
        from mathflow.simulation.monte_carlo import MonteCarlo
        mc = MonteCarlo(seed=42)
        pi_est, error = mc.estimate_pi(n_samples=100000)
        assert abs(pi_est - np.pi) < 0.05

    def test_integrate_x_squared(self):
        """积分 x^2 从 0 到 1 应接近 1/3."""
        from mathflow.simulation.monte_carlo import MonteCarlo
        mc = MonteCarlo(seed=42)
        result = mc.integrate(lambda x: x**2, 0, 1, n_samples=100000)
        assert abs(result.estimate - 1/3) < 0.01

    def test_integrate_confidence_interval(self):
        """置信区间应包含真实值."""
        from mathflow.simulation.monte_carlo import MonteCarlo
        mc = MonteCarlo(seed=42)
        result = mc.integrate(lambda x: x, 0, 1, n_samples=100000)
        true_value = 0.5
        assert result.confidence_interval[0] <= true_value <= result.confidence_interval[1]


class TestQueueModel:
    """测试排队论模型."""

    def test_mm1_basic(self):
        """M/M/1 基本测试."""
        from mathflow.simulation.queue_model import QueueModel
        qm = QueueModel(arrival_rate=2, service_rate=3)
        result = qm.solve()
        # solve() 返回 dict
        assert isinstance(result, dict)
        rho = 2 / 3
        assert abs(result["rho"] - rho) < 1e-6

    def test_mm1_unstable_raises(self):
        """lambda >= mu 应抛出异常."""
        from mathflow.simulation.queue_model import QueueModel
        with pytest.raises(ValueError):
            qm = QueueModel(arrival_rate=3, service_rate=2)
            qm.solve()

    def test_mmc_basic(self):
        """M/M/c 基本测试."""
        from mathflow.simulation.queue_model import QueueModel
        qm = QueueModel(arrival_rate=4, service_rate=3, n_servers=2)
        result = qm.solve()
        assert isinstance(result, dict)
        assert result["rho"] > 0
        assert result["W"] > 0

    def test_mm1k_basic(self):
        """M/M/1/K 基本测试."""
        from mathflow.simulation.queue_model import QueueModel
        qm = QueueModel(arrival_rate=2, service_rate=3, capacity=5)
        result = qm.solve()
        assert isinstance(result, dict)
        assert result["P_loss"] > 0
        assert result["lambda_eff"] < 2

    def test_summary_before_solve_raises(self):
        """solve 前调用 summary 应抛出异常."""
        from mathflow.simulation.queue_model import QueueModel
        qm = QueueModel(arrival_rate=2, service_rate=3)
        with pytest.raises(RuntimeError):
            qm.summary()


class TestEdgeCases:
    def test_queue_high_traffic(self):
        """高负载排队系统."""
        from mathflow.simulation.queue_model import QueueModel
        q = QueueModel(arrival_rate=0.99, service_rate=1.0, n_servers=1)
        result = q.solve()
        assert result is not None
        assert result['rho'] > 0.9

    def test_monte_carlo_small_n(self):
        """小样本蒙特卡洛."""
        from mathflow.simulation.monte_carlo import MonteCarlo
        mc = MonteCarlo(seed=42)
        pi_est, error = mc.estimate_pi(n_samples=100)
        assert 2.0 < pi_est < 4.0  # 粗略范围


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
