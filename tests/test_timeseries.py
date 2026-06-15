"""时间序列模块测试."""

import numpy as np
import pytest
from mathflow.timeseries import TimeSeriesDecompose, StationarityTest


class TestTimeSeriesDecompose:
    def test_additive(self):
        """加法分解应能还原."""
        np.random.seed(42)
        t = np.arange(48)
        trend = 0.5 * t
        seasonal = 10 * np.sin(2 * np.pi * t / 12)
        data = 50 + trend + seasonal + np.random.randn(48) * 0.5

        ts = TimeSeriesDecompose(data, period=12, model="additive")
        result = ts.decompose()

        assert len(result.trend) == 48
        assert len(result.seasonal) == 48
        assert len(result.residual) == 48

        # 季节性应有周期性
        assert abs(result.seasonal[0] - result.seasonal[12]) < 1e-10

    def test_multiplicative(self):
        """乘法分解."""
        np.random.seed(42)
        t = np.arange(48)
        data = (1 + 0.01 * t) * (1 + 0.2 * np.sin(2 * np.pi * t / 12))
        data = np.abs(data) * 100 + 1  # 避免零值

        ts = TimeSeriesDecompose(data, period=12, model="multiplicative")
        result = ts.decompose()

        assert len(result.trend) == 48


class TestStationarityTest:
    def test_stationary_series(self):
        """白噪声应是平稳的."""
        np.random.seed(42)
        data = np.random.randn(200)
        st = StationarityTest(data)
        result = st.test_all()
        assert result.adf_is_stationary

    def test_random_walk(self):
        """随机游走应是非平稳的."""
        np.random.seed(42)
        data = np.cumsum(np.random.randn(200)) + 50
        st = StationarityTest(data)
        result = st.test_all()
        assert not result.adf_is_stationary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
