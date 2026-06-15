"""预测类模型测试."""

import numpy as np
import pytest
from mathflow.predict import GreyPrediction, ExponentialSmoothing, RegressionPredict


class TestGreyPrediction:
    def test_basic(self):
        data = [124, 130, 138, 146, 155, 165]
        gp = GreyPrediction(data)
        gp.fit()
        forecasts = gp.predict(steps=3)
        assert len(forecasts) == 3
        assert all(f > 0 for f in forecasts)
        # 预测值应该有增长趋势
        assert forecasts[-1] > forecasts[0]

    def test_consistency(self):
        """拟合值应该接近原始值."""
        data = [100, 110, 121, 133, 146]
        gp = GreyPrediction(data)
        gp.fit()
        fitted = gp.result.fitted_values
        # 平均相对误差应小于 5%
        rel_errors = np.abs(np.array(data) - fitted) / np.array(data)
        assert rel_errors.mean() < 0.05


class TestExponentialSmoothing:
    def test_simple(self):
        data = [10, 12, 13, 15, 16, 18, 20]
        es = ExponentialSmoothing(data, method="simple", alpha=0.3)
        es.fit()
        forecast = es.predict(steps=3)
        assert len(forecast) == 3

    def test_holt(self):
        data = [10, 12, 14, 16, 18, 20, 22, 24]
        es = ExponentialSmoothing(data, method="holt", alpha=0.3, beta=0.1)
        es.fit()
        forecast = es.predict(steps=3)
        assert len(forecast) == 3
        # 趋势上升，预测值应该递增
        assert forecast[-1] > forecast[0]


class TestRegressionPredict:
    def test_linear(self):
        X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
        y = np.array([2.1, 3.9, 6.2, 7.8, 10.1])
        model = RegressionPredict(X, y)
        model.fit(method="linear")
        assert model.r2 > 0.95

    def test_polynomial(self):
        X = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)
        y = np.array([1, 4, 9, 16, 25, 36])
        model = RegressionPredict(X, y)
        model.fit(method="polynomial", degree=2)
        assert model.r2 > 0.99


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
