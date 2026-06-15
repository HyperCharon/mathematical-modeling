"""插值与拟合模块测试."""

import numpy as np
import pytest
from mathflow.interpolate import CurveFitter, Interpolator


class TestCurveFitter:
    def test_linear(self):
        x = np.array([1, 2, 3, 4, 5])
        y = 2 * x + 3
        cf = CurveFitter(x, y)
        cf.fit("linear")
        assert cf.result.r2 > 0.999

    def test_exponential(self):
        x = np.array([1, 2, 3, 4, 5])
        y = 2 * np.exp(0.5 * x)
        cf = CurveFitter(x, y)
        cf.fit("exponential")
        assert cf.result.r2 > 0.99

    def test_auto_fit(self):
        x = np.array([1, 2, 3, 4, 5, 6])
        y = x**2
        cf = CurveFitter(x, y)
        cf.auto_fit()
        assert cf.result.r2 > 0.99

    def test_predict(self):
        x = np.array([1, 2, 3, 4, 5])
        y = 2 * x + 3
        cf = CurveFitter(x, y)
        cf.fit("linear")
        pred = cf.predict([6])
        np.testing.assert_allclose(pred, [15.0], atol=0.5)


class TestInterpolator:
    def test_lagrange(self):
        x = np.array([0, 1, 2, 3, 4])
        y = x**2
        interp = Interpolator(x, y)
        val = interp.interpolate(2.5, method="lagrange")
        np.testing.assert_allclose(val, 6.25, atol=0.01)

    def test_cubic_spline(self):
        x = np.array([0, 1, 2, 3, 4, 5])
        y = np.sin(x)
        interp = Interpolator(x, y)
        val = interp.interpolate(1.5, method="cubic_spline")
        np.testing.assert_allclose(val, np.sin(1.5), atol=0.05)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
