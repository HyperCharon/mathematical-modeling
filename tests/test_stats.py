"""统计模块测试."""

import numpy as np
import pytest
from mathflow.stats import MultiRegression, ANOVA, SensitivityAnalysis


class TestMultiRegression:
    def test_perfect_linear(self):
        """完美线性关系应得到 R² ≈ 1."""
        X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
        y = 2 * X.flatten() + 3
        model = MultiRegression(X, y)
        model.fit()
        assert model.result.r2 > 0.999
        np.testing.assert_allclose(model.result.coefficients[0], 2.0, atol=0.01)
        np.testing.assert_allclose(model.result.intercept, 3.0, atol=0.01)

    def test_predict(self):
        X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
        y = 2 * X.flatten() + 3
        model = MultiRegression(X, y)
        model.fit()
        pred = model.predict([[6]])
        np.testing.assert_allclose(pred, [15.0], atol=0.1)


class TestANOVA:
    def test_significant_difference(self):
        """明显不同的组应得到显著结果."""
        g1 = [85, 90, 88, 92, 87]
        g2 = [60, 65, 62, 58, 63]
        g3 = [75, 78, 76, 74, 77]
        anova = ANOVA()
        anova.one_way(g1, g2, g3)
        assert anova.result.is_significant  # p < 0.05

    def test_no_difference(self):
        """相似的组应得到不显著结果."""
        np.random.seed(42)
        g1 = np.random.normal(50, 1, 30)
        g2 = np.random.normal(50, 1, 30)
        anova = ANOVA()
        anova.one_way(g1, g2)
        assert not anova.result.is_significant


class TestSensitivityAnalysis:
    def test_oat(self):
        def model(x):
            return x[0]**2 + x[1]

        sa = SensitivityAnalysis(model, n_vars=2, var_names=["x1", "x2"])
        result = sa.one_at_a_time(base_values=[5, 10], perturbation=0.2)
        # x1 的灵敏度应大于 x2 (因为 x1 是平方项)
        assert result.sensitivities[0] > result.sensitivities[1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
