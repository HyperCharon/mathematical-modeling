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


class TestTwoWayANOVA:
    def test_two_way(self):
        """Two-way ANOVA without replication."""
        data = np.array([
            [5.2, 4.8, 5.0, 4.5],
            [6.1, 5.8, 6.0, 5.5],
            [5.8, 5.5, 5.7, 5.2],
        ])  # 3 rows (factor A) x 4 cols (factor B)
        anova = ANOVA()
        result = anova.two_way(data)
        assert result is not None
        assert anova.two_way_result is not None

    def test_two_way_replicated(self):
        """Two-way ANOVA with replication."""
        # 2 factors (A: 3 levels, B: 2 levels), 2 replicates
        # Shape: (3, 2, 2)
        data = np.array([
            [[5.2, 4.8], [5.0, 4.5]],
            [[6.1, 5.8], [6.0, 5.5]],
            [[5.8, 5.5], [5.7, 5.2]],
        ])
        anova = ANOVA()
        result = anova.two_way_replicated(data)
        assert result is not None
        assert anova.two_way_result is not None


class TestEdgeCases:
    def test_correlation_2vars(self):
        """2变量相关性分析."""
        from mathflow.stats import CorrelationAnalysis
        data = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        ca = CorrelationAnalysis(data)
        ca.fit()
        assert ca.corr_matrix.shape == (2, 2)

    def test_anova_identical_groups(self):
        """完全相同组的 ANOVA."""
        anova = ANOVA()
        # 所有组完全相同
        anova.one_way(np.array([5, 5, 5]), np.array([5, 5, 5]))
        assert anova.result.p_value > 0.05  # 不应显著


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
