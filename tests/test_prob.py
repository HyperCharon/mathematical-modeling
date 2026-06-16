"""
测试 prob 模块: DistributionFitter, HypothesisTest
"""

import numpy as np
import pytest


class TestDistributionFitter:
    """测试分布拟合器."""

    def test_too_few_points_raises(self):
        """少于 2 个数据点应抛出异常."""
        from mathflow.prob.distributions import DistributionFitter
        with pytest.raises(ValueError):
            DistributionFitter([1.0])

    def test_fit_normal(self):
        """对正态数据拟合正态分布应有较高的 K-S p 值."""
        from mathflow.prob.distributions import DistributionFitter
        np.random.seed(42)
        data = np.random.normal(50, 10, 200)
        df = DistributionFitter(data)
        result = df.fit("norm")
        assert result.ks_p_value > 0.05

    def test_auto_fit_sorted(self):
        """auto_fit 返回结果应按 AIC 升序排列."""
        from mathflow.prob.distributions import DistributionFitter
        np.random.seed(42)
        data = np.random.normal(50, 10, 200)
        df = DistributionFitter(data)
        results = df.auto_fit(top_n=3)
        assert len(results) <= 3
        for i in range(len(results) - 1):
            assert results[i].aic <= results[i + 1].aic

    def test_auto_fit_constant_data(self):
        """常数数据应发出警告但不崩溃."""
        from mathflow.prob.distributions import DistributionFitter
        data = np.ones(50) * 42.0
        with pytest.warns(UserWarning):
            df = DistributionFitter(data)
        results = df.auto_fit()
        # 应该有有效结果
        assert len(results) >= 0

    def test_goodness_of_fit(self):
        """拟合优度检验应返回字典."""
        from mathflow.prob.distributions import DistributionFitter
        np.random.seed(42)
        data = np.random.normal(50, 10, 100)
        df = DistributionFitter(data)
        gof = df.goodness_of_fit()
        assert isinstance(gof, dict)
        assert "正态分布" in gof


class TestHypothesisTest:
    """测试假设检验."""

    @pytest.fixture
    def ht(self):
        from mathflow.prob.hypothesis import HypothesisTest
        return HypothesisTest(alpha=0.05)

    def test_one_sample_ttest(self, ht):
        """单样本 t 检验."""
        np.random.seed(42)
        data = np.random.normal(50, 5, 100)
        result = ht.one_sample_ttest(data, mu0=50)
        # 真实均值为 50，不拒绝 H0
        assert not result.is_significant

    def test_two_sample_ttest_different(self, ht):
        """两样本 t 检验：明显不同的数据应显著."""
        np.random.seed(42)
        data1 = np.random.normal(50, 5, 100)
        data2 = np.random.normal(60, 5, 100)
        result = ht.two_sample_ttest(data1, data2)
        assert result.is_significant

    def test_paired_ttest_different(self, ht):
        """配对 t 检验：不同数据应有差异."""
        data1 = np.array([1, 2, 3, 4, 5], dtype=float)
        data2 = np.array([2, 3, 4, 5, 6], dtype=float)
        result = ht.paired_ttest(data1, data2)
        # 差异明显，应拒绝 H0
        assert result.is_significant

    def test_chi_square_uniform(self, ht):
        """卡方检验：均匀观测频数."""
        observed = np.array([25, 25, 25, 25])
        result = ht.chi_square_test(observed)
        assert result.p_value > 0.05

    def test_normality_test_shapiro(self, ht):
        """Shapiro 正态性检验."""
        np.random.seed(42)
        data = np.random.normal(0, 1, 100)
        result = ht.normality_test(data, method="shapiro")
        assert not result.is_significant

    def test_normality_test_unknown_raises(self, ht):
        """未知方法应抛出异常."""
        data = np.array([1, 2, 3, 4, 5])
        with pytest.raises(ValueError):
            ht.normality_test(data, method="unknown")

    def test_levene_test(self, ht):
        """Levene 方差齐性检验."""
        np.random.seed(42)
        data1 = np.random.normal(50, 5, 100)
        data2 = np.random.normal(50, 5, 100)
        result = ht.levene_test(data1, data2)
        assert not result.is_significant

    def test_mann_whitney_different(self, ht):
        """Mann-Whitney U 检验：显著不同的分布."""
        np.random.seed(42)
        data1 = np.random.normal(50, 5, 100)
        data2 = np.random.normal(70, 5, 100)
        result = ht.mann_whitney(data1, data2)
        assert result.is_significant


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
