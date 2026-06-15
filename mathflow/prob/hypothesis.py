"""
假设检验

支持 t检验、卡方检验、方差齐性检验、正态性检验。

Example:
    >>> from mathflow.prob import HypothesisTest
    >>> import numpy as np
    >>> data1 = np.random.normal(50, 10, 30)
    >>> data2 = np.random.normal(55, 10, 30)
    >>> ht = HypothesisTest()
    >>> result = ht.two_sample_ttest(data1, data2)
"""

import numpy as np
from scipy import stats
from dataclasses import dataclass


@dataclass
class TestResult:
    """检验结果."""
    test_name: str
    statistic: float
    p_value: float
    is_significant: bool
    conclusion: str
    alpha: float


class HypothesisTest:
    """假设检验工具箱."""

    def __init__(self, alpha=0.05):
        self.alpha = alpha

    def one_sample_ttest(self, data, mu0: float) -> TestResult:
        """单样本 t 检验: H0: μ = μ0."""
        data = np.asarray(data, dtype=float)
        t_stat, p_value = stats.ttest_1samp(data, mu0)
        significant = p_value < self.alpha
        conclusion = f"拒绝H0，均值显著{'不' if not significant else ''}等于{mu0}"

        return TestResult(
            test_name="单样本t检验", statistic=t_stat, p_value=p_value,
            is_significant=significant, conclusion=conclusion, alpha=self.alpha,
        )

    def two_sample_ttest(self, data1, data2, equal_var=True) -> TestResult:
        """两样本 t 检验: H0: μ1 = μ2."""
        data1 = np.asarray(data1, dtype=float)
        data2 = np.asarray(data2, dtype=float)
        t_stat, p_value = stats.ttest_ind(data1, data2, equal_var=equal_var)
        significant = p_value < self.alpha
        conclusion = f"拒绝H0，两组均值{'有' if significant else '无'}显著差异"

        return TestResult(
            test_name="两样本t检验", statistic=t_stat, p_value=p_value,
            is_significant=significant, conclusion=conclusion, alpha=self.alpha,
        )

    def paired_ttest(self, data1, data2) -> TestResult:
        """配对 t 检验."""
        data1 = np.asarray(data1, dtype=float)
        data2 = np.asarray(data2, dtype=float)
        t_stat, p_value = stats.ttest_rel(data1, data2)
        significant = p_value < self.alpha
        conclusion = f"拒绝H0，配对数据均值{'有' if significant else '无'}显著差异"

        return TestResult(
            test_name="配对t检验", statistic=t_stat, p_value=p_value,
            is_significant=significant, conclusion=conclusion, alpha=self.alpha,
        )

    def chi_square_test(self, observed, expected=None) -> TestResult:
        """卡方检验."""
        observed = np.asarray(observed, dtype=float)
        if expected is not None:
            expected = np.asarray(expected, dtype=float)
        chi2, p_value = stats.chisquare(observed, expected)
        significant = p_value < self.alpha
        conclusion = f"拒绝H0，观测频数与期望频数{'有' if significant else '无'}显著差异"

        return TestResult(
            test_name="卡方检验", statistic=chi2, p_value=p_value,
            is_significant=significant, conclusion=conclusion, alpha=self.alpha,
        )

    def normality_test(self, data, method="shapiro") -> TestResult:
        """正态性检验."""
        data = np.asarray(data, dtype=float)
        if method == "shapiro":
            stat, p_value = stats.shapiro(data)
            test_name = "Shapiro-Wilk正态性检验"
        elif method == "kstest":
            stat, p_value = stats.kstest(data, "norm", args=(data.mean(), data.std()))
            test_name = "K-S正态性检验"
        elif method == "anderson":
            result = stats.anderson(data, "norm")
            stat = result.statistic
            p_value = 0.05 if stat > result.critical_values[2] else 0.5  # 近似
            test_name = "Anderson-Darling正态性检验"
        else:
            raise ValueError(f"未知方法: {method}")

        significant = p_value < self.alpha
        conclusion = f"{'拒绝' if significant else '不拒绝'}H0，数据{'不' if significant else ''}服从正态分布"

        return TestResult(
            test_name=test_name, statistic=stat, p_value=p_value,
            is_significant=significant, conclusion=conclusion, alpha=self.alpha,
        )

    def levene_test(self, *groups) -> TestResult:
        """Levene 方差齐性检验."""
        groups = [np.asarray(g, dtype=float) for g in groups]
        stat, p_value = stats.levene(*groups)
        significant = p_value < self.alpha
        conclusion = f"拒绝H0，各组方差{'不' if not significant else ''}齐性"

        return TestResult(
            test_name="Levene方差齐性检验", statistic=stat, p_value=p_value,
            is_significant=significant, conclusion=conclusion, alpha=self.alpha,
        )

    def mann_whitney(self, data1, data2) -> TestResult:
        """Mann-Whitney U 检验 (非参数)."""
        data1 = np.asarray(data1, dtype=float)
        data2 = np.asarray(data2, dtype=float)
        stat, p_value = stats.mannwhitneyu(data1, data2, alternative="two-sided")
        significant = p_value < self.alpha
        conclusion = f"拒绝H0，两组数据分布{'有' if significant else '无'}显著差异"

        return TestResult(
            test_name="Mann-Whitney U检验", statistic=stat, p_value=p_value,
            is_significant=significant, conclusion=conclusion, alpha=self.alpha,
        )

    def summary(self, result: TestResult) -> str:
        """输出检验结果."""
        lines = [
            "=" * 50,
            f"  {result.test_name}结果",
            "=" * 50,
            f"  检验统计量: {result.statistic:.4f}",
            f"  p值: {result.p_value:.4f}",
            f"  显著性水平: {result.alpha}",
            f"  结论: {result.conclusion}",
            "=" * 50,
        ]
        return "\n".join(lines)
