"""
方差分析 (Analysis of Variance, ANOVA)

支持单因素方差分析和双因素方差分析。

Example:
    >>> from mathflow.stats import ANOVA
    >>> # 单因素方差分析
    >>> group1 = [85, 90, 88, 92, 87]
    >>> group2 = [78, 82, 80, 76, 79]
    >>> group3 = [92, 95, 89, 91, 93]
    >>> anova = ANOVA()
    >>> anova.one_way(group1, group2, group3, group_names=["方法A","方法B","方法C"])
    >>> print(anova.summary())
    >>> # 双因素方差分析
    >>> data = np.array([[85, 90, 88], [78, 82, 80], [92, 95, 89]])
    >>> anova.two_way(data, row_names=["A","B","C"], col_names=["X","Y","Z"])
"""

import numpy as np
from scipy import stats as scipy_stats
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ANOVAResult:
    """方差分析结果."""
    ss_between: float     # 组间平方和
    ss_within: float      # 组内平方和
    ss_total: float       # 总平方和
    df_between: int       # 组间自由度
    df_within: int        # 组内自由度
    f_statistic: float    # F 统计量
    p_value: float        # p 值
    is_significant: bool  # 是否显著 (alpha=0.05)
    group_means: np.ndarray
    group_stds: np.ndarray
    group_names: List[str]
    anova_type: str = "one_way"


@dataclass
class TwoWayANOVAResult:
    """双因素方差分析结果."""
    ss_factor_a: float    # 因素A平方和
    ss_factor_b: float    # 因素B平方和
    ss_interaction: float # 交互效应平方和
    ss_error: float       # 误差平方和
    ss_total: float       # 总平方和
    df_factor_a: int
    df_factor_b: int
    df_interaction: int
    df_error: int
    f_a: float            # 因素A的F值
    f_b: float            # 因素B的F值
    f_interaction: float  # 交互效应F值
    p_a: float
    p_b: float
    p_interaction: float
    is_significant_a: bool
    is_significant_b: bool
    is_significant_interaction: bool
    row_names: List[str]
    col_names: List[str]


class ANOVA:
    """方差分析."""

    def __init__(self, alpha=0.05):
        self.alpha = alpha
        self._result = None
        self._two_way_result = None

    def __repr__(self) -> str:
        if self._result is not None:
            return f"ANOVA(alpha={self.alpha}, fitted=True)"
        return f"ANOVA(alpha={self.alpha})"

    def one_way(self, *groups, group_names=None):
        """
        单因素方差分析.

        Parameters
        ----------
        *groups : array-like
            各组数据
        group_names : list of str, optional
            组名
        """
        groups = [np.asarray(g, dtype=float) for g in groups]
        k = len(groups)
        n_total = sum(len(g) for g in groups)

        if group_names is None:
            group_names = [f"组{i+1}" for i in range(k)]

        # 各组统计量
        means = np.array([g.mean() for g in groups])
        stds = np.array([g.std(ddof=1) for g in groups])
        sizes = np.array([len(g) for g in groups])

        # 总均值
        grand_mean = np.concatenate(groups).mean()

        # 平方和
        ss_between = sum(sizes[i] * (means[i] - grand_mean)**2 for i in range(k))
        ss_within = sum(np.sum((groups[i] - means[i])**2) for i in range(k))
        ss_total = ss_between + ss_within

        # 自由度
        df_between = k - 1
        df_within = n_total - k

        # 均方
        ms_between = ss_between / df_between if df_between > 0 else 0
        ms_within = ss_within / df_within if df_within > 0 else 0

        # F 统计量
        f_stat = ms_between / ms_within if ms_within > 0 else 0
        p_value = 1 - scipy_stats.f.cdf(f_stat, df_between, df_within)

        self._result = ANOVAResult(
            ss_between=ss_between, ss_within=ss_within, ss_total=ss_total,
            df_between=df_between, df_within=df_within,
            f_statistic=f_stat, p_value=p_value,
            is_significant=p_value < self.alpha,
            group_means=means, group_stds=stds, group_names=group_names,
            anova_type="one_way",
        )
        return self

    def two_way(self, data, row_names=None, col_names=None):
        """
        双因素方差分析 (无重复).

        Parameters
        ----------
        data : array-like, shape (n_rows, n_cols)
            数据矩阵，行表示因素A的水平，列表示因素B的水平
        row_names : list of str, optional
            因素A的水平名称
        col_names : list of str, optional
            因素B的水平名称
        """
        data = np.asarray(data, dtype=float)
        if data.ndim != 2:
            raise ValueError("data 必须是二维矩阵")

        a, b = data.shape  # 因素A水平数, 因素B水平数
        n = a * b  # 总观测数

        if row_names is None:
            row_names = [f"A{i+1}" for i in range(a)]
        if col_names is None:
            col_names = [f"B{j+1}" for j in range(b)]

        # 总均值
        grand_mean = data.mean()

        # 因素A的均值 (行均值)
        row_means = data.mean(axis=1)
        # 因素B的均值 (列均值)
        col_means = data.mean(axis=0)

        # 平方和
        ss_total = np.sum((data - grand_mean)**2)
        ss_factor_a = b * np.sum((row_means - grand_mean)**2)
        ss_factor_b = a * np.sum((col_means - grand_mean)**2)
        ss_error = ss_total - ss_factor_a - ss_factor_b

        # 自由度
        df_a = a - 1
        df_b = b - 1
        df_error = (a - 1) * (b - 1)
        df_total = n - 1

        # 均方
        ms_a = ss_factor_a / df_a if df_a > 0 else 0
        ms_b = ss_factor_b / df_b if df_b > 0 else 0
        ms_error = ss_error / df_error if df_error > 0 else 0

        # F 统计量
        f_a = ms_a / ms_error if ms_error > 0 else 0
        f_b = ms_b / ms_error if ms_error > 0 else 0

        # p 值
        p_a = 1 - scipy_stats.f.cdf(f_a, df_a, df_error)
        p_b = 1 - scipy_stats.f.cdf(f_b, df_b, df_error)

        self._two_way_result = TwoWayANOVAResult(
            ss_factor_a=ss_factor_a,
            ss_factor_b=ss_factor_b,
            ss_interaction=0,  # 无重复时无交互效应
            ss_error=ss_error,
            ss_total=ss_total,
            df_factor_a=df_a,
            df_factor_b=df_b,
            df_interaction=0,
            df_error=df_error,
            f_a=f_a,
            f_b=f_b,
            f_interaction=0,
            p_a=p_a,
            p_b=p_b,
            p_interaction=1,
            is_significant_a=p_a < self.alpha,
            is_significant_b=p_b < self.alpha,
            is_significant_interaction=False,
            row_names=row_names,
            col_names=col_names,
        )
        return self

    def two_way_replicated(self, data, row_names=None, col_names=None):
        """
        双因素方差分析 (有重复).

        Parameters
        ----------
        data : array-like, shape (a, b, r)
            数据矩阵，a=因素A水平数，b=因素B水平数，r=重复次数
        row_names : list of str, optional
            因素A的水平名称
        col_names : list of str, optional
            因素B的水平名称
        """
        data = np.asarray(data, dtype=float)
        if data.ndim != 3:
            raise ValueError("data 必须是三维矩阵 (a, b, r)")

        a, b, r = data.shape

        if row_names is None:
            row_names = [f"A{i+1}" for i in range(a)]
        if col_names is None:
            col_names = [f"B{j+1}" for j in range(b)]

        n = a * b * r
        grand_mean = data.mean()

        # 因素A均值
        row_means = data.mean(axis=(1, 2))
        # 因素B均值
        col_means = data.mean(axis=(0, 2))
        # 交互效应均值
        cell_means = data.mean(axis=2)

        # 平方和
        ss_total = np.sum((data - grand_mean)**2)
        ss_a = b * r * np.sum((row_means - grand_mean)**2)
        ss_b = a * r * np.sum((col_means - grand_mean)**2)
        ss_subtotal = r * np.sum((cell_means - grand_mean)**2)
        ss_interaction = ss_subtotal - ss_a - ss_b
        ss_error = ss_total - ss_subtotal

        # 自由度
        df_a = a - 1
        df_b = b - 1
        df_interaction = (a - 1) * (b - 1)
        df_error = a * b * (r - 1)
        df_total = n - 1

        # 均方
        ms_a = ss_a / df_a if df_a > 0 else 0
        ms_b = ss_b / df_b if df_b > 0 else 0
        ms_interaction = ss_interaction / df_interaction if df_interaction > 0 else 0
        ms_error = ss_error / df_error if df_error > 0 else 0

        # F 统计量
        f_a = ms_a / ms_error if ms_error > 0 else 0
        f_b = ms_b / ms_error if ms_error > 0 else 0
        f_interaction = ms_interaction / ms_error if ms_error > 0 else 0

        # p 值
        p_a = 1 - scipy_stats.f.cdf(f_a, df_a, df_error)
        p_b = 1 - scipy_stats.f.cdf(f_b, df_b, df_error)
        p_interaction = 1 - scipy_stats.f.cdf(f_interaction, df_interaction, df_error)

        self._two_way_result = TwoWayANOVAResult(
            ss_factor_a=ss_a,
            ss_factor_b=ss_b,
            ss_interaction=ss_interaction,
            ss_error=ss_error,
            ss_total=ss_total,
            df_factor_a=df_a,
            df_factor_b=df_b,
            df_interaction=df_interaction,
            df_error=df_error,
            f_a=f_a,
            f_b=f_b,
            f_interaction=f_interaction,
            p_a=p_a,
            p_b=p_b,
            p_interaction=p_interaction,
            is_significant_a=p_a < self.alpha,
            is_significant_b=p_b < self.alpha,
            is_significant_interaction=p_interaction < self.alpha,
            row_names=row_names,
            col_names=col_names,
        )
        return self

    @property
    def result(self):
        if self._result is None:
            raise RuntimeError("请先调用 one_way()")
        return self._result

    @property
    def two_way_result(self):
        if self._two_way_result is None:
            raise RuntimeError("请先调用 two_way() 或 two_way_replicated()")
        return self._two_way_result

    def summary(self):
        """输出 ANOVA 表."""
        if self._two_way_result is not None:
            return self._summary_two_way()
        if self._result is None:
            raise RuntimeError("请先调用 one_way() 或 two_way()")

        r = self._result
        if r.anova_type == "one_way":
            return self._summary_one_way()
        return self._summary_one_way()

    def _summary_one_way(self):
        """单因素方差分析摘要."""
        r = self._result
        ms_between = r.ss_between / r.df_between if r.df_between > 0 else 0
        ms_within = r.ss_within / r.df_within if r.df_within > 0 else 0

        lines = [
            "=" * 60,
            "  单因素方差分析 (One-Way ANOVA)",
            "=" * 60,
            f"  {'来源':>10s}  {'平方和':>12s}  {'自由度':>8s}  {'均方':>12s}  {'F值':>10s}  {'p值':>10s}",
            "-" * 60,
            f"  {'组间':>10s}  {r.ss_between:>12.4f}  {r.df_between:>8d}  "
            f"{ms_between:>12.4f}  {r.f_statistic:>10.4f}  {r.p_value:>10.4f}",
            f"  {'组内':>10s}  {r.ss_within:>12.4f}  {r.df_within:>8d}  "
            f"{ms_within:>12.4f}",
            f"  {'总计':>10s}  {r.ss_total:>12.4f}  {r.df_between+r.df_within:>8d}",
            "-" * 60,
            f"  F = {r.f_statistic:.4f}, p = {r.p_value:.4f}",
            f"  结论: {'拒绝H0，各组均值有显著差异 ✅' if r.is_significant else '不拒绝H0，各组均值无显著差异 ❌'} (α={self.alpha})",
            "-" * 60,
            "  各组描述统计:",
        ]
        for i, name in enumerate(r.group_names):
            lines.append(f"    {name}: 均值={r.group_means[i]:.4f}, 标准差={r.group_stds[i]:.4f}")
        lines.append("=" * 60)
        return "\n".join(lines)

    def _summary_two_way(self):
        """双因素方差分析摘要."""
        r = self._two_way_result
        ms_a = r.ss_factor_a / r.df_factor_a if r.df_factor_a > 0 else 0
        ms_b = r.ss_factor_b / r.df_factor_b if r.df_factor_b > 0 else 0
        ms_error = r.ss_error / r.df_error if r.df_error > 0 else 0

        lines = [
            "=" * 70,
            "  双因素方差分析 (Two-Way ANOVA)",
            "=" * 70,
            f"  {'来源':>12s}  {'平方和':>12s}  {'自由度':>8s}  {'均方':>12s}  {'F值':>10s}  {'p值':>10s}",
            "-" * 70,
            f"  {'因素A':>12s}  {r.ss_factor_a:>12.4f}  {r.df_factor_a:>8d}  "
            f"{ms_a:>12.4f}  {r.f_a:>10.4f}  {r.p_a:>10.4f}",
            f"  {'因素B':>12s}  {r.ss_factor_b:>12.4f}  {r.df_factor_b:>8d}  "
            f"{ms_b:>12.4f}  {r.f_b:>10.4f}  {r.p_b:>10.4f}",
        ]

        if r.df_interaction > 0:
            ms_interaction = r.ss_interaction / r.df_interaction
            lines.append(
                f"  {'交互效应':>12s}  {r.ss_interaction:>12.4f}  {r.df_interaction:>8d}  "
                f"{ms_interaction:>12.4f}  {r.f_interaction:>10.4f}  {r.p_interaction:>10.4f}"
            )

        lines.extend([
            f"  {'误差':>12s}  {r.ss_error:>12.4f}  {r.df_error:>8d}  {ms_error:>12.4f}",
            f"  {'总计':>12s}  {r.ss_total:>12.4f}  {r.df_factor_a+r.df_factor_b+r.df_interaction+r.df_error:>8d}",
            "-" * 70,
            "  结论:",
            f"    因素A: {'显著 ✅' if r.is_significant_a else '不显著 ❌'} (p={r.p_a:.4f})",
            f"    因素B: {'显著 ✅' if r.is_significant_b else '不显著 ❌'} (p={r.p_b:.4f})",
        ])

        if r.df_interaction > 0:
            lines.append(
                f"    交互效应: {'显著 ✅' if r.is_significant_interaction else '不显著 ❌'} (p={r.p_interaction:.4f})"
            )

        lines.append("=" * 70)
        return "\n".join(lines)
