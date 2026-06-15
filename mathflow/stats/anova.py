"""
方差分析 (Analysis of Variance, ANOVA)

支持单因素方差分析和双因素方差分析。

Example:
    >>> from mathflow.stats import ANOVA
    >>> # 3种教学方法的考试成绩
    >>> group1 = [85, 90, 88, 92, 87]
    >>> group2 = [78, 82, 80, 76, 79]
    >>> group3 = [92, 95, 89, 91, 93]
    >>> anova = ANOVA()
    >>> anova.one_way(group1, group2, group3, group_names=["方法A","方法B","方法C"])
    >>> print(anova.summary())
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


class ANOVA:
    """方差分析."""

    def __init__(self, alpha=0.05):
        self.alpha = alpha
        self._result = None

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
        )
        return self

    @property
    def result(self):
        if self._result is None:
            raise RuntimeError("请先调用 one_way()")
        return self._result

    def summary(self):
        """输出 ANOVA 表."""
        r = self._result
        lines = [
            "=" * 60,
            "  单因素方差分析 (One-Way ANOVA)",
            "=" * 60,
            f"  {'来源':>10s}  {'平方和':>12s}  {'自由度':>8s}  {'均方':>12s}  {'F值':>10s}  {'p值':>10s}",
            "-" * 60,
            f"  {'组间':>10s}  {r.ss_between:>12.4f}  {r.df_between:>8d}  "
            f"{r.ss_between/r.df_between:>12.4f}  {r.f_statistic:>10.4f}  {r.p_value:>10.4f}",
            f"  {'组内':>10s}  {r.ss_within:>12.4f}  {r.df_within:>8d}  "
            f"{r.ss_within/r.df_within:>12.4f}",
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
