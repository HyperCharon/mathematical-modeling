"""
PROMETHEE 偏好排序方法 (Preference Ranking Organization Method for Enrichment Evaluations)

一种多准则决策方法，基于方案间的偏好关系进行排序。

Example:
    >>> from mathflow.evaluate import PROMETHEE
    >>> import numpy as np
    >>> data = np.array([
    ...     [80, 90, 85],  # 方案A
    ...     [70, 80, 90],  # 方案B
    ...     [90, 85, 75],  # 方案C
    ... ])
    >>> weights = [0.4, 0.3, 0.3]
    >>> types = [1, 1, -1]  # 1=效益型, -1=成本型
    >>> p = PROMETHEE(data, weights, types)
    >>> result = p.fit()
    >>> print(p.summary())
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class PROMETHEEResult:
    """PROMETHEE 结果."""
    phi_plus: np.ndarray      # 正流量 (流出)
    phi_minus: np.ndarray     # 负流量 (流入)
    phi_net: np.ndarray       # 净流量
    rankings: np.ndarray      # 排名 (1=最优)
    preference_matrix: np.ndarray  # 偏好矩阵
    n_alternatives: int
    n_criteria: int


class PROMETHEE:
    """
    PROMETHEE 偏好排序方法.

    Parameters
    ----------
    data : array-like, shape (n_alternatives, n_criteria)
        决策矩阵
    weights : array-like, shape (n_criteria,)
        各准则权重
    types : list of int
        准则类型: 1=效益型(越大越好), -1=成本型(越小越好)
    preference_function : str
        偏好函数类型: "usual", "u_shape", "v_shape", "level", "linear", "gaussian"
    p_threshold : float, optional
        偏好阈值 (用于 level 和 linear 函数)
    q_threshold : float, optional
        无差异阈值 (用于 u_shape, v_shape, level, linear 函数)
    """

    def __init__(self, data, weights, types=None,
                 preference_function="linear", p_threshold=None, q_threshold=None):
        self.data = np.asarray(data, dtype=float)
        if self.data.ndim != 2:
            raise ValueError("data 必须是二维矩阵")

        self.n_alternatives, self.n_criteria = self.data.shape

        self.weights = np.asarray(weights, dtype=float)
        if len(self.weights) != self.n_criteria:
            raise ValueError(f"weights 长度 ({len(self.weights)}) 必须等于准则数 ({self.n_criteria})")
        self.weights = self.weights / self.weights.sum()  # 归一化

        if types is None:
            types = [1] * self.n_criteria
        self.types = np.asarray(types)

        self.preference_function = preference_function

        # 设置默认阈值
        if p_threshold is None:
            self.p_threshold = np.std(self.data, axis=0)
        else:
            self.p_threshold = np.full(self.n_criteria, p_threshold)

        if q_threshold is None:
            self.q_threshold = np.zeros(self.n_criteria)
        else:
            self.q_threshold = np.full(self.n_criteria, q_threshold)

        self._result = None

    def _compute_preference(self, d, j):
        """
        计算偏好函数值.

        Parameters
        ----------
        d : float
            差值 (f(a) - f(b))
        j : int
            准则索引
        """
        pf = self.preference_function
        p = self.p_threshold[j]
        q = self.q_threshold[j]

        if pf == "usual":
            # 通常函数: d > 0 则偏好为1
            return 1.0 if d > 0 else 0.0

        elif pf == "u_shape":
            # U型函数: d > q 则偏好为1
            return 1.0 if d > q else 0.0

        elif pf == "v_shape":
            # V型函数: 线性
            if d <= 0:
                return 0.0
            elif d <= p:
                return d / p
            else:
                return 1.0

        elif pf == "level":
            # 水平函数
            if d <= q:
                return 0.0
            elif d <= p:
                return 0.5
            else:
                return 1.0

        elif pf == "linear":
            # 线性函数
            if d <= q:
                return 0.0
            elif d <= p:
                return (d - q) / (p - q) if p > q else 0.0
            else:
                return 1.0

        elif pf == "gaussian":
            # 高斯函数
            if d <= 0:
                return 0.0
            else:
                sigma = p / 3
                return 1 - np.exp(-d**2 / (2 * sigma**2))

        else:
            raise ValueError(f"未知偏好函数: {pf}")

    def fit(self) -> PROMETHEEResult:
        """执行 PROMETHEE 分析."""
        n = self.n_alternatives
        m = self.n_criteria

        # 计算偏好矩阵
        preference = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                pref_sum = 0.0
                for k in range(m):
                    # 计算差值
                    if self.types[k] == 1:  # 效益型
                        d = self.data[i, k] - self.data[j, k]
                    else:  # 成本型
                        d = self.data[j, k] - self.data[i, k]

                    # 计算偏好函数值
                    pref = self._compute_preference(d, k)
                    pref_sum += self.weights[k] * pref

                preference[i, j] = pref_sum

        # 计算流量
        phi_plus = np.sum(preference, axis=1) / (n - 1)  # 正流量 (流出)
        phi_minus = np.sum(preference, axis=0) / (n - 1)  # 负流量 (流入)
        phi_net = phi_plus - phi_minus  # 净流量

        # 排名 (净流量越大排名越靠前)
        rankings = np.argsort(np.argsort(-phi_net)) + 1

        self._result = PROMETHEEResult(
            phi_plus=phi_plus,
            phi_minus=phi_minus,
            phi_net=phi_net,
            rankings=rankings,
            preference_matrix=preference,
            n_alternatives=n,
            n_criteria=m,
        )
        return self._result

    @property
    def rankings(self):
        self._ensure_fitted()
        return self._result.rankings

    @property
    def phi_net(self):
        self._ensure_fitted()
        return self._result.phi_net

    def _ensure_fitted(self):
        if self._result is None:
            raise RuntimeError("请先调用 fit()")

    def plot(self, figsize=(10, 5)):
        """绘制结果."""
        import matplotlib.pyplot as plt

        self._ensure_fitted()
        r = self._result

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # 流量图
        ax = axes[0]
        x = range(r.n_alternatives)
        width = 0.35
        ax.bar([i - width/2 for i in x], r.phi_plus, width, label="Phi+ (Outflow)", alpha=0.7)
        ax.bar([i + width/2 for i in x], r.phi_minus, width, label="Phi- (Inflow)", alpha=0.7)
        ax.set_xlabel("Alternative")
        ax.set_ylabel("Flow")
        ax.set_title("PROMETHEE Flows")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 净流量图
        ax = axes[1]
        colors = ["green" if phi > 0 else "red" for phi in r.phi_net]
        ax.bar(x, r.phi_net, color=colors, alpha=0.7)
        ax.axhline(y=0, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("Alternative")
        ax.set_ylabel("Net Flow")
        ax.set_title("PROMETHEE Net Flow")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def summary(self) -> str:
        """打印摘要."""
        self._ensure_fitted()
        r = self._result

        lines = [
            "=" * 70,
            "  PROMETHEE 偏好排序结果",
            "=" * 70,
            f"  准则数: {r.n_criteria}",
            f"  方案数: {r.n_alternatives}",
            f"  偏好函数: {self.preference_function}",
            "-" * 70,
            f"  {'方案':>8s}  {'Phi+':>10s}  {'Phi-':>10s}  {'Phi Net':>10s}  {'排名':>6s}",
            "-" * 70,
        ]

        for i in range(r.n_alternatives):
            lines.append(
                f"  Alt {i+1:>4d}  {r.phi_plus[i]:>10.4f}  {r.phi_minus[i]:>10.4f}  "
                f"{r.phi_net[i]:>10.4f}  {r.rankings[i]:>6d}"
            )

        lines.append("=" * 70)

        # 最优方案
        best_idx = np.argmin(r.rankings)
        lines.append(f"  最优方案: Alt {best_idx + 1} (Phi Net = {r.phi_net[best_idx]:.4f})")

        return "\n".join(lines)
