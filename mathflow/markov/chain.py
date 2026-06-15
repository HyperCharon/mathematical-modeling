"""
马尔可夫链

支持状态转移矩阵分析、稳态分布计算、n步转移概率。

Example:
    >>> from mathflow.markov import MarkovChain
    >>> import numpy as np
    >>> P = np.array([[0.7, 0.3], [0.4, 0.6]])
    >>> mc = MarkovChain(P, states=["晴天", "雨天"])
    >>> mc.steady_state()
    >>> mc.simulate(n_steps=100, start=0)
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class MarkovResult:
    """马尔可夫链结果."""
    transition_matrix: np.ndarray
    steady_state: Optional[np.ndarray]
    states: List[str]
    n_states: int


class MarkovChain:
    """
    马尔可夫链.

    Parameters
    ----------
    P : array-like, shape (n, n)
        状态转移矩阵
    states : list of str, optional
        状态名称
    """

    def __init__(self, P, states=None):
        self.P = np.asarray(P, dtype=float)
        n = self.P.shape[0]
        self.n_states = n
        self.states = states or [f"状态{i+1}" for i in range(n)]

        # 验证转移矩阵
        row_sums = self.P.sum(axis=1)
        if not np.allclose(row_sums, 1):
            raise ValueError("转移矩阵每行之和必须为1")

    def n_step_transition(self, n: int) -> np.ndarray:
        """计算n步转移矩阵."""
        return np.linalg.matrix_power(self.P, n)

    def steady_state(self) -> np.ndarray:
        """计算稳态分布."""
        n = self.n_states
        # 求解 πP = π, 即 (P^T - I)π = 0, Σπ = 1
        A = self.P.T - np.eye(n)
        A[-1, :] = 1
        b = np.zeros(n)
        b[-1] = 1
        try:
            pi = np.linalg.solve(A, b)
            pi = np.maximum(pi, 0)
            pi = pi / pi.sum()
        except np.linalg.LinAlgError:
            pi = np.ones(n) / n

        self._steady_state = pi
        return pi

    def simulate(self, n_steps: int = 100, start: int = 0, seed: int = 42) -> List[int]:
        """模拟马尔可夫链."""
        np.random.seed(seed)
        states = [start]
        current = start

        for _ in range(n_steps - 1):
            current = np.random.choice(self.n_states, p=self.P[current])
            states.append(current)

        self._simulation = states
        return states

    def absorption_probability(self, absorbing_states: List[int]) -> np.ndarray:
        """计算吸收概率 (适用于吸收马尔可夫链)."""
        n = self.n_states
        non_absorbing = [i for i in range(n) if i not in absorbing_states]
        k = len(non_absorbing)

        if k == 0:
            return np.zeros((n, len(absorbing_states)))

        Q = self.P[np.ix_(non_absorbing, non_absorbing)]
        R = self.P[np.ix_(non_absorbing, absorbing_states)]

        # 基本矩阵 N = (I - Q)^(-1)
        N = np.linalg.inv(np.eye(k) - Q)
        B = N @ R

        return B

    def classify_states(self) -> dict:
        """状态分类: 常返/非常返/周期性."""
        n = self.n_states
        classification = {}

        # 通过可达性分析
        reachable = np.zeros((n, n), dtype=bool)
        P_bool = self.P > 0
        reachable = P_bool.copy()
        for _ in range(n):
            reachable = reachable | (reachable @ P_bool)

        for i in range(n):
            is_recurrent = reachable[i, i]
            period = self._find_period(i)
            classification[self.states[i]] = {
                "recurrent": is_recurrent,
                "period": period,
                "accessible_from": [self.states[j] for j in range(n) if reachable[j, i]],
            }

        return classification

    def _find_period(self, state: int) -> int:
        """计算状态周期."""
        # 简化实现: 通过检查返回步数的GCD
        n = self.n_states
        return_steps = []
        P_power = np.eye(n)
        for k in range(1, n + 1):
            P_power = P_power @ self.P
            if P_power[state, state] > 1e-10:
                return_steps.append(k)

        if not return_steps:
            return 0

        from math import gcd
        from functools import reduce
        return reduce(gcd, return_steps)

    def plot_transition_matrix(self, figsize=(8, 6)):
        """绘制转移矩阵热力图."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(self.P, cmap="YlOrRd", aspect="auto")
        plt.colorbar(im, ax=ax, label="转移概率")

        for i in range(self.n_states):
            for j in range(self.n_states):
                ax.text(j, i, f"{self.P[i,j]:.2f}", ha="center", va="center", fontsize=12)

        ax.set_xticks(range(self.n_states))
        ax.set_yticks(range(self.n_states))
        ax.set_xticklabels(self.states)
        ax.set_yticklabels(self.states)
        ax.set_xlabel("下一状态")
        ax.set_ylabel("当前状态")
        ax.set_title("状态转移矩阵")
        plt.tight_layout()
        return fig

    def plot_simulation(self, figsize=(12, 4)):
        """绘制模拟轨迹."""
        import matplotlib.pyplot as plt

        if not hasattr(self, '_simulation'):
            self.simulate()

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(self._simulation, "b-o", markersize=3, linewidth=0.8)
        ax.set_yticks(range(self.n_states))
        ax.set_yticklabels(self.states)
        ax.set_xlabel("时间步")
        ax.set_ylabel("状态")
        ax.set_title("马尔可夫链模拟轨迹")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    def summary(self):
        lines = [
            "=" * 50,
            "  马尔可夫链分析结果",
            "=" * 50,
            f"  状态数: {self.n_states}",
            f"  状态: {', '.join(self.states)}",
        ]

        # 稳态分布
        if hasattr(self, '_steady_state'):
            lines.append("-" * 50)
            lines.append("  稳态分布:")
            for i, pi in enumerate(self._steady_state):
                lines.append(f"    {self.states[i]}: {pi:.4f}")

        # 状态分类
        classification = self.classify_states()
        lines.append("-" * 50)
        lines.append("  状态分类:")
        for state, info in classification.items():
            status = "常返" if info["recurrent"] else "非常返"
            lines.append(f"    {state}: {status}, 周期={info['period']}")

        lines.append("=" * 50)
        return "\n".join(lines)
