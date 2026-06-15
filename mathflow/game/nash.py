"""
Nash 均衡求解

支持 2 人有限博弈的纯策略和混合策略 Nash 均衡。

Example:
    >>> from mathflow.game import NashEquilibrium
    >>> # 囚徒困境
    >>> payoff_p1 = [[-1, -3], [0, -2]]  # 玩家1支付矩阵
    >>> payoff_p2 = [[-1, 0], [-3, -2]]  # 玩家2支付矩阵
    >>> ne = NashEquilibrium(payoff_p1, payoff_p2)
    >>> result = ne.find_all()
"""

import numpy as np
from itertools import product
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class NashResult:
    """Nash 均衡结果."""
    pure_strategies: List[Tuple]      # 纯策略 Nash 均衡
    mixed_strategies: Optional[Tuple] # 混合策略 Nash 均衡 (p1_probs, p2_probs)
    payoff_matrix_p1: np.ndarray
    payoff_matrix_p2: np.ndarray
    n_strategies_p1: int
    n_strategies_p2: int


class NashEquilibrium:
    """
    Nash 均衡求解器.

    Parameters
    ----------
    payoff_p1 : array-like, shape (m, n)
        玩家1的支付矩阵 (行=玩家1策略, 列=玩家2策略)
    payoff_p2 : array-like, shape (m, n)
        玩家2的支付矩阵
    p1_names : list of str, optional
        玩家1策略名
    p2_names : list of str, optional
        玩家2策略名
    """

    def __init__(self, payoff_p1, payoff_p2, p1_names=None, p2_names=None):
        self.payoff_p1 = np.asarray(payoff_p1, dtype=float)
        self.payoff_p2 = np.asarray(payoff_p2, dtype=float)
        self.m, self.n = self.payoff_p1.shape
        self.p1_names = p1_names or [f"s{i+1}" for i in range(self.m)]
        self.p2_names = p2_names or [f"t{j+1}" for j in range(self.n)]
        self._result = None

    def find_pure(self) -> List[Tuple]:
        """找纯策略 Nash 均衡."""
        equilibria = []

        for i in range(self.m):
            for j in range(self.n):
                # 检查 (i, j) 是否是 NE
                # 玩家1: 给定玩家2选j, 玩家1选i是否最优
                p1_payoffs_given_j = self.payoff_p1[:, j]
                p1_best = np.max(p1_payoffs_given_j)
                p1_optimal = self.payoff_p1[i, j] >= p1_best - 1e-10

                # 玩家2: 给定玩家1选i, 玩家2选j是否最优
                p2_payoffs_given_i = self.payoff_p2[i, :]
                p2_best = np.max(p2_payoffs_given_i)
                p2_optimal = self.payoff_p2[i, j] >= p2_best - 1e-10

                if p1_optimal and p2_optimal:
                    equilibria.append((i, j))

        return equilibria

    def find_mixed_2x2(self) -> Optional[Tuple]:
        """
        求解 2×2 博弈的混合策略 Nash 均衡.

        仅适用于 2×2 博弈且无纯策略 NE 的情况。
        """
        if self.m != 2 or self.n != 2:
            return None

        a, b = self.payoff_p1[0, 0], self.payoff_p1[0, 1]
        c, d = self.payoff_p1[1, 0], self.payoff_p1[1, 1]
        e, f = self.payoff_p2[0, 0], self.payoff_p2[0, 1]
        g, h = self.payoff_p2[1, 0], self.payoff_p2[1, 1]

        # 玩家1混合概率 p (选策略1的概率)
        denom_p = (a - b - c + d)
        if abs(denom_p) < 1e-10:
            return None
        p = (d - b) / denom_p

        # 玩家2混合概率 q (选策略1的概率)
        denom_q = (e - f - g + h)
        if abs(denom_q) < 1e-10:
            return None
        q = (h - f) / denom_q

        if 0 <= p <= 1 and 0 <= q <= 1:
            return (np.array([p, 1 - p]), np.array([q, 1 - q]))
        return None

    def find_best_response(self, player: int, opponent_strategy):
        """
        求最优响应.

        Parameters
        ----------
        player : int
            1 或 2
        opponent_strategy : array-like
            对手的混合策略概率
        """
        opponent_strategy = np.asarray(opponent_strategy)

        if player == 1:
            # 给定玩家2的策略，玩家1的最优响应
            expected_payoffs = self.payoff_p1 @ opponent_strategy
            return np.argmax(expected_payoffs)
        else:
            expected_payoffs = opponent_strategy @ self.payoff_p2
            return np.argmax(expected_payoffs)

    def dominant_strategy(self, player: int):
        """
        检查是否存在优势策略.

        Returns
        -------
        int or None
            优势策略索引, None 表示不存在
        """
        if player == 1:
            payoffs = self.payoff_p1
        else:
            payoffs = self.payoff_p2

        n_strat = payoffs.shape[0] if player == 1 else payoffs.shape[1]

        for i in range(n_strat):
            dominated = True
            for j in range(n_strat):
                if i == j:
                    continue
                if player == 1:
                    if not np.all(payoffs[i] >= payoffs[j] - 1e-10):
                        dominated = False
                        break
                else:
                    if not np.all(payoffs[:, i] >= payoffs[:, j] - 1e-10):
                        dominated = False
                        break
            if dominated:
                return i
        return None

    def find_all(self) -> NashResult:
        """找所有 Nash 均衡."""
        pure = self.find_pure()
        mixed = self.find_mixed_2x2()

        self._result = NashResult(
            pure_strategies=pure,
            mixed_strategies=mixed,
            payoff_matrix_p1=self.payoff_p1,
            payoff_matrix_p2=self.payoff_p2,
            n_strategies_p1=self.m,
            n_strategies_p2=self.n,
        )
        return self._result

    def plot_payoff_matrix(self, figsize=(10, 6)):
        """绘制支付矩阵."""
        import matplotlib.pyplot as plt

        self._ensure_result()
        r = self._result

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        for idx, (payoff, title) in enumerate([(r.payoff_matrix_p1, "玩家1 支付矩阵"),
                                                 (r.payoff_matrix_p2, "玩家2 支付矩阵")]):
            ax = axes[idx]
            im = ax.imshow(payoff, cmap="YlOrRd", aspect="auto")
            plt.colorbar(im, ax=ax)

            for i in range(r.n_strategies_p1):
                for j in range(r.n_strategies_p2):
                    ax.text(j, i, f"{payoff[i,j]:.0f}", ha="center", va="center", fontsize=12)

            ax.set_xticks(range(r.n_strategies_p2))
            ax.set_yticks(range(r.n_strategies_p1))
            ax.set_xticklabels(self.p2_names)
            ax.set_yticklabels(self.p1_names)
            ax.set_xlabel("玩家2策略")
            ax.set_ylabel("玩家1策略")
            ax.set_title(title)

        # 标记 Nash 均衡
        for i, j in r.pure_strategies:
            for ax in axes:
                ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1,
                                           fill=False, edgecolor="blue", linewidth=3))

        plt.suptitle(f"Nash 均衡 (共 {len(r.pure_strategies)} 个纯策略 NE)", fontsize=14)
        plt.tight_layout()
        return fig

    def _ensure_result(self):
        if self._result is None:
            raise RuntimeError("请先调用 find_all()")

    def summary(self):
        self._ensure_result()
        r = self._result
        lines = [
            "=" * 60,
            "  Nash 均衡分析结果",
            "=" * 60,
            f"  策略空间: 玩家1 ({r.n_strategies_p1} 策略) × 玩家2 ({r.n_strategies_p2} 策略)",
        ]

        # 优势策略
        ds1 = self.dominant_strategy(1)
        ds2 = self.dominant_strategy(2)
        if ds1 is not None:
            lines.append(f"  玩家1优势策略: {self.p1_names[ds1]}")
        if ds2 is not None:
            lines.append(f"  玩家2优势策略: {self.p2_names[ds2]}")

        lines.append("-" * 60)
        lines.append(f"  纯策略 Nash 均衡 ({len(r.pure_strategies)} 个):")
        for i, (s1, s2) in enumerate(r.pure_strategies):
            lines.append(f"    ({self.p1_names[s1]}, {self.p2_names[s2]})  "
                         f"支付=({r.payoff_matrix_p1[s1,s2]:.1f}, {r.payoff_matrix_p2[s1,s2]:.1f})")

        if r.mixed_strategies:
            p, q = r.mixed_strategies
            lines.append("-" * 60)
            lines.append("  混合策略 Nash 均衡:")
            lines.append(f"    玩家1: {dict(zip(self.p1_names, [f'{v:.4f}' for v in p]))}")
            lines.append(f"    玩家2: {dict(zip(self.p2_names, [f'{v:.4f}' for v in q]))}")

        lines.append("=" * 60)
        return "\n".join(lines)
