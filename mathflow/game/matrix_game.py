"""
矩阵博弈 (零和博弈)

Example:
    >>> from mathflow.game import MatrixGame
    >>> payoff = [[3, 1], [0, 2]]  # 玩家A的收益矩阵 (B的收益 = -A)
    >>> game = MatrixGame(payoff)
    >>> result = game.solve()
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class MatrixGameResult:
    """矩阵博弈结果."""
    value: float              # 博弈值
    strategy_p1: np.ndarray   # 玩家1最优混合策略
    strategy_p2: np.ndarray   # 玩家2最优混合策略
    has_saddle_point: bool
    saddle_point: Optional[tuple]


class MatrixGame:
    """
    零和矩阵博弈.

    Parameters
    ----------
    payoff : array-like, shape (m, n)
        玩家A的收益矩阵 (B的收益 = -A的收益)
    """

    def __init__(self, payoff):
        self.payoff = np.asarray(payoff, dtype=float)
        self.m, self.n = self.payoff.shape
        self._result = None

    def __repr__(self) -> str:
        if self._result is not None:
            return f"MatrixGame({self.m}x{self.n}, value={self._result.value:.4f})"
        return f"MatrixGame({self.m}x{self.n})"

    def _ensure_result(self) -> None:
        if self._result is None:
            raise RuntimeError("请先调用 solve() 方法")

    def solve(self) -> MatrixGameResult:
        """求解矩阵博弈."""
        # 先检查鞍点
        row_mins = self.payoff.min(axis=1)
        col_maxs = self.payoff.max(axis=0)

        maximin = row_mins.max()
        minimax = col_maxs.min()

        has_saddle = abs(maximin - minimax) < 1e-10
        saddle_point = None

        if has_saddle:
            # 鞍点存在
            i = np.argmax(row_mins)
            j = np.argmin(col_maxs)
            saddle_point = (i, j)
            strategy_p1 = np.zeros(self.m)
            strategy_p1[i] = 1
            strategy_p2 = np.zeros(self.n)
            strategy_p2[j] = 1
            value = maximin
        else:
            # 用线性规划求解
            value, strategy_p1, strategy_p2 = self._solve_lp()

        self._result = MatrixGameResult(
            value=value,
            strategy_p1=strategy_p1,
            strategy_p2=strategy_p2,
            has_saddle_point=has_saddle,
            saddle_point=saddle_point,
        )
        return self._result

    def _solve_lp(self):
        """用线性规划求解混合策略."""
        import pulp

        # 玩家1: max v, s.t. A^T p >= v, sum(p)=1, p>=0
        prob = pulp.LpProblem("MatrixGame", pulp.LpMaximize)
        v = pulp.LpVariable("v")
        p = [pulp.LpVariable(f"p{i}", lowBound=0) for i in range(self.m)]

        prob += v

        for j in range(self.n):
            prob += pulp.lpDot(self.payoff[:, j], p) >= v

        prob += pulp.lpSum(p) == 1
        prob.solve(pulp.PULP_CBC_CMD(msg=0))

        value = pulp.value(v)
        strategy_p1 = np.array([pulp.value(pi) for pi in p])

        # 玩家2: min v, s.t. A q <= v, sum(q)=1, q>=0
        prob2 = pulp.LpProblem("MatrixGame2", pulp.LpMinimize)
        v2 = pulp.LpVariable("v2")
        q = [pulp.LpVariable(f"q{j}", lowBound=0) for j in range(self.n)]

        prob2 += v2

        for i in range(self.m):
            prob2 += pulp.lpDot(self.payoff[i, :], q) <= v2

        prob2 += pulp.lpSum(q) == 1
        prob2.solve(pulp.PULP_CBC_CMD(msg=0))

        strategy_p2 = np.array([pulp.value(qj) for qj in q])

        return value, strategy_p1, strategy_p2

    @property
    def result(self):
        if self._result is None:
            raise RuntimeError("请先调用 solve()")
        return self._result

    def summary(self):
        self._ensure_result()
        r = self._result
        lines = [
            "=" * 50,
            "  矩阵博弈结果",
            "=" * 50,
            f"  博弈值 v = {r.value:.4f}",
            f"  鞍点: {'存在' if r.has_saddle_point else '不存在'}",
        ]
        if r.saddle_point:
            lines.append(f"  鞍点位置: {r.saddle_point}")

        lines.append("-" * 50)
        lines.append("  玩家A 最优策略:")
        for i, prob in enumerate(r.strategy_p1):
            if prob > 1e-6:
                lines.append(f"    策略{i+1}: {prob:.4f}")
        lines.append("  玩家B 最优策略:")
        for j, prob in enumerate(r.strategy_p2):
            if prob > 1e-6:
                lines.append(f"    策略{j+1}: {prob:.4f}")
        lines.append("=" * 50)
        return "\n".join(lines)
