"""
线性规划 (Linear Programming)

使用 PuLP 求解线性规划问题。
支持可视化可行域 (二维问题)。

Example:
    >>> from mathflow.optimize import LinearProgramming
    >>> lp = LinearProgramming()
    >>> lp.set_objective([4, 3], sense="max")
    >>> lp.add_constraint([2, 1], "<=", 10)
    >>> lp.add_constraint([1, 1], "<=", 8)
    >>> lp.add_constraint([1, 0], ">=", 0)
    >>> lp.add_constraint([0, 1], ">=", 0)
    >>> result = lp.solve()
    >>> print(result)
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class LPResult:
    """线性规划结果."""
    status: str
    optimal_value: float
    solution: np.ndarray
    variables: dict


class LinearProgramming:
    """
    线性规划求解器.

    支持:
    - 最大化/最小化目标函数
    - <=, >=, == 约束
    - 可行域可视化 (二维)
    """

    def __init__(self):
        self._obj_coeffs = None
        self._obj_sense = "min"
        self._constraints = []  # [(coeffs, sense, rhs)]
        self._var_names = None
        self._bounds = []
        self._result = None

    def _ensure_result(self):
        if self._result is None:
            raise RuntimeError("请先调用 solve() 求解")

    def __repr__(self) -> str:
        if self._result is not None:
            return f"LinearProgramming(status={self._result.status!r}, obj={self._result.optimal_value:.4f})"
        return f"LinearProgramming(sense={self._obj_sense!r})"

    def set_objective(self, coeffs, sense="min", var_names=None):
        """
        设置目标函数.

        Parameters
        ----------
        coeffs : list
            目标函数系数
        sense : str
            "min" 或 "max"
        var_names : list of str, optional
            变量名
        """
        self._obj_coeffs = np.asarray(coeffs, dtype=float)
        self._obj_sense = sense
        n = len(coeffs)
        if var_names is None:
            self._var_names = [f"x{i+1}" for i in range(n)]
        else:
            self._var_names = var_names
        return self

    def add_constraint(self, coeffs, sense, rhs):
        """
        添加约束条件.

        Parameters
        ----------
        coeffs : list
            约束系数
        sense : str
            "<=", ">=", "=="
        rhs : float
            右端值
        """
        self._constraints.append((np.asarray(coeffs, dtype=float), sense, float(rhs)))
        return self

    def solve(self):
        """求解线性规划."""
        import pulp

        n = len(self._obj_coeffs)

        # 创建问题
        if self._obj_sense == "max":
            prob = pulp.LpProblem("LP", pulp.LpMaximize)
        else:
            prob = pulp.LpProblem("LP", pulp.LpMinimize)

        # 创建变量
        x = [pulp.LpVariable(self._var_names[i], lowBound=0) for i in range(n)]

        # 目标函数
        prob += pulp.lpDot(self._obj_coeffs, x)

        # 约束条件
        for coeffs, sense, rhs in self._constraints:
            expr = pulp.lpDot(coeffs, x)
            if sense == "<=":
                prob += expr <= rhs
            elif sense == ">=":
                prob += expr >= rhs
            elif sense == "==":
                prob += expr == rhs

        # 求解
        prob.solve(pulp.PULP_CBC_CMD(msg=0))

        status = pulp.LpStatus[prob.status]
        solution = np.array([var.varValue for var in x])
        optimal_value = pulp.value(prob.objective)

        variables = {self._var_names[i]: solution[i] for i in range(n)}

        self._prob = prob
        self._x = x
        self._result = LPResult(
            status=status,
            optimal_value=optimal_value,
            solution=solution,
            variables=variables,
        )
        return self._result

    def plot_feasible_region(self, x_range=(0, 15), y_range=(0, 15), figsize=(8, 8)):
        """绘制可行域 (仅限二维问题)."""
        import matplotlib.pyplot as plt

        if len(self._obj_coeffs) != 2:
            raise ValueError("可行域可视化仅支持二维问题")

        fig, ax = plt.subplots(figsize=figsize)

        # 生成网格
        x = np.linspace(x_range[0], x_range[1], 400)
        y = np.linspace(y_range[0], y_range[1], 400)
        X, Y = np.meshgrid(x, y)

        # 绘制约束线和可行域
        feasible = np.ones_like(X, dtype=bool)
        colors = ["blue", "red", "green", "purple", "orange", "brown"]
        for i, (coeffs, sense, rhs) in enumerate(self._constraints):
            c = colors[i % len(colors)]
            if coeffs[1] != 0:
                y_line = (rhs - coeffs[0] * x) / coeffs[1]
                ax.plot(x, y_line, c=c, linewidth=2, label=f"约束{i+1}: {coeffs[0]}x₁+{coeffs[1]}x₂ {sense} {rhs}")

                if sense == "<=":
                    feasible &= (coeffs[0] * X + coeffs[1] * Y <= rhs)
                elif sense == ">=":
                    feasible &= (coeffs[0] * X + coeffs[1] * Y >= rhs)
            else:
                x_line = rhs / coeffs[0]
                ax.axvline(x=x_line, color=c, linewidth=2, label=f"约束{i+1}: {coeffs[0]}x₁ {sense} {rhs}")
                if sense == "<=":
                    feasible &= (coeffs[0] * X <= rhs)
                elif sense == ">=":
                    feasible &= (coeffs[0] * X >= rhs)

        # 填充可行域
        ax.contourf(X, Y, feasible.astype(int), levels=[0.5, 1.5], colors=["lightblue"], alpha=0.5)

        # 绘制等值线
        if self._result is not None:
            opt = self._result.optimal_value
            z = self._obj_coeffs[0] * X + self._obj_coeffs[1] * Y
            ax.contour(X, Y, z, levels=[opt], colors=["gold"], linewidths=3, linestyles="--")
            ax.plot(self._result.solution[0], self._result.solution[1], "r*",
                    markersize=20, label=f"最优解 ({self._result.solution[0]:.2f}, {self._result.solution[1]:.2f})")

        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        ax.set_xlabel("x₁", fontsize=12)
        ax.set_ylabel("x₂", fontsize=12)
        ax.set_title("线性规划 - 可行域与最优解", fontsize=14)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def summary(self):
        """打印结果摘要."""
        self._ensure_result()
        r = self._result
        lines = [
            "=" * 50,
            "  线性规划求解结果",
            "=" * 50,
            f"  状态: {r.status}",
            f"  最优值: {r.optimal_value:.4f}",
            "-" * 50,
        ]
        for name, val in r.variables.items():
            lines.append(f"  {name} = {val:.4f}")
        lines.append("=" * 50)
        return "\n".join(lines)
