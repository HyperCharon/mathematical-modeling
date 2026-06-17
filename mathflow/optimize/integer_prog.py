"""
整数规划 (Integer Programming)

支持纯整数规划和混合整数规划。

Example:
    >>> from mathflow.optimize import IntegerProgramming
    >>> ip = IntegerProgramming()
    >>> ip.set_objective([5, 8], sense="max")
    >>> ip.add_constraint([1, 1], "<=", 6)
    >>> ip.add_constraint([5, 9], "<=", 45)
    >>> result = ip.solve(integer_vars=[0, 1])
    >>> print(result)
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class IPResult:
    """整数规划结果."""
    status: str
    optimal_value: float
    solution: np.ndarray
    variables: dict


class IntegerProgramming:
    """整数规划求解器."""

    def __init__(self):
        self._obj_coeffs = None
        self._obj_sense = "min"
        self._constraints = []
        self._var_names = None

    def __repr__(self) -> str:
        if hasattr(self, '_ip_result') and self._ip_result is not None:
            return f"IntegerProgramming(status={self._ip_result.status!r}, obj={self._ip_result.optimal_value:.4f})"
        return f"IntegerProgramming(sense={self._obj_sense!r})"

    def set_objective(self, coeffs, sense="min", var_names=None):
        """设置目标函数."""
        self._obj_coeffs = np.asarray(coeffs, dtype=float)
        self._obj_sense = sense
        n = len(coeffs)
        if var_names is None:
            self._var_names = [f"x{i+1}" for i in range(n)]
        else:
            self._var_names = var_names
        return self

    def add_constraint(self, coeffs, sense, rhs):
        """添加约束条件."""
        self._constraints.append((np.asarray(coeffs, dtype=float), sense, float(rhs)))
        return self

    def solve(self, integer_vars=None, binary_vars=None):
        """
        求解整数规划.

        Parameters
        ----------
        integer_vars : list of int, optional
            整数变量的索引 (默认全部为整数)
        binary_vars : list of int, optional
            0-1 变量的索引
        """
        import pulp

        n = len(self._obj_coeffs)

        if self._obj_sense == "max":
            prob = pulp.LpProblem("IP", pulp.LpMaximize)
        else:
            prob = pulp.LpProblem("IP", pulp.LpMinimize)

        # 创建变量
        if binary_vars is None:
            binary_vars = []
        if integer_vars is None:
            integer_vars = list(range(n))

        x = []
        for i in range(n):
            if i in binary_vars:
                x.append(pulp.LpVariable(self._var_names[i], cat="Binary"))
            elif i in integer_vars:
                x.append(pulp.LpVariable(self._var_names[i], lowBound=0, cat="Integer"))
            else:
                x.append(pulp.LpVariable(self._var_names[i], lowBound=0))

        # 目标函数
        prob += pulp.lpDot(self._obj_coeffs, x)

        # 约束
        for coeffs, sense, rhs in self._constraints:
            expr = pulp.lpDot(coeffs, x)
            if sense == "<=":
                prob += expr <= rhs
            elif sense == ">=":
                prob += expr >= rhs
            elif sense == "==":
                prob += expr == rhs

        prob.solve(pulp.PULP_CBC_CMD(msg=0))

        status = pulp.LpStatus[prob.status]
        solution = np.array([var.varValue for var in x])
        optimal_value = pulp.value(prob.objective)
        variables = {self._var_names[i]: solution[i] for i in range(n)}

        self._ip_result = IPResult(
            status=status, optimal_value=optimal_value,
            solution=solution, variables=variables,
        )
        return self._ip_result

    def summary(self):
        """打印结果摘要."""
        if not hasattr(self, '_ip_result') or self._ip_result is None:
            raise RuntimeError("请先调用 solve() 求解")
        r = self._ip_result
        lines = ["=" * 50, "  整数规划求解结果", "=" * 50,
                  f"  状态: {r.status}", f"  最优值: {r.optimal_value:.4f}", "-" * 50]
        for name, val in r.variables.items():
            lines.append(f"  {name} = {int(val)}")
        lines.append("=" * 50)
        return "\n".join(lines)
