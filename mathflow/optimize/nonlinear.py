"""
非线性规划 (Nonlinear Programming)

支持带约束的非线性优化问题求解。

Example:
    >>> from mathflow.optimize import NonlinearProgramming
    >>> # min x^2 + y^2, s.t. x + y >= 1
    >>> nlp = NonlinearProgramming()
    >>> nlp.set_objective(lambda x: x[0]**2 + x[1]**2)
    >>> nlp.add_constraint(lambda x: x[0] + x[1] - 1, type="ineq")  # >= 0
    >>> result = nlp.solve(x0=[0.5, 0.5])
    >>> print(f"最优解: {result.solution}, 最优值: {result.optimal_value}")
"""

import numpy as np
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple
from scipy.optimize import minimize, NonlinearConstraint


@dataclass
class NLPResult:
    """非线性规划结果."""
    solution: np.ndarray
    optimal_value: float
    success: bool
    message: str
    n_iter: int
    method: str


class NonlinearProgramming:
    """
    非线性规划求解器.

    支持:
    - 等式约束
    - 不等式约束
    - 变量边界
    - 多种求解方法
    """

    def __init__(self):
        self.objective = None
        self.constraints = []
        self.bounds = None
        self._result = None

    def __repr__(self) -> str:
        if self._result is not None:
            return f"NonlinearProgramming(success={self._result.success}, obj={self._result.optimal_value:.4f})"
        return f"NonlinearProgramming()"

    def set_objective(self, func: Callable):
        """
        设置目标函数.

        Parameters
        ----------
        func : callable
            目标函数 f(x) -> float
        """
        if not callable(func):
            raise TypeError("目标函数必须是可调用对象")
        self.objective = func
        return self

    def add_constraint(self, func: Callable, type: str = "ineq"):
        """
        添加约束条件.

        Parameters
        ----------
        func : callable
            约束函数 g(x) -> float
        type : str
            "ineq" (不等式 g(x) >= 0) 或 "eq" (等式 g(x) = 0)
        """
        if not callable(func):
            raise TypeError("约束函数必须是可调用对象")
        if type not in ("ineq", "eq"):
            raise ValueError(f"约束类型必须是 'ineq' 或 'eq'，got '{type}'")

        self.constraints.append({"type": type, "fun": func})
        return self

    def set_bounds(self, bounds: List[Tuple[Optional[float], Optional[float]]]):
        """
        设置变量边界.

        Parameters
        ----------
        bounds : list of (lower, upper)
            每个变量的上下界，None 表示无界
        """
        self.bounds = bounds
        return self

    def solve(self, x0: np.ndarray, method: str = "SLSQP",
              max_iter: int = 1000, tol: float = 1e-8) -> NLPResult:
        """
        求解非线性规划.

        Parameters
        ----------
        x0 : array-like
            初始点
        method : str
            求解方法: "SLSQP", "trust-constr", "COBYLA", "Nelder-Mead"
        max_iter : int
            最大迭代次数
        tol : float
            收敛精度
        """
        if self.objective is None:
            raise RuntimeError("请先调用 set_objective() 设置目标函数")

        x0 = np.asarray(x0, dtype=float)

        # 设置优化选项
        options = {"maxiter": max_iter}

        try:
            if method in ("SLSQP", "trust-constr"):
                result = minimize(
                    self.objective, x0,
                    method=method,
                    constraints=self.constraints,
                    bounds=self.bounds,
                    tol=tol,
                    options=options,
                )
            elif method == "COBYLA":
                # COBYLA 只支持不等式约束
                cobyla_constraints = []
                for c in self.constraints:
                    if c["type"] == "ineq":
                        cobyla_constraints.append(c["fun"])
                    else:
                        # 等式约束转为两个不等式约束
                        cobyla_constraints.append(lambda x, f=c["fun"]: f(x))
                        cobyla_constraints.append(lambda x, f=c["fun"]: -f(x))

                result = minimize(
                    self.objective, x0,
                    method="COBYLA",
                    constraints=[{"type": "ineq", "fun": c} for c in cobyla_constraints],
                    tol=tol,
                    options=options,
                )
            else:
                # 无约束方法
                result = minimize(
                    self.objective, x0,
                    method=method,
                    tol=tol,
                    options=options,
                )

            self._result = NLPResult(
                solution=result.x,
                optimal_value=result.fun,
                success=result.success,
                message=result.message,
                n_iter=result.nit if hasattr(result, "nit") else 0,
                method=method,
            )

        except Exception as e:
            self._result = NLPResult(
                solution=x0,
                optimal_value=self.objective(x0),
                success=False,
                message=str(e),
                n_iter=0,
                method=method,
            )

        return self._result

    @property
    def result(self):
        if self._result is None:
            raise RuntimeError("请先调用 solve()")
        return self._result

    def summary(self) -> str:
        """打印结果摘要."""
        if self._result is None:
            raise RuntimeError("请先调用 solve()")

        r = self._result
        lines = [
            "=" * 50,
            "  非线性规划求解结果",
            "=" * 50,
            f"  方法: {r.method}",
            f"  是否成功: {'是' if r.success else '否'}",
            f"  最优值: {r.optimal_value:.6f}",
            f"  迭代次数: {r.n_iter}",
            "-" * 50,
            "  最优解:",
        ]

        for i, val in enumerate(r.solution):
            lines.append(f"    x{i+1} = {val:.6f}")

        if not r.success:
            lines.append("-" * 50)
            lines.append(f"  信息: {r.message}")

        lines.append("=" * 50)
        return "\n".join(lines)
