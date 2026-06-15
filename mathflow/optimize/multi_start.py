"""
多起点优化

通过多次随机初始化避免陷入局部最优。

Example:
    >>> from mathflow.optimize.multi_start import multi_start_optimize
    >>> def f(x): return -(x[0]-3)**2 - (x[1]+2)**2
    >>> result = multi_start_optimize(f, bounds=[(-10,10)]*2, n_starts=10)
"""

import numpy as np
from dataclasses import dataclass
from typing import Callable, List, Tuple


@dataclass
class MultiStartResult:
    """多起点优化结果."""
    best_solution: np.ndarray
    best_value: float
    all_solutions: List[np.ndarray]
    all_values: List[float]
    n_starts: int
    n_success: int


def multi_start_optimize(func: Callable, bounds: List[Tuple[float, float]],
                         n_starts: int = 10, method: str = "ga",
                         maximize: bool = True, seed: int = 42, **kwargs) -> MultiStartResult:
    """
    多起点优化.

    Parameters
    ----------
    func : callable
        目标函数
    bounds : list of (min, max)
        变量范围
    n_starts : int
        随机起点数
    method : str
        优化方法: "ga", "pso", "sa", "scipy"
    maximize : bool
        True 最大化, False 最小化
    """
    np.random.seed(seed)
    n_vars = len(bounds)
    bounds_arr = np.array(bounds)

    all_solutions = []
    all_values = []

    for start in range(n_starts):
        try:
            # 随机初始点
            x0 = np.array([np.random.uniform(bounds_arr[i, 0], bounds_arr[i, 1])
                           for i in range(n_vars)])

            if method == "ga":
                from mathflow.optimize.genetic_algo import GeneticAlgorithm
                opt = GeneticAlgorithm(func, n_vars=n_vars, bounds=bounds,
                                       pop_size=kwargs.get("pop_size", 50),
                                       generations=kwargs.get("generations", 100))
                result = opt.run()
                x_best = result.best_solution
                f_best = result.best_fitness

            elif method == "pso":
                from mathflow.optimize.pso import PSO
                opt = PSO(func, n_vars=n_vars, bounds=bounds,
                          n_particles=kwargs.get("n_particles", 30),
                          max_iter=kwargs.get("max_iter", 100))
                result = opt.run()
                x_best = result.best_position
                f_best = result.best_fitness

            elif method == "scipy":
                from scipy.optimize import minimize
                result = minimize(lambda x: -func(x) if maximize else func(x),
                                  x0, bounds=bounds, method="L-BFGS-B")
                x_best = result.x
                f_best = -result.fun if maximize else result.fun

            else:
                raise ValueError(f"未知方法: {method}")

            all_solutions.append(x_best)
            all_values.append(f_best)

        except Exception:
            continue

    if not all_values:
        raise RuntimeError("所有起点优化失败")

    if maximize:
        best_idx = np.argmax(all_values)
    else:
        best_idx = np.argmin(all_values)

    return MultiStartResult(
        best_solution=all_solutions[best_idx],
        best_value=all_values[best_idx],
        all_solutions=all_solutions,
        all_values=all_values,
        n_starts=n_starts,
        n_success=len(all_values),
    )
