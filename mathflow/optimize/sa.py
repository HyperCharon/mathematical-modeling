"""
模拟退火算法 (Simulated Annealing, SA)

Example:
    >>> from mathflow.optimize import SimulatedAnnealing
    >>> def rastrigin(x):
    ...     return -(10*len(x) + sum(xi**2 - 10*np.cos(2*np.pi*xi) for xi in x))
    >>> sa = SimulatedAnnealing(rastrigin, n_vars=2, bounds=[(-5,5)]*2)
    >>> result = sa.run()
"""

import numpy as np
from dataclasses import dataclass
from typing import Callable, List, Tuple


@dataclass
class SAResult:
    """模拟退火结果."""
    best_solution: np.ndarray
    best_energy: float
    history: dict
    acceptance_rate: float


class SimulatedAnnealing:
    """
    模拟退火算法.

    Parameters
    ----------
    objective : callable
        目标函数 (越小越好)
    n_vars : int
        变量维度
    bounds : list of (min, max)
        各变量范围
    T_init : float
        初始温度
    T_min : float
        终止温度
    cooling_rate : float
        降温速率 (0, 1)
    max_iter_per_temp : int
        每个温度下的迭代次数
    """

    def __init__(self, objective: Callable, n_vars: int,
                 bounds: List[Tuple[float, float]],
                 T_init: float = 1000, T_min: float = 1e-8,
                 cooling_rate: float = 0.995, max_iter_per_temp: int = 100):
        self.objective = objective
        self.n_vars = n_vars
        self.bounds = np.array(bounds)
        self.T_init = T_init
        self.T_min = T_min
        self.cooling_rate = cooling_rate
        self.max_iter_per_temp = max_iter_per_temp
        self._result = None

    def run(self, verbose=False):
        """运行模拟退火."""
        np.random.seed(42)

        # 初始化
        current = np.zeros(self.n_vars)
        for i in range(self.n_vars):
            current[i] = np.random.uniform(self.bounds[i, 0], self.bounds[i, 1])
        current_energy = self.objective(current)

        best = current.copy()
        best_energy = current_energy

        T = self.T_init
        history = {"energy": [], "temperature": [], "best_energy": []}
        total_accepts = 0
        total_tries = 0

        while T > self.T_min:
            for _ in range(self.max_iter_per_temp):
                total_tries += 1

                # 产生新解 (高斯扰动)
                neighbor = current.copy()
                i = np.random.randint(self.n_vars)
                step = np.random.normal(0, T / self.T_init * (self.bounds[i, 1] - self.bounds[i, 0]) * 0.1)
                neighbor[i] = np.clip(neighbor[i] + step, self.bounds[i, 0], self.bounds[i, 1])

                neighbor_energy = self.objective(neighbor)
                delta = neighbor_energy - current_energy

                # Metropolis 准则
                if delta < 0 or np.random.random() < np.exp(-delta / max(T, 1e-10)):
                    current = neighbor
                    current_energy = neighbor_energy
                    total_accepts += 1

                    if current_energy < best_energy:
                        best = current.copy()
                        best_energy = current_energy

            history["energy"].append(current_energy)
            history["temperature"].append(T)
            history["best_energy"].append(best_energy)

            T *= self.cooling_rate

            if verbose and len(history["energy"]) % 100 == 0:
                print(f"T={T:.4f}, Energy={best_energy:.6f}")

        self._result = SAResult(
            best_solution=best,
            best_energy=best_energy,
            history=history,
            acceptance_rate=total_accepts / total_tries if total_tries > 0 else 0,
        )
        return self._result

    def plot(self, figsize=(12, 5)):
        """绘制退火过程."""
        import matplotlib.pyplot as plt

        self._ensure_result()
        h = self._result.history

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        ax = axes[0]
        ax.plot(h["best_energy"], "g-", linewidth=2, label="最优能量")
        ax.plot(h["energy"], "b-", alpha=0.3, label="当前能量")
        ax.set_xlabel("温度步")
        ax.set_ylabel("能量")
        ax.set_title("模拟退火过程")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax2 = axes[1]
        ax2.semilogy(h["temperature"], h["best_energy"], "r-")
        ax2.set_xlabel("温度")
        ax2.set_ylabel("最优能量")
        ax2.set_title("温度 vs 最优能量")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def _ensure_result(self):
        if self._result is None:
            raise RuntimeError("请先调用 run() 运行算法")

    def summary(self):
        self._ensure_result()
        r = self._result
        lines = ["=" * 50, "  模拟退火 (SA) 结果", "=" * 50,
                  f"  最优能量: {r.best_energy:.6f}",
                  f"  接受率: {r.acceptance_rate:.2%}", "-" * 50, "  最优解:"]
        for i, v in enumerate(r.best_solution):
            lines.append(f"    x{i+1} = {v:.6f}")
        lines.append("=" * 50)
        return "\n".join(lines)
