"""
粒子群优化算法 (Particle Swarm Optimization, PSO)

Example:
    >>> from mathflow.optimize import PSO
    >>> def sphere(x):
    ...     return -sum(xi**2 for xi in x)  # 最大化
    >>> pso = PSO(sphere, n_vars=3, bounds=[(-5,5)]*3)
    >>> result = pso.run()
"""

import numpy as np
from dataclasses import dataclass
from typing import Callable, List, Tuple


@dataclass
class PSOResult:
    """PSO 结果."""
    best_position: np.ndarray
    best_fitness: float
    history: dict
    convergence_generation: int


class PSO:
    """
    粒子群优化算法.

    Parameters
    ----------
    fitness_func : callable
        适应度函数 (越大越好)
    n_vars : int
        变量维度
    bounds : list of (min, max)
        各变量范围
    n_particles : int
        粒子数量
    max_iter : int
        最大迭代次数
    w : float
        惯性权重
    c1 : float
        个体学习因子
    c2 : float
        社会学习因子
    """

    def __init__(self, fitness_func: Callable, n_vars: int,
                 bounds: List[Tuple[float, float]],
                 n_particles: int = 50, max_iter: int = 200,
                 w: float = 0.7, c1: float = 1.5, c2: float = 1.5):
        self.fitness_func = fitness_func
        self.n_vars = n_vars
        self.bounds = np.array(bounds)
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self._result = None

    def run(self, verbose=False):
        """运行 PSO."""
        np.random.seed(42)

        n = self.n_particles
        d = self.n_vars

        # 初始化位置和速度
        pos = np.zeros((n, d))
        vel = np.zeros((n, d))
        for i in range(d):
            pos[:, i] = np.random.uniform(self.bounds[i, 0], self.bounds[i, 1], n)
            vel[:, i] = np.random.uniform(
                -(self.bounds[i, 1] - self.bounds[i, 0]) * 0.1,
                (self.bounds[i, 1] - self.bounds[i, 0]) * 0.1, n
            )

        # 评估初始适应度
        fitness = np.array([self.fitness_func(p) for p in pos])

        # 个体最优和全局最优
        p_best_pos = pos.copy()
        p_best_fit = fitness.copy()
        g_best_idx = np.argmax(fitness)
        g_best_pos = pos[g_best_idx].copy()
        g_best_fit = fitness[g_best_idx]

        history = {"best": [], "mean": []}
        conv_gen = 0

        for it in range(self.max_iter):
            # 自适应惯性权重 (线性递减)
            w = self.w - (self.w - 0.4) * it / self.max_iter

            for i in range(n):
                # 更新速度
                r1, r2 = np.random.random(d), np.random.random(d)
                vel[i] = (w * vel[i] +
                          self.c1 * r1 * (p_best_pos[i] - pos[i]) +
                          self.c2 * r2 * (g_best_pos - pos[i]))

                # 更新位置
                pos[i] += vel[i]

                # 边界处理
                pos[i] = np.clip(pos[i], self.bounds[:, 0], self.bounds[:, 1])

                # 评估
                f = self.fitness_func(pos[i])
                fitness[i] = f

                # 更新个体最优
                if f > p_best_fit[i]:
                    p_best_fit[i] = f
                    p_best_pos[i] = pos[i].copy()

                    # 更新全局最优
                    if f > g_best_fit:
                        g_best_fit = f
                        g_best_pos = pos[i].copy()
                        conv_gen = it

            history["best"].append(g_best_fit)
            history["mean"].append(fitness.mean())

            if verbose and (it + 1) % 50 == 0:
                print(f"Iter {it+1}: Best={g_best_fit:.6f}")

        self._result = PSOResult(
            best_position=g_best_pos,
            best_fitness=g_best_fit,
            history=history,
            convergence_generation=conv_gen,
        )
        return self._result

    def plot_convergence(self, figsize=(10, 5)):
        """绘制收敛曲线."""
        import matplotlib.pyplot as plt

        self._ensure_result()
        h = self._result.history

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(h["best"], "g-", linewidth=2, label="全局最优")
        ax.plot(h["mean"], "b--", alpha=0.7, label="平均适应度")
        ax.set_xlabel("迭代次数")
        ax.set_ylabel("适应度")
        ax.set_title("PSO 粒子群优化收敛曲线")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    def _ensure_result(self):
        if self._result is None:
            raise RuntimeError("请先调用 run() 运行算法")

    def summary(self):
        self._ensure_result()
        r = self._result
        lines = ["=" * 50, "  PSO 粒子群优化结果", "=" * 50,
                  f"  最优适应度: {r.best_fitness:.6f}",
                  f"  收敛代数: {r.convergence_generation}", "-" * 50, "  最优解:"]
        for i, v in enumerate(r.best_position):
            lines.append(f"    x{i+1} = {v:.6f}")
        lines.append("=" * 50)
        return "\n".join(lines)
