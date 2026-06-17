"""
遗传算法 (Genetic Algorithm)

通用的遗传算法框架，适用于各类优化问题。
内置 TSP、函数优化等经典问题模板。

Example:
    >>> from mathflow.optimize import GeneticAlgorithm
    >>> import numpy as np
    >>> # 函数优化: 求 f(x) = -(x-3)^2 + 9 的最大值
    >>> def fitness(x):
    ...     return -(x[0] - 3)**2 + 9
    >>> ga = GeneticAlgorithm(
    ...     fitness_func=fitness,
    ...     n_vars=1, bounds=[(0, 6)],
    ...     pop_size=50, generations=100
    >>> )
    >>> result = ga.run()
"""

import numpy as np
from dataclasses import dataclass
from typing import Callable, List, Tuple, Optional


@dataclass
class GAResult:
    """遗传算法结果."""
    best_solution: np.ndarray
    best_fitness: float
    history: dict  # {"best": [...], "mean": [...], "worst": [...]}
    convergence_generation: int


class GeneticAlgorithm:
    """
    遗传算法.

    Parameters
    ----------
    fitness_func : callable
        适应度函数 f(x) -> float (越大越好)
    n_vars : int
        变量个数
    bounds : list of (min, max)
        每个变量的取值范围
    pop_size : int
        种群大小
    generations : int
        迭代代数
    crossover_rate : float
        交叉概率
    mutation_rate : float
        变异概率
    elitism : int
        精英保留个数
    encoding : str
        编码方式: "real" (实数编码), "binary" (二进制编码)
    """

    def __init__(self, fitness_func: Callable, n_vars: int,
                 bounds: List[Tuple[float, float]],
                 pop_size: int = 100, generations: int = 200,
                 crossover_rate: float = 0.8, mutation_rate: float = 0.1,
                 elitism: int = 2, encoding: str = "real",
                 seed: Optional[int] = 42):
        # 验证 fitness_func 可调用
        if not callable(fitness_func):
            raise TypeError(f"fitness_func 必须是可调用对象，got {type(fitness_func).__name__}")

        # 验证参数
        if not isinstance(n_vars, int) or n_vars < 1:
            raise ValueError(f"n_vars 必须是正整数，got {n_vars}")
        if len(bounds) != n_vars:
            raise ValueError(f"bounds 长度 ({len(bounds)}) 必须等于 n_vars ({n_vars})")

        self.fitness_func = fitness_func
        self.n_vars = n_vars
        self.bounds = np.array(bounds)
        self.pop_size = pop_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism = elitism
        self.encoding = encoding
        self.seed = seed
        self._result = None

    def __repr__(self) -> str:
        if self._result is not None:
            return f"GeneticAlgorithm(n_vars={self.n_vars}, pop={self.pop_size}, best={self._result.best_fitness:.4f})"
        return f"GeneticAlgorithm(n_vars={self.n_vars}, pop={self.pop_size}, gen={self.generations})"

    def run(self, verbose=False):
        """运行遗传算法."""
        if self.seed is not None:
            np.random.seed(self.seed)

        # 初始化种群
        pop = self._init_population()
        fitness = self._evaluate(pop)

        history = {"best": [], "mean": [], "worst": []}
        best_ever = pop[np.argmax(fitness)].copy()
        best_ever_fitness = fitness.max()
        conv_gen = 0

        for gen in range(self.generations):
            # 选择
            parents = self._selection(pop, fitness)

            # 交叉
            offspring = self._crossover(parents)

            # 变异
            offspring = self._mutation(offspring)

            # 评估
            off_fitness = self._evaluate(offspring)

            # 精英保留 + 替换
            if self.elitism > 0:
                elite_idx = np.argsort(fitness)[-self.elitism:]
                offspring[:self.elitism] = pop[elite_idx]
                off_fitness[:self.elitism] = fitness[elite_idx]

            pop = offspring
            fitness = off_fitness

            # 更新最优
            gen_best_idx = np.argmax(fitness)
            if fitness[gen_best_idx] > best_ever_fitness:
                best_ever = pop[gen_best_idx].copy()
                best_ever_fitness = fitness[gen_best_idx]
                conv_gen = gen

            history["best"].append(best_ever_fitness)
            history["mean"].append(fitness.mean())
            history["worst"].append(fitness.min())

            if verbose and (gen + 1) % 50 == 0:
                print(f"Gen {gen+1}: Best={best_ever_fitness:.6f}, Mean={fitness.mean():.6f}")

        self._result = GAResult(
            best_solution=best_ever,
            best_fitness=best_ever_fitness,
            history=history,
            convergence_generation=conv_gen,
        )
        return self._result

    def _init_population(self):
        """初始化种群."""
        pop = np.zeros((self.pop_size, self.n_vars))
        for i in range(self.n_vars):
            pop[:, i] = np.random.uniform(self.bounds[i, 0], self.bounds[i, 1], self.pop_size)
        return pop

    def _evaluate(self, pop):
        """评估适应度."""
        fitness = np.array([self.fitness_func(ind) for ind in pop])
        # 处理 NaN/Inf 值，替换为最差适应度
        valid_mask = np.isfinite(fitness)
        if not np.all(valid_mask):
            n_invalid = np.sum(~valid_mask)
            if np.any(valid_mask):
                worst_fitness = np.min(fitness[valid_mask]) - 1
            else:
                worst_fitness = -1e10
            fitness[~valid_mask] = worst_fitness
            if n_invalid > 0:
                import warnings
                warnings.warn(f"检测到 {n_invalid} 个无效适应度值 (NaN/Inf)，已替换为最差值")
        return fitness

    def _selection(self, pop, fitness):
        """锦标赛选择."""
        selected = np.zeros_like(pop)
        for i in range(len(pop)):
            # 锦标赛大小 = 3
            candidates = np.random.choice(len(pop), 3, replace=False)
            best = candidates[np.argmax(fitness[candidates])]
            selected[i] = pop[best]
        return selected

    def _crossover(self, parents):
        """模拟二进制交叉 (SBX) 或算术交叉."""
        offspring = parents.copy()
        for i in range(0, len(parents) - 1, 2):
            if np.random.random() < self.crossover_rate:
                alpha = np.random.random()
                offspring[i] = alpha * parents[i] + (1 - alpha) * parents[i + 1]
                offspring[i + 1] = (1 - alpha) * parents[i] + alpha * parents[i + 1]
        # 边界修复
        offspring = np.clip(offspring, self.bounds[:, 0], self.bounds[:, 1])
        return offspring

    def _mutation(self, pop):
        """高斯变异."""
        for i in range(len(pop)):
            for j in range(self.n_vars):
                if np.random.random() < self.mutation_rate:
                    sigma = (self.bounds[j, 1] - self.bounds[j, 0]) * 0.1
                    pop[i, j] += np.random.normal(0, sigma)
                    pop[i, j] = np.clip(pop[i, j], self.bounds[j, 0], self.bounds[j, 1])
        return pop

    def plot_convergence(self, figsize=(10, 5)):
        """绘制收敛曲线."""
        import matplotlib.pyplot as plt

        self._ensure_result()
        h = self._result.history

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(h["best"], "g-", linewidth=2, label="最优适应度")
        ax.plot(h["mean"], "b--", alpha=0.7, label="平均适应度")
        ax.fill_between(range(len(h["best"])), h["worst"], h["best"], alpha=0.1, color="green")
        ax.set_xlabel("迭代代数")
        ax.set_ylabel("适应度")
        ax.set_title("遗传算法收敛曲线")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    def _ensure_result(self):
        if self._result is None:
            raise RuntimeError("请先调用 run() 运行算法")

    def summary(self):
        """打印结果摘要."""
        self._ensure_result()
        r = self._result
        lines = [
            "=" * 50,
            "  遗传算法 (GA) 结果",
            "=" * 50,
            f"  最优适应度: {r.best_fitness:.6f}",
            f"  收敛代数: {r.convergence_generation}",
            "-" * 50,
            "  最优解:",
        ]
        for i, v in enumerate(r.best_solution):
            lines.append(f"    x{i+1} = {v:.6f}")
        lines.append("=" * 50)
        return "\n".join(lines)
