"""
多目标优化 (NSGA-II)

支持 Pareto 前沿求解，适用于数模中的多目标决策问题。

Example:
    >>> from mathflow.optimize import NSGA2
    >>> # ZDT1 测试函数
    >>> def f1(x): return x[0]
    >>> def f2(x): return 1 - np.sqrt(x[0]) + x[0] * np.sum(x[1:])
    >>> nsga = NSGA2([f1, f2], n_vars=3, bounds=[(0,1)]*3)
    >>> result = nsga.run()
"""

import numpy as np
from dataclasses import dataclass
from typing import Callable, List, Tuple


@dataclass
class NSGAResult:
    """NSGA-II 结果."""
    pareto_front: np.ndarray      # Pareto 前沿目标值 (n_solutions, n_objectives)
    pareto_solutions: np.ndarray   # Pareto 最优解 (n_solutions, n_vars)
    n_generations: int
    n_solutions: int


class NSGA2:
    """
    NSGA-II 多目标优化算法.

    Parameters
    ----------
    objectives : list of callable
        目标函数列表 (全部最小化)
    n_vars : int
        变量维度
    bounds : list of (min, max)
        各变量范围
    pop_size : int
        种群大小
    generations : int
        迭代代数
    """

    def __init__(self, objectives: List[Callable], n_vars: int,
                 bounds: List[Tuple[float, float]],
                 pop_size: int = 100, generations: int = 200):
        self.objectives = objectives
        self.n_obj = len(objectives)
        self.n_vars = n_vars
        self.bounds = np.array(bounds)
        self.pop_size = pop_size
        self.generations = generations
        self._result = None

    def run(self, verbose=False):
        """运行 NSGA-II."""
        np.random.seed(42)

        # 初始化种群
        pop = self._init_population()
        obj_values = self._evaluate(pop)

        for gen in range(self.generations):
            # 非支配排序
            fronts = self._non_dominated_sort(obj_values)
            crowding = self._crowding_distance(obj_values, fronts)

            # 选择、交叉、变异
            offspring = self._generate_offspring(pop, obj_values, fronts, crowding)
            off_obj = self._evaluate(offspring)

            # 合并父代和子代
            combined_pop = np.vstack([pop, offspring])
            combined_obj = np.vstack([obj_values, off_obj])

            # 环境选择
            fronts = self._non_dominated_sort(combined_obj)
            crowding = self._crowding_distance(combined_obj, fronts)

            new_pop = []
            new_obj = []
            for front in fronts:
                if len(new_pop) + len(front) <= self.pop_size:
                    new_pop.extend(combined_pop[front])
                    new_obj.extend(combined_obj[front])
                else:
                    # 按拥挤度排序
                    remaining = self.pop_size - len(new_pop)
                    sorted_by_crowd = sorted(front, key=lambda i: crowding[i], reverse=True)
                    new_pop.extend(combined_pop[sorted_by_crowd[:remaining]])
                    new_obj.extend(combined_obj[sorted_by_crowd[:remaining]])
                    break

            pop = np.array(new_pop)
            obj_values = np.array(new_obj)

            if verbose and (gen + 1) % 50 == 0:
                print(f"Gen {gen+1}: Pareto solutions = {len(fronts[0])}")

        # 提取 Pareto 前沿
        final_fronts = self._non_dominated_sort(obj_values)
        pareto_idx = final_fronts[0]

        # Deduplicate
        unique_solutions = {}
        for idx in pareto_idx:
            key = tuple(np.round(pop[idx], 6))
            if key not in unique_solutions:
                unique_solutions[key] = idx
        pareto_idx = list(unique_solutions.values())

        self._result = NSGAResult(
            pareto_front=obj_values[pareto_idx],
            pareto_solutions=pop[pareto_idx],
            n_generations=self.generations,
            n_solutions=len(pareto_idx),
        )
        return self._result

    def _init_population(self):
        pop = np.zeros((self.pop_size, self.n_vars))
        for i in range(self.n_vars):
            pop[:, i] = np.random.uniform(self.bounds[i, 0], self.bounds[i, 1], self.pop_size)
        return pop

    def _evaluate(self, pop):
        obj = np.zeros((len(pop), self.n_obj))
        for i, func in enumerate(self.objectives):
            obj[:, i] = np.array([func(ind) for ind in pop])
        return obj

    def _non_dominated_sort(self, obj_values):
        """快速非支配排序."""
        n = len(obj_values)
        domination_count = np.zeros(n, dtype=int)
        dominated_set = [[] for _ in range(n)]
        fronts = [[]]

        for i in range(n):
            for j in range(i + 1, n):
                if self._dominates(obj_values[i], obj_values[j]):
                    dominated_set[i].append(j)
                    domination_count[j] += 1
                elif self._dominates(obj_values[j], obj_values[i]):
                    dominated_set[j].append(i)
                    domination_count[i] += 1

            if domination_count[i] == 0:
                fronts[0].append(i)

        k = 0
        while fronts[k]:
            next_front = []
            for i in fronts[k]:
                for j in dominated_set[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            k += 1
            fronts.append(next_front)

        return [f for f in fronts if f]

    def _dominates(self, a, b):
        """判断 a 是否支配 b (全部最小化)."""
        return np.all(a <= b) and np.any(a < b)

    def _crowding_distance(self, obj_values, fronts):
        """计算拥挤度距离."""
        n = len(obj_values)
        distances = np.zeros(n)

        for front in fronts:
            if len(front) <= 2:
                for i in front:
                    distances[i] = float("inf")
                continue

            for m in range(self.n_obj):
                sorted_idx = sorted(front, key=lambda i: obj_values[i, m])
                distances[sorted_idx[0]] = float("inf")
                distances[sorted_idx[-1]] = float("inf")

                obj_range = obj_values[sorted_idx[-1], m] - obj_values[sorted_idx[0], m]
                if obj_range < 1e-10:
                    continue

                for k in range(1, len(sorted_idx) - 1):
                    distances[sorted_idx[k]] += (
                        obj_values[sorted_idx[k+1], m] - obj_values[sorted_idx[k-1], m]
                    ) / obj_range

        return distances

    def _generate_offspring(self, pop, obj_values, fronts, crowding):
        """锦标赛选择 + 模拟二进制交叉 + 多项式变异."""
        n = len(pop)
        offspring = np.zeros_like(pop)

        # Build rank_dict for O(1) rank lookups
        rank_dict = {}
        for rank, front in enumerate(fronts):
            for idx in front:
                rank_dict[idx] = rank

        for i in range(0, n, 2):
            # 锦标赛选择
            p1 = self._tournament_select(crowding, n, rank_dict, len(fronts))
            p2 = self._tournament_select(crowding, n, rank_dict, len(fronts))

            # 交叉 (SBX)
            if np.random.random() < 0.9:
                child1, child2 = self._sbx_crossover(pop[p1], pop[p2])
            else:
                child1, child2 = pop[p1].copy(), pop[p2].copy()

            # 变异
            child1 = self._polynomial_mutation(child1)
            child2 = self._polynomial_mutation(child2)

            offspring[i] = child1
            if i + 1 < n:
                offspring[i + 1] = child2

        return offspring

    def _tournament_select(self, crowding, n, rank_dict, n_fronts):
        """锦标赛选择."""
        a, b = np.random.randint(0, n, 2)
        rank_a = rank_dict.get(a, n_fronts)
        rank_b = rank_dict.get(b, n_fronts)
        if rank_a < rank_b:
            return a
        elif rank_b < rank_a:
            return b
        return a if crowding[a] > crowding[b] else b

    def _sbx_crossover(self, p1, p2, eta=20):
        """模拟二进制交叉."""
        child1, child2 = p1.copy(), p2.copy()
        for i in range(self.n_vars):
            if np.random.random() < 0.5:
                if abs(p1[i] - p2[i]) > 1e-10:
                    u = np.random.random()
                    if u <= 0.5:
                        beta = (2 * u) ** (1 / (eta + 1))
                    else:
                        beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))
                    child1[i] = 0.5 * ((1 + beta) * p1[i] + (1 - beta) * p2[i])
                    child2[i] = 0.5 * ((1 - beta) * p1[i] + (1 + beta) * p2[i])
                    child1[i] = np.clip(child1[i], self.bounds[i, 0], self.bounds[i, 1])
                    child2[i] = np.clip(child2[i], self.bounds[i, 0], self.bounds[i, 1])
        return child1, child2

    def _polynomial_mutation(self, x, eta=20, prob=0.1):
        """多项式变异."""
        for i in range(self.n_vars):
            if np.random.random() < prob:
                delta = (self.bounds[i, 1] - self.bounds[i, 0])
                u = np.random.random()
                if u < 0.5:
                    delta_q = (2 * u) ** (1 / (eta + 1)) - 1
                else:
                    delta_q = 1 - (2 * (1 - u)) ** (1 / (eta + 1))
                x[i] += delta_q * delta
                x[i] = np.clip(x[i], self.bounds[i, 0], self.bounds[i, 1])
        return x

    def plot_pareto(self, figsize=(10, 7)):
        """绘制 Pareto 前沿."""
        import matplotlib.pyplot as plt

        self._ensure_result()
        r = self._result

        if r.pareto_front.shape[1] == 2:
            fig, ax = plt.subplots(figsize=figsize)
            ax.scatter(r.pareto_front[:, 0], r.pareto_front[:, 1],
                       c="red", s=50, zorder=5, label="Pareto 最优解")
            ax.set_xlabel("$f_1$ (最小化)")
            ax.set_ylabel("$f_2$ (最小化)")
            ax.set_title(f"Pareto 前沿 ({r.n_solutions} 个解)")
            ax.legend()
            ax.grid(True, alpha=0.3)
        elif r.pareto_front.shape[1] == 3:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(r.pareto_front[:, 0], r.pareto_front[:, 1], r.pareto_front[:, 2],
                       c="red", s=50)
            ax.set_xlabel("$f_1$")
            ax.set_ylabel("$f_2$")
            ax.set_zlabel("$f_3$")
            ax.set_title("3D Pareto 前沿")
        else:
            raise ValueError("仅支持 2D/3D 可视化")

        plt.tight_layout()
        return fig

    def _ensure_result(self):
        if self._result is None:
            raise RuntimeError("请先调用 run()")

    def summary(self):
        self._ensure_result()
        r = self._result
        lines = [
            "=" * 50,
            "  NSGA-II 多目标优化结果",
            "=" * 50,
            f"  Pareto 解数: {r.n_solutions}",
            f"  迭代代数: {r.n_generations}",
            "-" * 50,
            "  Pareto 前沿范围:",
        ]
        for i in range(r.pareto_front.shape[1]):
            lines.append(f"    f{i+1}: [{r.pareto_front[:, i].min():.4f}, {r.pareto_front[:, i].max():.4f}]")
        lines.append("=" * 50)
        return "\n".join(lines)
