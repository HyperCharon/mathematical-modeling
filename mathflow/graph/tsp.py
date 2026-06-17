"""
旅行商问题 (TSP) 求解器

支持精确算法 (DP, 适用于小规模) 和近似算法 (遗传算法/模拟退火, 适用于大规模)。

Example:
    >>> from mathflow.graph import TSPSolver
    >>> import numpy as np
    >>> coords = np.array([[0,0],[1,5],[5,2],[6,6],[8,3]])
    >>> tsp = TSPSolver(coords)
    >>> result = tsp.solve(method="brute_force")
"""

import numpy as np
from itertools import permutations
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class TSPResult:
    """TSP 结果."""
    route: List[int]
    total_distance: float
    method: str


class TSPSolver:
    """
    TSP 旅行商问题求解器.

    Parameters
    ----------
    coords : array-like, shape (n, 2)
        城市坐标
    dist_matrix : array-like, shape (n, n), optional
        距离矩阵 (与 coords 二选一)
    """

    def __init__(self, coords=None, dist_matrix=None):
        if dist_matrix is not None:
            self.dist_matrix = np.asarray(dist_matrix, dtype=float)
            self.n = self.dist_matrix.shape[0]
            self.coords = None
        elif coords is not None:
            self.coords = np.asarray(coords, dtype=float)
            self.n = len(self.coords)
            self.dist_matrix = self._calc_distance_matrix()
        else:
            raise ValueError("需要提供 coords 或 dist_matrix")

    def __repr__(self) -> str:
        return f"TSPSolver(n_cities={self.n})"

    def _calc_distance_matrix(self):
        """计算欧氏距离矩阵."""
        n = self.n
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = np.sqrt(((self.coords[i] - self.coords[j]) ** 2).sum())
                D[i, j] = D[j, i] = d
        return D

    def solve(self, method="nearest_neighbor"):
        """
        求解 TSP.

        Parameters
        ----------
        method : str
            "brute_force" (暴力, n<=10), "nearest_neighbor" (最近邻),
            "two_opt" (2-opt 局部搜索), "genetic" (遗传算法)
        """
        if method == "brute_force":
            return self._brute_force()
        elif method == "nearest_neighbor":
            return self._nearest_neighbor()
        elif method == "two_opt":
            return self._two_opt()
        elif method == "genetic":
            return self._genetic()
        else:
            raise ValueError(f"未知方法: {method}")

    def _brute_force(self):
        """暴力枚举 (n <= 10)."""
        if self.n > 10:
            raise ValueError("暴力法仅适用于 n <= 10")

        best_dist = float("inf")
        best_route = None

        for perm in permutations(range(1, self.n)):
            route = [0] + list(perm) + [0]
            d = sum(self.dist_matrix[route[i]][route[i + 1]] for i in range(self.n))
            if d < best_dist:
                best_dist = d
                best_route = route

        return TSPResult(route=best_route, total_distance=best_dist, method="brute_force")

    def _nearest_neighbor(self):
        """最近邻启发式."""
        best_dist = float("inf")
        best_route = None

        for start in range(self.n):
            visited = {start}
            route = [start]
            total = 0
            current = start

            while len(visited) < self.n:
                nearest = None
                nearest_dist = float("inf")
                for j in range(self.n):
                    if j not in visited and self.dist_matrix[current][j] < nearest_dist:
                        nearest = j
                        nearest_dist = self.dist_matrix[current][j]
                visited.add(nearest)
                route.append(nearest)
                total += nearest_dist
                current = nearest

            route.append(start)
            total += self.dist_matrix[current][start]

            if total < best_dist:
                best_dist = total
                best_route = route

        return TSPResult(route=best_route, total_distance=best_dist, method="nearest_neighbor")

    def _two_opt(self):
        """2-opt 局部搜索 (以最近邻解为初始解)."""
        init = self._nearest_neighbor()
        route = init.route[:-1]  # 去掉末尾的起点
        n = len(route)
        improved = True

        while improved:
            improved = False
            for i in range(1, n - 1):
                for j in range(i + 1, n):
                    # 计算 2-opt 交换后的距离变化
                    d1 = self.dist_matrix[route[i - 1]][route[i]] + self.dist_matrix[route[j]][route[(j + 1) % n]]
                    d2 = self.dist_matrix[route[i - 1]][route[j]] + self.dist_matrix[route[i]][route[(j + 1) % n]]
                    if d2 < d1:
                        route[i:j + 1] = reversed(route[i:j + 1])
                        improved = True

        route.append(route[0])
        total = sum(self.dist_matrix[route[i]][route[i + 1]] for i in range(n))
        return TSPResult(route=route, total_distance=total, method="2-opt")

    def _genetic(self):
        """遗传算法求解 TSP."""
        pop_size = min(100, max(50, self.n * 10))
        generations = 500

        # 初始化种群
        population = []
        for _ in range(pop_size):
            perm = np.random.permutation(self.n)
            population.append(perm.tolist())

        def route_distance(route):
            return sum(self.dist_matrix[route[i]][route[(i + 1) % self.n]] for i in range(self.n))

        def fitness(route):
            return 1.0 / (1.0 + route_distance(route))

        best_route = min(population, key=route_distance)
        best_dist = route_distance(best_route)

        for gen in range(generations):
            # 选择 + 交叉 (OX 顺序交叉)
            new_pop = []
            # 精英保留
            sorted_pop = sorted(population, key=route_distance)
            new_pop.extend(sorted_pop[:2])

            while len(new_pop) < pop_size:
                # 锦标赛选择
                p1 = min(np.random.choice(len(population), 3, replace=False), key=lambda i: route_distance(population[i]))
                p2 = min(np.random.choice(len(population), 3, replace=False), key=lambda i: route_distance(population[i]))
                parent1, parent2 = population[p1], population[p2]

                # OX 交叉
                if np.random.random() < 0.8:
                    start, end = sorted(np.random.choice(self.n, 2, replace=False))
                    child = [-1] * self.n
                    child[start:end + 1] = parent1[start:end + 1]
                    fill = [x for x in parent2 if x not in child[start:end + 1]]
                    idx = 0
                    for i in range(self.n):
                        if child[i] == -1:
                            child[i] = fill[idx]
                            idx += 1
                else:
                    child = parent1[:]

                # 变异 (swap)
                if np.random.random() < 0.2:
                    i, j = np.random.choice(self.n, 2, replace=False)
                    child[i], child[j] = child[j], child[i]

                new_pop.append(child)

            population = new_pop
            gen_best = min(population, key=route_distance)
            gen_best_dist = route_distance(gen_best)
            if gen_best_dist < best_dist:
                best_route = gen_best[:]
                best_dist = gen_best_dist

        best_route.append(best_route[0])
        return TSPResult(route=best_route, total_distance=best_dist, method="genetic")

    def plot(self, result: TSPResult = None, figsize=(8, 8)):
        """绘制 TSP 路径."""
        import matplotlib.pyplot as plt

        if self.coords is None:
            raise ValueError("需要坐标才能绘图")

        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(self.coords[:, 0], self.coords[:, 1], c="red", s=100, zorder=5)

        for i, (x, y) in enumerate(self.coords):
            ax.annotate(f"  {i}", (x, y), fontsize=12, fontweight="bold")

        if result:
            route = result.route
            for i in range(len(route) - 1):
                ax.annotate("",
                            xy=self.coords[route[i + 1]],
                            xytext=self.coords[route[i]],
                            arrowprops=dict(arrowstyle="->", color="blue", lw=1.5))
            ax.set_title(f"TSP 路径 ({result.method})\n总距离: {result.total_distance:.2f}")
        else:
            ax.set_title("TSP 城市分布")

        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig
