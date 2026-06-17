"""
元胞自动机 (Cellular Automata)

支持一维和二维元胞自动机，可自定义规则，用于模拟传染病扩散、
交通流、森林火灾等空间动态过程。

Example:
    >>> from mathflow.simulation import CellularAutomata
    >>> # 2D 元胞自动机模拟传染病扩散
    >>> ca = CellularAutomata(grid_size=(50, 50), n_states=3)
    >>> ca.add_rule("SIR", infection_rate=0.3, recovery_rate=0.1)
    >>> ca.initialize(infected_ratio=0.05)
    >>> history = ca.run(steps=100)
    >>> ca.plot()
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Dict


@dataclass
class CAResult:
    """元胞自动机结果."""
    history: List[np.ndarray]       # 每步的网格状态
    stats: Dict[str, List[int]]     # 各状态的计数统计
    steps: int
    grid_size: Tuple[int, int]
    n_states: int


class CellularAutomata:
    """
    二维元胞自动机.

    Parameters
    ----------
    grid_size : tuple (rows, cols)
        网格大小
    n_states : int
        状态数量 (如 2=生/死, 3=S/I/R)
    boundary : str
        边界条件: "wrap" (周期), "fixed" (固定), "reflect" (反射)
    """

    def __init__(self, grid_size: Tuple[int, int] = (50, 50),
                 n_states: int = 2, boundary: str = "wrap"):
        self.grid_size = grid_size
        self.n_states = n_states
        self.boundary = boundary
        self.grid = np.zeros(grid_size, dtype=int)
        self.rules = {}
        self._custom_rule = None
        self._result = None

    def __repr__(self) -> str:
        return f"CellularAutomata(grid={self.grid_size}, states={self.n_states}, boundary={self.boundary!r})"

    def _ensure_result(self) -> None:
        if self._result is None:
            raise RuntimeError("请先调用 run()")

    def initialize(self, pattern: str = "random", **kwargs):
        """
        初始化网格.

        Parameters
        ----------
        pattern : str
            "random" (随机), "center" (中心), "custom" (自定义)
        **kwargs :
            random: infected_ratio=0.05
            center: radius=5
        """
        rows, cols = self.grid_size
        self.grid = np.zeros(self.grid_size, dtype=int)

        if pattern == "random":
            ratio = kwargs.get("infected_ratio", 0.05)
            mask = np.random.random(self.grid_size) < ratio
            self.grid[mask] = 1  # 感染状态

        elif pattern == "center":
            radius = kwargs.get("radius", 5)
            center_r, center_c = rows // 2, cols // 2
            for i in range(rows):
                for j in range(cols):
                    if (i - center_r)**2 + (j - center_c)**2 <= radius**2:
                        self.grid[i, j] = 1

        elif pattern == "glider":
            # Game of Life 滑翔机
            self.grid = np.zeros(self.grid_size, dtype=int)
            r, c = rows // 2, cols // 2
            self.grid[r, c+1] = 1
            self.grid[r+1, c+2] = 1
            self.grid[r+2, c] = 1
            self.grid[r+2, c+1] = 1
            self.grid[r+2, c+2] = 1

        return self

    def set_grid(self, grid: np.ndarray):
        """直接设置网格状态."""
        self.grid = np.asarray(grid, dtype=int)
        return self

    def add_rule(self, rule_type: str = "custom", **kwargs):
        """
        添加规则.

        Parameters
        ----------
        rule_type : str
            "game_of_life" - Conway 生命游戏
            "SIR" - 传染病模型
            "SIS" - SIS 模型
            "forest_fire" - 森林火灾
            "custom" - 自定义规则
        """
        if rule_type == "game_of_life":
            self.n_states = 2
            self.rules["type"] = "game_of_life"

        elif rule_type == "SIR":
            self.n_states = 3  # 0=S, 1=I, 2=R
            self.rules["type"] = "SIR"
            self.rules["infection_rate"] = kwargs.get("infection_rate", 0.3)
            self.rules["recovery_rate"] = kwargs.get("recovery_rate", 0.1)

        elif rule_type == "SIS":
            self.n_states = 2  # 0=S, 1=I
            self.rules["type"] = "SIS"
            self.rules["infection_rate"] = kwargs.get("infection_rate", 0.3)
            self.rules["recovery_rate"] = kwargs.get("recovery_rate", 0.1)

        elif rule_type == "forest_fire":
            self.n_states = 3  # 0=空地, 1=树木, 2=燃烧
            self.rules["type"] = "forest_fire"
            self.rules["grow_rate"] = kwargs.get("grow_rate", 0.01)
            self.rules["spread_rate"] = kwargs.get("spread_rate", 0.3)

        elif rule_type == "custom":
            self.rules["type"] = "custom"
            self._custom_rule = kwargs.get("rule_func")

        return self

    def _get_neighbors(self, i: int, j: int) -> np.ndarray:
        """获取邻域状态 (Moore 邻域)."""
        rows, cols = self.grid_size
        neighbors = []

        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                ni, nj = i + di, j + dj

                if self.boundary == "wrap":
                    ni = ni % rows
                    nj = nj % cols
                elif self.boundary == "fixed":
                    if ni < 0 or ni >= rows or nj < 0 or nj >= cols:
                        continue
                elif self.boundary == "reflect":
                    if ni < 0:
                        ni = -ni
                    if nj < 0:
                        nj = -nj
                    if ni >= rows:
                        ni = 2 * rows - ni - 2
                    if nj >= cols:
                        nj = 2 * cols - nj - 2

                neighbors.append(self.grid[ni, nj])

        return np.array(neighbors)

    def _step_game_of_life(self) -> np.ndarray:
        """Game of Life 单步演化."""
        rows, cols = self.grid_size
        new_grid = self.grid.copy()

        for i in range(rows):
            for j in range(cols):
                neighbors = self._get_neighbors(i, j)
                n_alive = np.sum(neighbors == 1)

                if self.grid[i, j] == 1:  # 活细胞
                    if n_alive < 2 or n_alive > 3:
                        new_grid[i, j] = 0  # 死亡
                else:  # 死细胞
                    if n_alive == 3:
                        new_grid[i, j] = 1  # 复活

        return new_grid

    def _step_SIR(self) -> np.ndarray:
        """SIR 传染病模型单步演化."""
        rows, cols = self.grid_size
        new_grid = self.grid.copy()
        beta = self.rules["infection_rate"]
        gamma = self.rules["recovery_rate"]

        for i in range(rows):
            for j in range(cols):
                if self.grid[i, j] == 0:  # 易感者
                    neighbors = self._get_neighbors(i, j)
                    n_infected = np.sum(neighbors == 1)
                    if n_infected > 0 and np.random.random() < 1 - (1 - beta)**n_infected:
                        new_grid[i, j] = 1  # 感染

                elif self.grid[i, j] == 1:  # 感染者
                    if np.random.random() < gamma:
                        new_grid[i, j] = 2  # 恢复

        return new_grid

    def _step_SIS(self) -> np.ndarray:
        """SIS 模型单步演化."""
        rows, cols = self.grid_size
        new_grid = self.grid.copy()
        beta = self.rules["infection_rate"]
        gamma = self.rules["recovery_rate"]

        for i in range(rows):
            for j in range(cols):
                if self.grid[i, j] == 0:  # 易感者
                    neighbors = self._get_neighbors(i, j)
                    n_infected = np.sum(neighbors == 1)
                    if n_infected > 0 and np.random.random() < 1 - (1 - beta)**n_infected:
                        new_grid[i, j] = 1

                elif self.grid[i, j] == 1:  # 感染者
                    if np.random.random() < gamma:
                        new_grid[i, j] = 0  # 恢复为易感者

        return new_grid

    def _step_forest_fire(self) -> np.ndarray:
        """森林火灾模型单步演化."""
        rows, cols = self.grid_size
        new_grid = self.grid.copy()
        p = self.rules["grow_rate"]
        f = self.rules["spread_rate"]

        for i in range(rows):
            for j in range(cols):
                if self.grid[i, j] == 0:  # 空地
                    if np.random.random() < p:
                        new_grid[i, j] = 1  # 长树

                elif self.grid[i, j] == 1:  # 树木
                    neighbors = self._get_neighbors(i, j)
                    if np.any(neighbors == 2):  # 邻居着火
                        if np.random.random() < f:
                            new_grid[i, j] = 2  # 着火
                    # 自发着火 (闪电)
                    if np.random.random() < 0.0001:
                        new_grid[i, j] = 2

                elif self.grid[i, j] == 2:  # 燃烧
                    new_grid[i, j] = 0  # 变为空地

        return new_grid

    def _step(self) -> np.ndarray:
        """单步演化."""
        rule_type = self.rules.get("type", "custom")

        if rule_type == "game_of_life":
            return self._step_game_of_life()
        elif rule_type == "SIR":
            return self._step_SIR()
        elif rule_type == "SIS":
            return self._step_SIS()
        elif rule_type == "forest_fire":
            return self._step_forest_fire()
        elif rule_type == "custom" and self._custom_rule:
            return self._custom_rule(self.grid, self._get_neighbors)
        else:
            raise ValueError(f"未知规则类型: {rule_type}")

    def run(self, steps: int = 100, verbose: bool = False) -> CAResult:
        """
        运行模拟.

        Parameters
        ----------
        steps : int
            模拟步数
        verbose : bool
            是否打印进度
        """
        history = [self.grid.copy()]
        stats = {f"state_{i}": [np.sum(self.grid == i)] for i in range(self.n_states)}

        for t in range(steps):
            self.grid = self._step()
            history.append(self.grid.copy())

            for i in range(self.n_states):
                stats[f"state_{i}"].append(np.sum(self.grid == i))

            if verbose and (t + 1) % 10 == 0:
                print(f"Step {t+1}/{steps}")

        self._result = CAResult(
            history=history,
            stats=stats,
            steps=steps,
            grid_size=self.grid_size,
            n_states=self.n_states,
        )
        return self._result

    def plot(self, figsize: Tuple[int, int] = (12, 5)):
        """绘制结果."""
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        from mathflow.viz.style import set_paper_style

        set_paper_style()

        if self._result is None:
            raise RuntimeError("请先调用 run()")

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # 左图: 最终状态
        ax = axes[0]
        colors = ["white", "red", "green", "blue", "orange", "purple"]
        cmap = ListedColormap(colors[:self.n_states])
        ax.imshow(self._result.history[-1], cmap=cmap, interpolation="nearest")
        ax.set_title("Final State")
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")

        # 右图: 各状态数量变化
        ax = axes[1]
        state_names = {0: "S (Susceptible)", 1: "I (Infected)", 2: "R (Recovered)"}
        for i in range(self.n_states):
            name = state_names.get(i, f"State {i}")
            ax.plot(self._result.stats[f"state_{i}"], label=name)
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Count")
        ax.set_title("State Evolution")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def summary(self) -> str:
        """打印摘要."""
        if self._result is None:
            raise RuntimeError("请先调用 run()")

        r = self._result
        lines = [
            "=" * 50,
            "  元胞自动机模拟结果",
            "=" * 50,
            f"  网格大小: {r.grid_size}",
            f"  状态数: {r.n_states}",
            f"  模拟步数: {r.steps}",
            "-" * 50,
            "  最终各状态数量:",
        ]
        for i in range(r.n_states):
            count = r.stats[f"state_{i}"][-1]
            total = r.grid_size[0] * r.grid_size[1]
            lines.append(f"    状态 {i}: {count} ({count/total*100:.1f}%)")

        lines.append("=" * 50)
        return "\n".join(lines)
