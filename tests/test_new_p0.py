"""
测试 P0 级别新增功能: CellularAutomata, DEA, CorrelationAnalysis, NonlinearProgramming
"""

import numpy as np
import pytest


class TestCellularAutomata:
    """测试元胞自动机."""

    def test_basic_creation(self):
        from mathflow.simulation import CellularAutomata
        ca = CellularAutomata(grid_size=(10, 10), n_states=2)
        assert ca.grid_size == (10, 10)
        assert ca.n_states == 2

    def test_initialize_random(self):
        from mathflow.simulation import CellularAutomata
        ca = CellularAutomata(grid_size=(20, 20))
        ca.initialize(pattern="random", infected_ratio=0.1)
        assert ca.grid.shape == (20, 20)
        # 应该有一些感染的细胞
        assert np.sum(ca.grid == 1) > 0

    def test_initialize_center(self):
        from mathflow.simulation import CellularAutomata
        ca = CellularAutomata(grid_size=(20, 20))
        ca.initialize(pattern="center", radius=3)
        assert np.sum(ca.grid == 1) > 0

    def test_game_of_life(self):
        from mathflow.simulation import CellularAutomata
        ca = CellularAutomata(grid_size=(20, 20))
        ca.add_rule("game_of_life")
        ca.initialize(pattern="random", infected_ratio=0.3)
        result = ca.run(steps=10)
        assert result.steps == 10
        assert len(result.history) == 11  # 初始 + 10步

    def test_sir_model(self):
        from mathflow.simulation import CellularAutomata
        ca = CellularAutomata(grid_size=(20, 20), n_states=3)
        ca.add_rule("SIR", infection_rate=0.3, recovery_rate=0.1)
        ca.initialize(pattern="random", infected_ratio=0.05)
        result = ca.run(steps=20)
        assert result.n_states == 3
        # 应该有状态变化
        assert len(result.stats["state_0"]) == 21

    def test_forest_fire(self):
        from mathflow.simulation import CellularAutomata
        ca = CellularAutomata(grid_size=(30, 30), n_states=3)
        ca.add_rule("forest_fire", grow_rate=0.05, spread_rate=0.3)
        ca.initialize(pattern="random", infected_ratio=0.01)
        result = ca.run(steps=10)
        assert result.steps == 10

    def test_set_grid(self):
        from mathflow.simulation import CellularAutomata
        ca = CellularAutomata(grid_size=(5, 5))
        grid = np.array([
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 0, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ])
        ca.set_grid(grid)
        assert np.array_equal(ca.grid, grid)


class TestDEA:
    """测试数据包络分析."""

    def test_basic_dea(self):
        from mathflow.evaluate import DEA
        # 3个DMU，2个投入，1个产出
        inputs = np.array([[2, 5], [3, 3], [5, 2]])
        outputs = np.array([[1], [1], [1]])
        dea = DEA(inputs, outputs, model="CCR", orientation="input")
        result = dea.fit()
        assert len(result.efficiencies) == 3
        assert all(0 <= e <= 1.01 for e in result.efficiencies)

    def test_efficient_dmu(self):
        from mathflow.evaluate import DEA
        # 一个明显有效的DMU
        inputs = np.array([[1, 1], [2, 2], [3, 3]])
        outputs = np.array([[3], [3], [3]])
        dea = DEA(inputs, outputs)
        result = dea.fit()
        # 第一个DMU投入最少产出相同，应该最有效
        assert result.efficiencies[0] >= result.efficiencies[1]

    def test_output_oriented(self):
        from mathflow.evaluate import DEA
        inputs = np.array([[2, 5], [3, 3], [5, 2]])
        outputs = np.array([[1], [1], [1]])
        dea = DEA(inputs, outputs, orientation="output")
        result = dea.fit()
        assert len(result.efficiencies) == 3

    def test_summary(self):
        from mathflow.evaluate import DEA
        inputs = np.array([[1, 2], [2, 3], [3, 4]])
        outputs = np.array([[3], [2], [1]])
        dea = DEA(inputs, outputs)
        dea.fit()
        s = dea.summary()
        assert "DEA" in s
        assert "效率" in s


class TestCorrelationAnalysis:
    """测试相关性分析."""

    def test_pearson(self):
        from mathflow.stats import CorrelationAnalysis
        np.random.seed(42)
        x = np.random.randn(100)
        X = np.column_stack([x, x + np.random.randn(100) * 0.1, np.random.randn(100)])
        ca = CorrelationAnalysis(X, var_names=["A", "B", "C"])
        result = ca.fit(method="pearson")
        # A和B高度相关
        assert abs(result.corr_matrix[0, 1]) > 0.8
        # A和C不相关
        assert abs(result.corr_matrix[0, 2]) < 0.3

    def test_spearman(self):
        from mathflow.stats import CorrelationAnalysis
        np.random.seed(42)
        X = np.random.randn(50, 3)
        ca = CorrelationAnalysis(X)
        result = ca.fit(method="spearman")
        assert result.corr_matrix.shape == (3, 3)
        # 对角线为1
        assert np.allclose(np.diag(result.corr_matrix), 1.0)

    def test_significant_pairs(self):
        from mathflow.stats import CorrelationAnalysis
        np.random.seed(42)
        x = np.random.randn(100)
        X = np.column_stack([x, x + np.random.randn(100) * 0.01])
        ca = CorrelationAnalysis(X)
        ca.fit()
        pairs = ca.get_significant_pairs(alpha=0.05)
        assert len(pairs) > 0

    def test_summary(self):
        from mathflow.stats import CorrelationAnalysis
        X = np.random.randn(30, 3)
        ca = CorrelationAnalysis(X, var_names=["X1", "X2", "X3"])
        ca.fit()
        s = ca.summary()
        assert "相关" in s


class TestNonlinearProgramming:
    """测试非线性规划."""

    def test_unconstrained(self):
        from mathflow.optimize import NonlinearProgramming
        nlp = NonlinearProgramming()
        nlp.set_objective(lambda x: (x[0] - 2)**2 + (x[1] - 3)**2)
        result = nlp.solve(x0=[0, 0], method="Nelder-Mead")
        assert abs(result.solution[0] - 2) < 0.1
        assert abs(result.solution[1] - 3) < 0.1

    def test_constrained(self):
        from mathflow.optimize import NonlinearProgramming
        nlp = NonlinearProgramming()
        nlp.set_objective(lambda x: x[0]**2 + x[1]**2)
        nlp.add_constraint(lambda x: x[0] + x[1] - 1, type="ineq")
        result = nlp.solve(x0=[0.5, 0.5])
        # 最优解应该在 x + y = 1 上，且 x = y = 0.5
        assert result.success or abs(result.optimal_value - 0.5) < 0.1

    def test_with_bounds(self):
        from mathflow.optimize import NonlinearProgramming
        nlp = NonlinearProgramming()
        nlp.set_objective(lambda x: x[0]**2 + x[1]**2)
        nlp.set_bounds([(0, 1), (0, 1)])
        result = nlp.solve(x0=[0.5, 0.5])
        assert all(0 <= x <= 1 for x in result.solution)

    def test_no_objective_raises(self):
        from mathflow.optimize import NonlinearProgramming
        nlp = NonlinearProgramming()
        with pytest.raises(RuntimeError):
            nlp.solve(x0=[0, 0])

    def test_summary(self):
        from mathflow.optimize import NonlinearProgramming
        nlp = NonlinearProgramming()
        nlp.set_objective(lambda x: x[0]**2 + x[1]**2)
        nlp.solve(x0=[1, 1])
        s = nlp.summary()
        assert "非线性" in s


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
