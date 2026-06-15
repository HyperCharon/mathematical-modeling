"""优化类模型测试."""

import numpy as np
import pytest
from mathflow.optimize import LinearProgramming, GeneticAlgorithm, PSO, SimulatedAnnealing


class TestLinearProgramming:
    def test_basic(self):
        lp = LinearProgramming()
        lp.set_objective([4, 3], sense="max")
        lp.add_constraint([2, 1], "<=", 10)
        lp.add_constraint([1, 1], "<=", 8)
        result = lp.solve()
        assert result.status == "Optimal"
        assert result.optimal_value > 0
        assert all(v >= 0 for v in result.solution)

    def test_minimization(self):
        lp = LinearProgramming()
        lp.set_objective([1, 2], sense="min")
        lp.add_constraint([1, 0], ">=", 1)
        lp.add_constraint([0, 1], ">=", 1)
        result = lp.solve()
        assert result.optimal_value == 3.0


class TestGeneticAlgorithm:
    def test_simple_optimization(self):
        def fitness(x):
            return -(x[0] - 3)**2 + 9

        ga = GeneticAlgorithm(fitness, n_vars=1, bounds=[(0, 6)],
                              pop_size=50, generations=100)
        result = ga.run()
        assert abs(result.best_solution[0] - 3) < 0.5
        assert result.best_fitness > 8.0

    def test_convergence(self):
        def sphere(x):
            return -sum(xi**2 for xi in x)

        ga = GeneticAlgorithm(sphere, n_vars=3, bounds=[(-5, 5)] * 3,
                              pop_size=100, generations=200)
        result = ga.run()
        assert result.best_fitness > -1.0


class TestPSO:
    def test_basic(self):
        def sphere(x):
            return -sum(xi**2 for xi in x)

        pso = PSO(sphere, n_vars=3, bounds=[(-5, 5)] * 3)
        result = pso.run()
        assert result.best_fitness > -1.0


class TestSimulatedAnnealing:
    def test_basic(self):
        def f(x):
            return (x[0] - 3)**2 + (x[1] + 2)**2

        sa = SimulatedAnnealing(f, n_vars=2, bounds=[(-10, 10)] * 2)
        result = sa.run()
        assert result.best_energy < 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
