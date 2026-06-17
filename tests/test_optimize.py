"""优化类模型测试."""

import numpy as np
import pytest
from mathflow.optimize import LinearProgramming, GeneticAlgorithm, PSO, SimulatedAnnealing, NSGA2, IntegerProgramming


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


class TestNSGA2:
    def test_basic(self):
        def f1(x): return x[0]**2
        def f2(x): return (x[0] - 1)**2
        nsga = NSGA2([f1, f2], n_vars=1, bounds=[(-5, 5)], pop_size=20, generations=10)
        result = nsga.run()
        assert len(result.pareto_front) > 0

    def test_repr(self):
        nsga = NSGA2([lambda x: x[0]], n_vars=1, bounds=[(0, 1)])
        assert "NSGA2" in repr(nsga)


class TestIntegerProgramming:
    def test_basic(self):
        ip = IntegerProgramming()
        ip.set_objective([1, 2], sense="max")
        ip.add_constraint([1, 1], "<=", 10)
        ip.add_constraint([1, 0], "<=", 6)
        result = ip.solve(integer_vars=[0, 1])
        assert result.status in ["Optimal", "Feasible"]

    def test_repr(self):
        ip = IntegerProgramming()
        assert "IntegerProgramming" in repr(ip)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
