"""ODE 模块测试."""

import numpy as np
import pytest
from mathflow.ode import ODESolver, sir_model, damped_oscillator, EventODESolver


class TestODESolver:
    def test_simple_decay(self):
        """dy/dt = -y, y(0)=1 → y=e^(-t)."""
        solver = ODESolver(lambda t, y: -y, y0=1.0)
        result = solver.solve(t_span=(0, 5), dt=0.01)
        expected = np.exp(-result.t)
        np.testing.assert_allclose(result.y[:, 0], expected, atol=0.01)

    def test_rk4_accuracy(self):
        """RK4 应该比 Euler 精确."""
        exact = np.exp(-1)  # e^(-1)

        solver_euler = ODESolver(lambda t, y: -y, y0=1.0)
        r_euler = solver_euler.solve(t_span=(0, 1), dt=0.01, method="euler")

        solver_rk4 = ODESolver(lambda t, y: -y, y0=1.0)
        r_rk4 = solver_rk4.solve(t_span=(0, 1), dt=0.01, method="rk4")

        err_euler = abs(r_euler.y[-1, 0] - exact)
        err_rk4 = abs(r_rk4.y[-1, 0] - exact)
        assert err_rk4 < err_euler  # RK4 更精确

    def test_system(self):
        """测试方程组."""
        def harmonic(t, y):
            return [y[1], -y[0]]  # y'' + y = 0

        solver = ODESolver(harmonic, y0=[1.0, 0.0])
        result = solver.solve(t_span=(0, 2 * np.pi), dt=0.01)
        # y(2π) ≈ 1, y'(2π) ≈ 0
        np.testing.assert_allclose(result.y[-1, 0], 1.0, atol=0.01)
        np.testing.assert_allclose(result.y[-1, 1], 0.0, atol=0.01)


class TestSIRModel:
    def test_conservation(self):
        """S + I + R = 1 应守恒."""
        func = sir_model(beta=0.3, gamma=0.1)
        solver = ODESolver(func, y0=[0.99, 0.01, 0])
        result = solver.solve(t_span=(0, 200), dt=0.1)
        total = result.y.sum(axis=1)
        np.testing.assert_allclose(total, 1.0, atol=1e-6)

    def test_epidemic_peak(self):
        """感染人数应先增后减."""
        func = sir_model(beta=0.5, gamma=0.1)
        solver = ODESolver(func, y0=[0.99, 0.01, 0])
        result = solver.solve(t_span=(0, 200), dt=0.1)
        I = result.y[:, 1]
        peak = np.argmax(I)
        assert peak > 0  # 有峰值
        assert I[peak] > I[0]  # 峰值大于初始值


class TestDampedOscillator:
    def test_decay(self):
        """阻尼振动振幅应衰减."""
        func = damped_oscillator(omega=5.0, zeta=0.2)
        solver = ODESolver(func, y0=[1.0, 0.0])
        result = solver.solve(t_span=(0, 10), dt=0.001)
        # 前半段最大振幅应大于后半段
        mid = len(result.y) // 2
        first_half_max = np.max(np.abs(result.y[:mid, 0]))
        second_half_max = np.max(np.abs(result.y[mid:, 0]))
        assert first_half_max > second_half_max


class TestEventODESolver:
    def test_basic(self):
        def f(t, y): return [-y[0]]
        def event_zero(t, y): return y[0] - 0.5
        solver = EventODESolver(f, y0=[1.0], events=[event_zero])
        result = solver.solve(t_span=(0, 5), dt=0.01)
        assert len(result.t) > 0

    def test_repr(self):
        solver = EventODESolver(lambda t, y: [-y[0]], y0=[1.0])
        assert "EventODESolver" in repr(solver)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
