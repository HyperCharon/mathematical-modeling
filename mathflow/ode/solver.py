"""
ODE 常微分方程数值求解器

支持 Euler、改进 Euler (Heun)、经典 Runge-Kutta (RK4) 方法。
内置经典模型: SIR 传染病、Lotka-Volterra 捕食者- prey、阻尼振动。

Example:
    >>> from mathflow.ode import ODESolver
    >>> # 求解 dy/dt = -2y + 1, y(0) = 0
    >>> solver = ODESolver(lambda t, y: -2*y + 1, y0=0)
    >>> result = solver.solve(t_span=(0, 5), dt=0.01)
    >>> solver.plot()
"""

import numpy as np
from dataclasses import dataclass
from typing import Callable
from scipy.integrate import solve_ivp


@dataclass
class ODEResult:
    """ODE 求解结果."""
    t: np.ndarray
    y: np.ndarray
    method: str
    n_steps: int


class ODESolver:
    """
    ODE 初值问题求解器.

    支持:
    - 单个方程: dy/dt = f(t, y)
    - 方程组: dy/dt = f(t, y), y 是向量

    Parameters
    ----------
    func : callable
        右端函数 f(t, y)
    y0 : float or array-like
        初始条件
    """

    def __init__(self, func: Callable, y0):
        self.func = func
        self.y0 = np.atleast_1d(np.asarray(y0, dtype=float))
        self._result = None

    def __repr__(self) -> str:
        if self._result is not None:
            return f"ODESolver(y0={self.y0}, n_steps={len(self._result.t)})"
        return f"ODESolver(y0={self.y0})"

    def solve(self, t_span=(0, 10), dt=0.01, method="rk4"):
        """
        求解 ODE.

        Parameters
        ----------
        t_span : tuple (t_start, t_end)
            时间区间
        dt : float
            时间步长
        method : str
            "euler", "heun" (改进Euler), "rk4" (经典RK4), "scipy" (调用 scipy)
        """
        t_start, t_end = t_span
        n_steps = int((t_end - t_start) / dt) + 1
        t = np.linspace(t_start, t_end, n_steps)

        if method == "scipy":
            return self._solve_scipy(t_span, t)

        y = np.zeros((n_steps, len(self.y0)))
        y[0] = self.y0

        if method == "euler":
            stepper = self._euler_step
        elif method == "heun":
            stepper = self._heun_step
        elif method == "rk4":
            stepper = self._rk4_step
        else:
            raise ValueError(f"未知方法: {method}")

        for i in range(n_steps - 1):
            y[i + 1] = stepper(t[i], y[i], dt)

        self._result = ODEResult(t=t, y=y, method=method, n_steps=n_steps)
        return self._result

    def _euler_step(self, t, y, dt):
        """Euler 方法."""
        return y + dt * np.atleast_1d(self.func(t, y))

    def _heun_step(self, t, y, dt):
        """改进 Euler (Heun) 方法."""
        k1 = np.atleast_1d(self.func(t, y))
        k2 = np.atleast_1d(self.func(t + dt, y + dt * k1))
        return y + dt / 2 * (k1 + k2)

    def _rk4_step(self, t, y, dt):
        """经典四阶 Runge-Kutta."""
        k1 = np.atleast_1d(self.func(t, y))
        k2 = np.atleast_1d(self.func(t + dt / 2, y + dt / 2 * k1))
        k3 = np.atleast_1d(self.func(t + dt / 2, y + dt / 2 * k2))
        k4 = np.atleast_1d(self.func(t + dt, y + dt * k3))
        return y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    def _solve_scipy(self, t_span, t_eval):
        """调用 scipy 求解 (高精度)."""
        sol = solve_ivp(self.func, t_span, self.y0, t_eval=t_eval,
                        method="RK45", rtol=1e-8, atol=1e-10)
        self._result = ODEResult(
            t=sol.t, y=sol.y.T, method="scipy_RK45", n_steps=len(sol.t)
        )
        return self._result

    @property
    def result(self):
        self._ensure_result()
        return self._result

    def _ensure_result(self):
        if self._result is None:
            raise RuntimeError("请先调用 solve()")

    def plot(self, labels=None, figsize=(10, 6)):
        """绘制解曲线."""
        import matplotlib.pyplot as plt

        self._ensure_result()
        r = self._result
        n_vars = r.y.shape[1] if r.y.ndim > 1 else 1

        fig, ax = plt.subplots(figsize=figsize)
        if n_vars == 1:
            ax.plot(r.t, r.y[:, 0] if r.y.ndim > 1 else r.y, "b-", linewidth=1.5)
            ax.set_ylabel("y")
        else:
            for i in range(n_vars):
                lbl = labels[i] if labels and i < len(labels) else f"y{i+1}"
                ax.plot(r.t, r.y[:, i], linewidth=1.5, label=lbl)
            ax.legend()
            ax.set_ylabel("y")

        ax.set_xlabel("t")
        ax.set_title(f"ODE 数值解 ({r.method})")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    def summary(self):
        self._ensure_result()
        r = self._result
        lines = [
            "=" * 50,
            "  ODE 求解结果",
            "=" * 50,
            f"  方法: {r.method}",
            f"  步数: {r.n_steps}",
            f"  时间: [{r.t[0]:.2f}, {r.t[-1]:.2f}]",
            f"  初始值: {self.y0}",
            f"  终值: {r.y[-1]}",
            "=" * 50,
        ]
        return "\n".join(lines)


# ========== 内置经典模型 ==========

def sir_model(beta=0.3, gamma=0.1):
    """
    SIR 传染病模型.

    dS/dt = -beta * S * I
    dI/dt = beta * S * I - gamma * I
    dR/dt = gamma * I

    Parameters
    ----------
    beta : float
        传染率
    gamma : float
        恢复率
    """
    def func(t, y):
        S, I, R = y
        return [-beta * S * I, beta * S * I - gamma * I, gamma * I]
    return func


def lotka_volterra(alpha=1.5, beta=1.0, delta=0.5, gamma=3.0):
    """
    Lotka-Volterra 捕食者-被捕食者模型.

    dx/dt = alpha*x - beta*x*y  (猎物)
    dy/dt = delta*x*y - gamma*y  (捕食者)
    """
    def func(t, y):
        x, y_ = y
        return [alpha * x - beta * x * y_, delta * x * y_ - gamma * y_]
    return func


def damped_oscillator(omega=2.0, zeta=0.1):
    """
    阻尼振动模型: y'' + 2*zeta*omega*y' + omega^2*y = 0

    转化为一阶方程组:
    y1' = y2
    y2' = -2*zeta*omega*y2 - omega^2*y1
    """
    def func(t, y):
        y1, y2 = y
        return [y2, -2 * zeta * omega * y2 - omega**2 * y1]
    return func


def heat_equation_1d(T_left=100, T_right=25, T_init=None, alpha=0.01, L=1.0, nx=50):
    """
    一维热传导方程 (有限差分法).

    ∂T/∂t = α * ∂²T/∂x²

    Parameters
    ----------
    T_left, T_right : float
        左右边界温度
    alpha : float
        热扩散系数
    L : float
        杆长
    nx : int
        空间网格数
    """
    dx = L / (nx - 1)
    x = np.linspace(0, L, nx)

    if T_init is None:
        # 默认: 线性初始分布
        T_init = T_left + (T_right - T_left) * x / L

    def func(t, T):
        dTdt = np.zeros_like(T)
        # 内部节点: 中心差分 (向量化)
        dTdt[1:-1] = alpha * (T[2:] - 2*T[1:-1] + T[:-2]) / dx**2
        # 边界条件
        dTdt[0] = 0  # 左边界恒温
        dTdt[-1] = 0  # 右边界恒温
        return dTdt

    return func, T_init, x


def wave_equation_1d(c=1.0, L=1.0, nx=100):
    """
    一维波动方程 (转化为一阶方程组).

    ∂²u/∂t² = c² * ∂²u/∂x²

    令 v = ∂u/∂t, 则:
    ∂u/∂t = v
    ∂v/∂t = c² * ∂²u/∂x²
    """
    dx = L / (nx - 1)
    x = np.linspace(0, L, nx)

    # 初始条件: 钟形脉冲
    u0 = np.exp(-200 * (x - L/2)**2)
    v0 = np.zeros(nx)
    y0 = np.concatenate([u0, v0])

    def func(t, y):
        u = y[:nx]
        v = y[nx:]
        dudt = v
        dvdt = np.zeros(nx)
        # 向量化计算空间二阶导数
        dvdt[1:-1] = c**2 * (u[2:] - 2*u[1:-1] + u[:-2]) / dx**2
        return np.concatenate([dudt, dvdt])

    return func, y0, x
