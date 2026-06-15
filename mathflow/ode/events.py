"""
事件驱动ODE求解器

支持在ODE求解过程中检测事件 (如压力达到阈值、穿过零点等)。

Example:
    >>> from mathflow.ode.events import EventODESolver
    >>> def dydt(t, y): return [-y[0]]
    >>> def event_zero(t, y): return y[0] - 0.5  # y=0.5 时触发
    >>> solver = EventODESolver(dydt, y0=[1.0], events=[event_zero])
    >>> result = solver.solve(t_span=(0, 5))
"""

import numpy as np
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple


@dataclass
class EventODEResult:
    """事件驱动ODE结果."""
    t: np.ndarray
    y: np.ndarray
    event_times: List[float]      # 事件触发时间
    event_values: List[np.ndarray] # 事件触发时的状态
    event_indices: List[int]       # 事件触发的步索引
    method: str


class EventODESolver:
    """
    事件驱动ODE求解器.

    Parameters
    ----------
    func : callable
        右端函数 f(t, y) -> dy/dt
    y0 : array-like
        初始条件
    events : list of callable
        事件函数 g(t, y), 当 g(t,y)=0 时触发事件
    event_types : list of str, optional
        事件类型: "terminal" (终止求解), "record" (仅记录)
    """

    def __init__(self, func: Callable, y0, events: List[Callable] = None,
                 event_types: List[str] = None):
        self.func = func
        self.y0 = np.atleast_1d(np.asarray(y0, dtype=float))
        self.events = events or []
        self.event_types = event_types or ["record"] * len(self.events)
        self._result = None

    def solve(self, t_span=(0, 10), dt=0.01, method="rk4") -> EventODEResult:
        """求解ODE并检测事件."""
        t_start, t_end = t_span
        n_steps = int((t_end - t_start) / dt) + 1
        t = np.linspace(t_start, t_end, n_steps)

        y = np.zeros((n_steps, len(self.y0)))
        y[0] = self.y0

        event_times = []
        event_values = []
        event_indices = []
        terminated = False

        for i in range(n_steps - 1):
            if terminated:
                break

            # RK4 步进
            y[i + 1] = self._rk4_step(t[i], y[i], dt)

            # 检测事件
            for j, event_func in enumerate(self.events):
                val_prev = event_func(t[i], y[i])
                val_next = event_func(t[i + 1], y[i + 1])

                # 符号变化检测 (零点穿越)
                if val_prev * val_next < 0:
                    # 线性插值找精确事件时间
                    alpha = -val_prev / (val_next - val_prev)
                    t_event = t[i] + alpha * dt
                    y_event = y[i] + alpha * (y[i + 1] - y[i])

                    event_times.append(t_event)
                    event_values.append(y_event.copy())
                    event_indices.append(i + 1)

                    if self.event_types[j] == "terminal":
                        terminated = True
                        # 截断到事件点
                        t = t[:i + 2]
                        y = y[:i + 2]
                        y[-1] = y_event
                        break

        self._result = EventODEResult(
            t=t, y=y,
            event_times=event_times,
            event_values=event_values,
            event_indices=event_indices,
            method=method,
        )
        return self._result

    def _rk4_step(self, t, y, dt):
        """经典RK4步进."""
        k1 = np.atleast_1d(self.func(t, y))
        k2 = np.atleast_1d(self.func(t + dt / 2, y + dt / 2 * k1))
        k3 = np.atleast_1d(self.func(t + dt / 2, y + dt / 2 * k2))
        k4 = np.atleast_1d(self.func(t + dt, y + dt * k3))
        return y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    @property
    def result(self):
        if self._result is None:
            raise RuntimeError("请先调用 solve()")
        return self._result

    def plot(self, labels=None, figsize=(10, 6)):
        """绘制结果 (标记事件点)."""
        import matplotlib.pyplot as plt

        r = self.result
        fig, ax = plt.subplots(figsize=figsize)

        n_vars = r.y.shape[1] if r.y.ndim > 1 else 1
        if n_vars == 1:
            ax.plot(r.t, r.y[:, 0] if r.y.ndim > 1 else r.y, "b-", linewidth=1.5)
        else:
            for i in range(n_vars):
                lbl = labels[i] if labels else f"y{i+1}"
                ax.plot(r.t, r.y[:, i], linewidth=1.5, label=lbl)
            ax.legend()

        # 标记事件点
        for t_ev, y_ev in zip(r.event_times, r.event_values):
            ax.plot(t_ev, y_ev[0] if len(y_ev) > 0 else y_ev, "ro", markersize=8, zorder=5)
            ax.axvline(x=t_ev, color="red", linestyle="--", alpha=0.3)

        ax.set_xlabel("t")
        ax.set_ylabel("y")
        ax.set_title(f"事件驱动ODE求解 ({len(r.event_times)} 个事件)")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    def summary(self):
        r = self.result
        lines = [
            "=" * 50,
            "  事件驱动ODE求解结果",
            "=" * 50,
            f"  时间范围: [{r.t[0]:.4f}, {r.t[-1]:.4f}]",
            f"  步数: {len(r.t)}",
            f"  事件数: {len(r.event_times)}",
        ]
        for i, (t_ev, y_ev) in enumerate(zip(r.event_times, r.event_values)):
            lines.append(f"  事件{i+1}: t={t_ev:.4f}, y={y_ev}")
        lines.append("=" * 50)
        return "\n".join(lines)
