"""
蒙特卡洛模拟 (Monte Carlo Simulation)

通过大量随机抽样近似求解问题。
适用于积分计算、概率估计、风险分析等。

Example:
    >>> from mathflow.simulation import MonteCarlo
    >>> mc = MonteCarlo()
    >>> # 估算 π
    >>> pi_est, error = mc.estimate_pi(n_samples=100000)
    >>> print(f"π ≈ {pi_est:.4f} (误差: {error:.4f})")
    >>> # 蒙特卡洛积分
    >>> result = mc.integrate(lambda x: x**2, 0, 1, n_samples=100000)
"""

import numpy as np
from dataclasses import dataclass
from typing import Callable, List


@dataclass
class MCResult:
    """蒙特卡洛结果."""
    estimate: float
    std_error: float
    confidence_interval: tuple
    n_samples: int


class MonteCarlo:
    """蒙特卡洛模拟器."""

    def __init__(self, seed=42):
        self.seed = seed

    def __repr__(self) -> str:
        return f"MonteCarlo(seed={self.seed})"

    def estimate_pi(self, n_samples=100000):
        """用蒙特卡洛方法估算 π."""
        np.random.seed(self.seed)
        x = np.random.uniform(-1, 1, n_samples)
        y = np.random.uniform(-1, 1, n_samples)
        inside = (x**2 + y**2) <= 1
        pi_est = 4 * inside.mean()
        error = abs(pi_est - np.pi)
        return pi_est, error

    def integrate(self, func: Callable, a: float, b: float, n_samples=100000):
        """
        蒙特卡洛积分 ∫_a^b f(x) dx.

        Parameters
        ----------
        func : callable
            被积函数
        a, b : float
            积分上下限
        n_samples : int
            抽样次数
        """
        np.random.seed(self.seed)
        x = np.random.uniform(a, b, n_samples)
        # 向量化计算：如果func支持数组输入则使用向量化，否则回退到循环
        try:
            y = func(x)
            if np.isscalar(y):
                raise ValueError
        except (ValueError, TypeError):
            y = np.array([func(xi) for xi in x])
        estimate = (b - a) * y.mean()
        std_error = (b - a) * y.std() / np.sqrt(n_samples)
        ci = (estimate - 1.96 * std_error, estimate + 1.96 * std_error)
        return MCResult(estimate=estimate, std_error=std_error,
                        confidence_interval=ci, n_samples=n_samples)

    def simulate_risk(self, model_func: Callable, n_samples=10000,
                      distributions=None):
        """
        风险分析模拟.

        Parameters
        ----------
        model_func : callable
            模型函数，输入随机变量，输出结果
        distributions : list of callable
            各随机变量的分布函数 (返回抽样值)
        """
        if distributions is None:
            raise ValueError("需要提供 distributions 参数")
        np.random.seed(self.seed)
        results = []
        for _ in range(n_samples):
            inputs = [dist() for dist in distributions]
            results.append(model_func(*inputs))
        results = np.array(results)

        return MCResult(
            estimate=results.mean(),
            std_error=results.std() / np.sqrt(n_samples),
            confidence_interval=(np.percentile(results, 2.5), np.percentile(results, 97.5)),
            n_samples=n_samples,
        )

    def plot_convergence(self, func: Callable, a: float, b: float,
                         true_value=None, figsize=(10, 5)):
        """绘制蒙特卡洛收敛过程."""
        import matplotlib.pyplot as plt

        np.random.seed(self.seed)
        n_max = 10000
        x = np.random.uniform(a, b, n_max)
        y = np.array([func(xi) for xi in x])
        cumulative_mean = np.cumsum((b - a) * y) / np.arange(1, n_max + 1)

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(range(1, n_max + 1), cumulative_mean, "b-", alpha=0.7, label="MC 估计值")
        if true_value is not None:
            ax.axhline(y=true_value, color="r", linestyle="--", label=f"真实值 = {true_value}")
        ax.set_xlabel("样本数")
        ax.set_ylabel("估计值")
        ax.set_title("蒙特卡洛积分收敛过程")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log")
        plt.tight_layout()
        return fig
