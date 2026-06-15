"""
灵敏度分析 (Sensitivity Analysis)

分析模型参数变化对输出结果的影响程度。
数模论文中必备模块，用于证明模型的鲁棒性。

Example:
    >>> from mathflow.stats import SensitivityAnalysis
    >>> def model(x):
    ...     return 2*x[0]**2 + 3*x[1] - x[2]
    >>> sa = SensitivityAnalysis(model, n_vars=3, var_names=["温度","压力","浓度"])
    >>> result = sa.one_at_a_time(base_values=[5, 10, 3], perturbation=0.1)
    >>> print(sa.summary())
"""

import numpy as np
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple


@dataclass
class SensitivityResult:
    """灵敏度分析结果."""
    method: str
    sensitivities: np.ndarray    # 灵敏度指标
    var_names: List[str]
    base_output: float
    base_values: np.ndarray
    detail: dict


class SensitivityAnalysis:
    """
    灵敏度分析.

    Parameters
    ----------
    model_func : callable
        模型函数 f(x) -> float, x 是参数向量
    n_vars : int
        参数个数
    var_names : list of str, optional
        参数名称
    """

    def __init__(self, model_func: Callable, n_vars: int, var_names=None):
        self.model_func = model_func
        self.n_vars = n_vars
        self.var_names = var_names or [f"参数{i+1}" for i in range(n_vars)]
        self._result = None

    def one_at_a_time(self, base_values, perturbation=0.1, n_levels=21):
        """
        单因素灵敏度分析 (OAT/OAT).

        每次只改变一个参数，观察输出变化。

        Parameters
        ----------
        base_values : array-like
            基准参数值
        perturbation : float
            参数扰动比例 (如 0.1 表示 ±10%)
        n_levels : int
            参数取值的水平数
        """
        base = np.asarray(base_values, dtype=float)
        base_output = self.model_func(base)

        sensitivities = np.zeros(self.n_vars)
        param_outputs = {}

        for i in range(self.n_vars):
            low = base[i] * (1 - perturbation)
            high = base[i] * (1 + perturbation)
            levels = np.linspace(low, high, n_levels)

            outputs = []
            for val in levels:
                x = base.copy()
                x[i] = val
                outputs.append(self.model_func(x))

            outputs = np.array(outputs)
            param_outputs[self.var_names[i]] = (levels, outputs)

            # 灵敏度 = 输出变化范围 / 基准输出
            if abs(base_output) > 1e-10:
                sensitivities[i] = (outputs.max() - outputs.min()) / abs(base_output)
            else:
                sensitivities[i] = outputs.max() - outputs.min()

        self._result = SensitivityResult(
            method="OAT (单因素)",
            sensitivities=sensitivities,
            var_names=self.var_names,
            base_output=base_output,
            base_values=base,
            detail={"param_outputs": param_outputs, "perturbation": perturbation},
        )
        return self._result

    def morris_screening(self, n_trajectories=10, n_levels=4, seed=42):
        """
        Morris 筛选法 (Morris Screening / Elementary Effects).

        适用于高维参数空间的初步筛选。

        Parameters
        ----------
        n_trajectories : int
            轨迹数
        n_levels : int
            水平数
        """
        np.random.seed(seed)
        p = self.n_vars
        delta = 1.0 / (n_levels - 1)

        # 生成基准网格
        grid = np.linspace(0, 1, n_levels)

        elementary_effects = {i: [] for i in range(p)}

        for _ in range(n_trajectories):
            # 随机起点
            x_norm = np.random.choice(grid, p)

            # 计算基准输出
            x_base = x_norm.copy()
            base_out = self.model_func(x_base)

            # 随机排列参数扰动顺序
            order = np.random.permutation(p)

            for idx in order:
                x_pert = x_norm.copy()
                # 在网格上移动 delta
                if x_pert[idx] + delta <= 1:
                    x_pert[idx] += delta
                else:
                    x_pert[idx] -= delta

                pert_out = self.model_func(x_pert)
                ee = (pert_out - base_out) / delta
                elementary_effects[idx].append(ee)

                base_out = pert_out
                x_norm = x_pert

        # 计算 Morris 指标
        mu_star = np.zeros(p)  # 平均绝对 elementary effect
        sigma = np.zeros(p)    # 标准差

        for i in range(p):
            effects = np.array(elementary_effects[i])
            mu_star[i] = np.mean(np.abs(effects))
            sigma[i] = np.std(effects)

        self._result = SensitivityResult(
            method="Morris Screening",
            sensitivities=mu_star,
            var_names=self.var_names,
            base_output=0,
            base_values=np.zeros(p),
            detail={"mu_star": mu_star, "sigma": sigma, "elementary_effects": elementary_effects},
        )
        return self._result

    def sobol_indices(self, n_samples=1024, bounds=None, seed=42):
        """
        Sobol 全局灵敏度分析 (一阶和总效应指数).

        Parameters
        ----------
        n_samples : int
            采样数 (会自动调整为 2 的幂)
        bounds : list of (min, max), optional
            参数范围
        """
        np.random.seed(seed)
        p = self.n_vars

        if bounds is None:
            bounds = [(0, 1)] * p
        bounds = np.array(bounds)

        # Saltelli 采样方案
        n = int(2**np.ceil(np.log2(n_samples)))
        base_a = np.random.uniform(0, 1, (n, p))
        base_b = np.random.uniform(0, 1, (n, p))

        # 映射到实际范围
        def to_real(x):
            return bounds[:, 0] + x * (bounds[:, 1] - bounds[:, 0])

        # 计算 f(A), f(B)
        f_a = np.array([self.model_func(to_real(row)) for row in base_a])
        f_b = np.array([self.model_func(to_real(row)) for row in base_b])

        var_y = np.var(np.concatenate([f_a, f_b]))
        mean_y = np.mean(np.concatenate([f_a, f_b]))

        # 一阶 Sobol 指数
        s1 = np.zeros(p)
        st = np.zeros(p)

        for i in range(p):
            # 构造 AB_i 矩阵 (A 的第 i 列替换为 B 的第 i 列)
            ab_i = base_a.copy()
            ab_i[:, i] = base_b[:, i]
            f_ab_i = np.array([self.model_func(to_real(row)) for row in ab_i])

            # 一阶指数
            s1[i] = np.mean(f_b * (f_ab_i - f_a)) / var_y if var_y > 0 else 0
            # 总效应指数
            st[i] = 0.5 * np.mean((f_a - f_ab_i)**2) / var_y if var_y > 0 else 0

        self._result = SensitivityResult(
            method="Sobol",
            sensitivities=s1,
            var_names=self.var_names,
            base_output=mean_y,
            base_values=np.zeros(p),
            detail={"S1": s1, "ST": st, "var_Y": var_y},
        )
        return self._result

    @property
    def result(self):
        if self._result is None:
            raise RuntimeError("请先调用分析方法")
        return self._result

    def plot(self, figsize=(10, 6), style="bar"):
        """绘制灵敏度分析结果.

        Parameters
        ----------
        style : str
            "bar" (柱状图) 或 "tornado" (龙卷风图)
        """
        import matplotlib.pyplot as plt

        r = self._result
        n = len(r.sensitivities)

        if style == "tornado":
            return self._plot_tornado(figsize)

        fig, ax = plt.subplots(figsize=figsize)
        colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, n))
        order = np.argsort(-r.sensitivities)

        bars = ax.barh(range(n), r.sensitivities[order], color=colors)
        ax.set_yticks(range(n))
        ax.set_yticklabels([r.var_names[i] for i in order])
        ax.set_xlabel("灵敏度指标")
        ax.set_title(f"灵敏度分析 ({r.method})")
        ax.invert_yaxis()

        for bar, val in zip(bars, r.sensitivities[order]):
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", fontsize=10)

        plt.tight_layout()
        return fig

    def _plot_tornado(self, figsize=(10, 6)):
        """绘制龙卷风图."""
        import matplotlib.pyplot as plt

        r = self._result
        if r.method != "OAT (单因素)":
            raise ValueError("龙卷风图仅支持 OAT 方法")

        detail = r.detail
        n = len(r.var_names)

        # 计算每个参数的变化范围
        effects = []
        for i in range(n):
            outputs = detail["param_outputs"][r.var_names[i]][1]
            base = r.base_output
            low_effect = outputs.min() - base
            high_effect = outputs.max() - base
            effects.append((r.var_names[i], low_effect, high_effect))

        # 按影响范围排序
        effects.sort(key=lambda x: abs(x[2] - x[1]), reverse=True)

        fig, ax = plt.subplots(figsize=figsize)
        y_pos = range(n)

        for i, (name, low, high) in enumerate(effects):
            ax.barh(i, high, left=0, color="#e74c3c", alpha=0.7, height=0.6)
            ax.barh(i, low, left=0, color="#3498db", alpha=0.7, height=0.6)

        ax.set_yticks(y_pos)
        ax.set_yticklabels([e[0] for e in effects])
        ax.set_xlabel("对目标函数的影响")
        ax.set_title("龙卷风图 (灵敏度分析)")
        ax.axvline(x=0, color="black", linewidth=1)
        ax.invert_yaxis()

        # 图例
        from matplotlib.patches import Patch
        ax.legend([Patch(color="#e74c3c", alpha=0.7), Patch(color="#3498db", alpha=0.7)],
                  ["正向影响", "负向影响"], loc="lower right")

        plt.tight_layout()
        return fig

    def summary(self):
        r = self._result
        lines = [
            "=" * 60,
            f"  灵敏度分析结果 ({r.method})",
            "=" * 60,
        ]
        if r.method == "Sobol":
            detail = r.detail
            lines.append(f"  {'参数':>12s}  {'一阶指数S1':>12s}  {'总效应ST':>12s}")
            lines.append("-" * 60)
            for i in range(len(r.var_names)):
                lines.append(f"  {r.var_names[i]:>12s}  {detail['S1'][i]:>12.4f}  {detail['ST'][i]:>12.4f}")
        elif r.method == "Morris Screening":
            detail = r.detail
            lines.append(f"  {'参数':>12s}  {'μ*':>12s}  {'σ':>12s}")
            lines.append("-" * 60)
            for i in range(len(r.var_names)):
                lines.append(f"  {r.var_names[i]:>12s}  {detail['mu_star'][i]:>12.4f}  {detail['sigma'][i]:>12.4f}")
        else:
            lines.append(f"  基准输出: {r.base_output:.4f}")
            lines.append("-" * 60)
            for i in np.argsort(-r.sensitivities):
                lines.append(f"  {r.var_names[i]:>12s}  灵敏度 = {r.sensitivities[i]:.4f}")

        # 排名
        order = np.argsort(-r.sensitivities)
        lines.append("-" * 60)
        lines.append("  灵敏度排名:")
        for rank, i in enumerate(order, 1):
            lines.append(f"    第{rank}名: {r.var_names[i]} ({r.sensitivities[i]:.4f})")
        lines.append("=" * 60)
        return "\n".join(lines)
