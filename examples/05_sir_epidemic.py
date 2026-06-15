"""
示例 5: SIR 传染病模型 (ODE)

基于 2020 年 COVID-19 疫情建模，展示 ODE 在数模中的应用。
可类比: 2022A 波浪能 (ODE), 2019A 高压油管 (ODE), 2020A 炉温曲线 (PDE→ODE).
"""

import sys
sys.path.insert(0, "..")
import numpy as np

from mathflow.ode import ODESolver, sir_model, lotka_volterra, damped_oscillator
from mathflow.stats import SensitivityAnalysis


def main():
    print("=" * 60)
    print("  示例: ODE 动力学模型")
    print("=" * 60)

    # ======== 1. SIR 传染病模型 ========
    print("\n【1】SIR 传染病模型")
    print("-" * 40)

    # 参数: R0 = beta/gamma = 3.0
    beta, gamma = 0.3, 0.1
    sir = sir_model(beta=beta, gamma=gamma)
    solver = ODESolver(sir, y0=[0.99, 0.01, 0])  # S0=99%, I0=1%
    result = solver.solve(t_span=(0, 160), dt=0.1)
    print(solver.summary())

    fig = solver.plot(labels=["易感者 S", "感染者 I", "恢复者 R"])
    fig.savefig("05_sir_model.png", dpi=150, bbox_inches="tight")
    print("  ✅ 已保存: 05_sir_model.png")

    # 关键指标
    peak_idx = np.argmax(result.y[:, 1])
    print(f"  峰值感染人数: {result.y[peak_idx, 1]:.2%} (第 {result.t[peak_idx]:.0f} 天)")
    print(f"  最终感染比例: {result.y[-1, 2]:.2%}")
    print(f"  基本再生数 R0 = β/γ = {beta/gamma:.1f}")

    # ======== 2. 灵敏度分析: β 对峰值的影响 ========
    print("\n\n【2】灵敏度分析: β 和 γ 对峰值感染率的影响")
    print("-" * 40)

    def peak_infection(params):
        b, g = params
        if b <= 0 or g <= 0:
            return 0
        s = ODESolver(sir_model(beta=b, gamma=g), y0=[0.99, 0.01, 0])
        r = s.solve(t_span=(0, 200), dt=0.5)
        return r.y[:, 1].max()

    sa = SensitivityAnalysis(peak_infection, n_vars=2, var_names=["β(传染率)", "γ(恢复率)"])
    sa.one_at_a_time(base_values=[0.3, 0.1], perturbation=0.3)
    print(sa.summary())

    fig2 = sa.plot()
    fig2.savefig("05_sir_sensitivity.png", dpi=150, bbox_inches="tight")
    print("  ✅ 已保存: 05_sir_sensitivity.png")

    # ======== 3. Lotka-Volterra 捕食者模型 ========
    print("\n\n【3】Lotka-Volterra 捕食者-被捕食者模型")
    print("-" * 40)

    lv = lotka_volterra(alpha=1.5, beta=1.0, delta=0.5, gamma=3.0)
    solver_lv = ODESolver(lv, y0=[1.0, 0.5])
    result_lv = solver_lv.solve(t_span=(0, 30), dt=0.01)
    print(solver_lv.summary())

    fig3 = solver_lv.plot(labels=["猎物 x", "捕食者 y"])
    fig3.savefig("05_lotka_volterra.png", dpi=150, bbox_inches="tight")
    print("  ✅ 已保存: 05_lotka_volterra.png")

    # ======== 4. 阻尼振动 ========
    print("\n\n【4】阻尼振动模型")
    print("-" * 40)

    osc = damped_oscillator(omega=5.0, zeta=0.1)
    solver_osc = ODESolver(osc, y0=[1.0, 0.0])
    result_osc = solver_osc.solve(t_span=(0, 10), dt=0.001)
    print(solver_osc.summary())

    fig4 = solver_osc.plot(labels=["位移 y", "速度 y'"])
    fig4.savefig("05_damped_oscillator.png", dpi=150, bbox_inches="tight")
    print("  ✅ 已保存: 05_damped_oscillator.png")


if __name__ == "__main__":
    main()
