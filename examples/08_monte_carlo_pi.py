"""
示例 8: 蒙特卡洛方法

展示蒙特卡洛在数模中的多种应用: 积分、风险分析、概率估计。
"""

import sys
sys.path.insert(0, "..")
import numpy as np

from mathflow.simulation import MonteCarlo


def main():
    print("=" * 60)
    print("  示例: 蒙特卡洛方法应用")
    print("=" * 60)

    mc = MonteCarlo(seed=42)

    # ======== 1. 估算 π ========
    print("\n【1】蒙特卡洛估算 π")
    print("-" * 40)
    for n in [1000, 10000, 100000, 1000000]:
        pi_est, error = mc.estimate_pi(n_samples=n)
        print(f"  n={n:>8d}: π ≈ {pi_est:.6f}, 误差 = {error:.6f}")

    # ======== 2. 蒙特卡洛积分 ========
    print("\n\n【2】蒙特卡洛积分")
    print("-" * 40)

    # ∫_0^1 x² dx = 1/3
    result = mc.integrate(lambda x: x**2, 0, 1, n_samples=100000)
    print(f"  ∫₀¹ x² dx = {result.estimate:.6f} (真实值: {1/3:.6f})")
    print(f"  95% 置信区间: [{result.confidence_interval[0]:.6f}, {result.confidence_interval[1]:.6f}]")

    # ∫_0^π sin(x) dx = 2
    result2 = mc.integrate(lambda x: np.sin(x), 0, np.pi, n_samples=100000)
    print(f"  ∫₀π sin(x) dx = {result2.estimate:.6f} (真实值: {2.000000})")

    # 收敛过程图
    fig = mc.plot_convergence(lambda x: x**2, 0, 1, true_value=1/3)
    fig.savefig("08_mc_convergence.png", dpi=150, bbox_inches="tight")
    print("  ✅ 已保存: 08_mc_convergence.png")

    # ======== 3. 风险分析 ========
    print("\n\n【3】项目利润风险分析")
    print("-" * 40)

    # 项目利润 = 销量 × 单价 - 固定成本 - 变动成本 × 销量
    def profit_model(sales, price, var_cost, fixed_cost):
        return sales * price - fixed_cost - var_cost * sales

    distributions = [
        lambda: np.random.normal(10000, 2000),    # 销量 ~ N(10000, 2000²)
        lambda: np.random.uniform(50, 80),         # 单价 ~ U(50, 80)
        lambda: np.random.normal(30, 5),           # 变动成本 ~ N(30, 5²)
        lambda: np.random.normal(150000, 20000),   # 固定成本 ~ N(150000, 20000²)
    ]

    risk = mc.simulate_risk(profit_model, n_samples=50000, distributions=distributions)

    print(f"  期望利润: {risk.estimate:,.0f} 元")
    print(f"  标准误差: {risk.std_error:,.0f} 元")
    print(f"  95% 置信区间: [{risk.confidence_interval[0]:,.0f}, {risk.confidence_interval[1]:,.0f}] 元")
    print(f"  亏损概率: 需要更多模拟数据计算")

    # 计算亏损概率
    np.random.seed(42)
    profits = []
    for _ in range(50000):
        p = profit_model(
            np.random.normal(10000, 2000),
            np.random.uniform(50, 80),
            np.random.normal(30, 5),
            np.random.normal(150000, 20000),
        )
        profits.append(p)
    profits = np.array(profits)
    loss_prob = (profits < 0).mean()
    print(f"  亏损概率: {loss_prob:.2%}")
    print(f"  利润 > 20万的概率: {(profits > 200000).mean():.2%}")


if __name__ == "__main__":
    main()
