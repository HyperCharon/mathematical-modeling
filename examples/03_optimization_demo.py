"""
示例 3: 优化问题 — 线性规划 + 遗传算法

场景: 生产计划优化
- 产品A: 利润4元/个, 需要2小时加工
- 产品B: 利润3元/个, 需要1小时加工
- 约束: 总加工时间 ≤ 10小时, 产品A ≤ 4个
"""

import sys
sys.path.insert(0, "..")

from mathflow.optimize import LinearProgramming, GeneticAlgorithm
import numpy as np


def main():
    print("=" * 60)
    print("  示例: 优化问题求解")
    print("=" * 60)

    # ======== 方法1: 线性规划 (精确解) ========
    print("\n【方法1】线性规划")
    print("-" * 40)

    lp = LinearProgramming()
    lp.set_objective([4, 3], sense="max", var_names=["产品A", "产品B"])
    lp.add_constraint([2, 1], "<=", 10)   # 加工时间约束
    lp.add_constraint([1, 0], "<=", 4)    # 产品A数量约束
    lp.add_constraint([0, 1], "<=", 8)    # 产品B数量约束

    result = lp.solve()
    print(lp.summary())

    # 可视化可行域
    fig = lp.plot_feasible_region(x_range=(-1, 6), y_range=(-1, 10))
    fig.savefig("03_lp_feasible.png", dpi=150, bbox_inches="tight")
    print("  ✅ 已保存: 03_lp_feasible.png")

    # ======== 方法2: 遗传算法 (启发式) ========
    print("\n\n【方法2】遗传算法")
    print("-" * 40)

    def fitness(x):
        """适应度函数 (最大化利润)."""
        a, b = x[0], x[1]
        # 罚函数处理约束
        penalty = 0
        if 2 * a + b > 10:
            penalty += (2 * a + b - 10) * 100
        if a > 4:
            penalty += (a - 4) * 100
        if b > 8:
            penalty += (b - 8) * 100
        if a < 0 or b < 0:
            penalty += 1000
        return 4 * a + 3 * b - penalty

    ga = GeneticAlgorithm(
        fitness_func=fitness,
        n_vars=2,
        bounds=[(0, 6), (0, 10)],
        pop_size=100,
        generations=200,
    )
    ga_result = ga.run(verbose=False)
    print(ga.summary())

    fig2 = ga.plot_convergence()
    fig2.savefig("03_ga_convergence.png", dpi=150, bbox_inches="tight")
    print("  ✅ 已保存: 03_ga_convergence.png")

    # ======== 方法3: 粒子群优化 ========
    print("\n\n【方法3】粒子群优化 (PSO)")
    print("-" * 40)

    from mathflow.optimize import PSO

    pso = PSO(
        fitness_func=fitness,
        n_vars=2,
        bounds=[(0, 6), (0, 10)],
        n_particles=50,
        max_iter=200,
    )
    pso_result = pso.run()
    print(pso.summary())


if __name__ == "__main__":
    main()
