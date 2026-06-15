"""
示例 12: 多目标优化 + 项目调度

基于 CPM/PERT 和 NSGA-II，展示项目管理和多目标决策。
类比: 2024B 生产决策, 2022E 生产调度
"""

import sys
sys.path.insert(0, "..")
import numpy as np

from mathflow.graph import CPM, Hungarian
from mathflow.optimize import NSGA2


def main():
    print("=" * 60)
    print("  示例: 项目调度 + 多目标优化")
    print("=" * 60)

    # ======== 1. CPM 关键路径法 ========
    print("\n【1】CPM 关键路径法 — 项目工期分析")
    print("-" * 40)

    cpm = CPM()
    # 模拟一个软件开发项目
    cpm.add_activity("需求分析", duration=5, predecessors=[])
    cpm.add_activity("系统设计", duration=8, predecessors=["需求分析"])
    cpm.add_activity("数据库开发", duration=6, predecessors=["系统设计"])
    cpm.add_activity("前端开发", duration=10, predecessors=["系统设计"])
    cpm.add_activity("后端开发", duration=12, predecessors=["系统设计"])
    cpm.add_activity("集成测试", duration=5, predecessors=["数据库开发", "前端开发", "后端开发"])
    cpm.add_activity("部署上线", duration=3, predecessors=["集成测试"])

    result = cpm.solve()
    print(cpm.summary())

    fig = cpm.plot_gantt()
    fig.savefig("12_gantt.png", dpi=150, bbox_inches="tight")
    print("  ✅ 已保存: 12_gantt.png")

    # ======== 2. 匈牙利算法 — 指派问题 ========
    print("\n\n【2】匈牙利算法 — 任务指派")
    print("-" * 40)

    # 4个工人 × 4个任务的成本矩阵
    cost = np.array([
        [10, 5, 13, 15],
        [3, 9, 18, 13],
        [13, 7, 4,  15],
        [12, 11, 14, 8],
    ])

    hungarian = Hungarian(cost)
    result = hungarian.solve()
    print(hungarian.summary())

    fig2 = hungarian.plot()
    fig2.savefig("12_hungarian.png", dpi=150, bbox_inches="tight")
    print("  ✅ 已保存: 12_hungarian.png")

    # 最大收益指派
    print("\n  最大收益指派:")
    profit = np.array([
        [20, 15, 25, 30],
        [10, 25, 35, 20],
        [30, 20, 15, 25],
        [25, 30, 20, 15],
    ])
    h_max = Hungarian(profit, maximize=True)
    r_max = h_max.solve()
    print(h_max.summary())

    # ======== 3. NSGA-II 多目标优化 ========
    print("\n\n【3】NSGA-II 多目标优化")
    print("-" * 40)

    # 双目标: 最小化成本 + 最小化工期
    def cost_func(x):
        """成本 = 固定成本 + 变动成本 × 规模."""
        return 100 + 50 * x[0] + 30 * x[1]

    def time_func(x):
        """工期 = 基准工期 × (1 - 效率因子)."""
        return 20 * (1 - 0.3 * x[0] - 0.2 * x[1])

    nsga = NSGA2(
        objectives=[cost_func, time_func],
        n_vars=2,
        bounds=[(0, 1), (0, 1)],
        pop_size=100,
        generations=200,
    )
    nsga_result = nsga.run()
    print(nsga.summary())

    fig3 = nsga.plot_pareto()
    fig3.savefig("12_pareto.png", dpi=150, bbox_inches="tight")
    print("  ✅ 已保存: 12_pareto.png")

    # 选择折中解 (最接近理想点)
    front = nsga_result.pareto_front
    ideal = front.min(axis=0)
    distances = np.sqrt(((front - ideal) ** 2).sum(axis=1))
    best_idx = distances.argmin()
    print(f"\n  折中解:")
    print(f"    成本 = {front[best_idx, 0]:.2f}")
    print(f"    工期 = {front[best_idx, 1]:.2f}")
    print(f"    决策变量 = {nsga_result.pareto_solutions[best_idx]}")


if __name__ == "__main__":
    main()
