"""
示例 14: 动态规划 — 背包 + 资源分配

展示动态规划在数模中的应用。
"""

import sys
sys.path.insert(0, "..")

from mathflow.dp import Knapsack, ResourceAllocation


def main():
    print("=" * 60)
    print("  示例: 动态规划")
    print("=" * 60)

    # ======== 1. 0-1 背包 ========
    print("\n【1】0-1 背包问题")
    print("-" * 40)

    knap = Knapsack(capacity=15)
    knap.add_item(weight=2, value=3, name="物资A")
    knap.add_item(weight=3, value=4, name="物资B")
    knap.add_item(weight=4, value=5, name="物资C")
    knap.add_item(weight=5, value=8, name="物资D")
    knap.add_item(weight=6, value=9, name="物资E")
    knap.add_item(weight=7, value=10, name="物资F")

    result = knap.solve_01()
    print(knap.summary("01"))

    fig = knap.plot_dp("01")
    fig.savefig("14_knapsack_dp.png", dpi=150, bbox_inches="tight")
    print("  ✅ 已保存: 14_knapsack_dp.png")

    # ======== 2. 完全背包 ========
    print("\n\n【2】完全背包问题")
    print("-" * 40)

    knap2 = Knapsack(capacity=10)
    knap2.add_item(weight=2, value=6, name="物品A")
    knap2.add_item(weight=3, value=8, name="物品B")
    knap2.add_item(weight=4, value=9, name="物品C")

    result2 = knap2.solve_complete()
    print(knap2.summary("complete"))

    # ======== 3. 资源分配 ========
    print("\n\n【3】资源分配问题")
    print("-" * 40)

    # 将5台设备分配给3个车间
    profit = [
        [0, 3, 7, 9, 12, 13],   # 车间A
        [0, 5, 10, 12, 14, 15],  # 车间B
        [0, 4, 6, 11, 12, 13],   # 车间C
    ]

    ra = ResourceAllocation(profit, total_resource=5)
    result3 = ra.solve()
    print(ra.summary())


if __name__ == "__main__":
    main()
