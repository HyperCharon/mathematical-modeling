"""
示例 7: 排队论 — 机场出租车问题

基于 2019C 机场的出租车问题，展示排队论在数模中的应用。
"""

import sys
sys.path.insert(0, "..")
import numpy as np

from mathflow.simulation import QueueModel, MonteCarlo


def main():
    print("=" * 60)
    print("  示例: 排队论 — 机场出租车调度")
    print("=" * 60)

    # ======== 1. 基础排队模型 ========
    print("\n【1】M/M/c 排队模型分析")
    print("-" * 40)

    # 到达率: 每小时 60 辆出租车到达
    # 服务率: 每个通道每小时处理 15 辆
    arrival_rate = 60   # 辆/小时
    service_rate = 15   # 辆/小时/通道

    # 分析不同通道数的效果
    print(f"  到达率 λ = {arrival_rate} 辆/小时")
    print(f"  服务率 μ = {service_rate} 辆/小时/通道")
    print()

    for c in range(5, 9):
        try:
            q = QueueModel(arrival_rate, service_rate, n_servers=c)
            r = q.solve()
            print(f"  c={c}: ρ={r['rho']:.3f}, L={r['L']:.2f}, Lq={r['Lq']:.2f}, "
                  f"W={r['W']*60:.1f}min, Wq={r['Wq']*60:.1f}min")
        except ValueError as e:
            print(f"  c={c}: {e}")

    # ======== 2. 最优通道数分析 ========
    print("\n\n【2】最优通道数决策")
    print("-" * 40)

    # 成本分析
    # - 每个通道运营成本: 200 元/小时
    # - 乘客等待成本: 50 元/人/小时
    channel_cost = 200
    wait_cost = 50

    print(f"  通道成本: {channel_cost} 元/小时")
    print(f"  等待成本: {wait_cost} 元/人/小时")
    print()

    best_total = float("inf")
    best_c = 0

    for c in range(4, 10):
        try:
            q = QueueModel(arrival_rate, service_rate, n_servers=c)
            r = q.solve()
            total_cost = c * channel_cost + r['Lq'] * wait_cost
            print(f"  c={c}: 运营成本={c*channel_cost}元, 等待成本={r['Lq']*wait_cost:.0f}元, "
                  f"总成本={total_cost:.0f}元")
            if total_cost < best_total:
                best_total = total_cost
                best_c = c
        except ValueError:
            pass

    print(f"\n  ✅ 最优通道数: {best_c}, 总成本: {best_total:.0f} 元/小时")

    # ======== 3. 蒙特卡洛仿真验证 ========
    print("\n\n【3】蒙特卡洛仿真验证")
    print("-" * 40)

    q = QueueModel(arrival_rate, service_rate, n_servers=best_c)
    sim = q.simulate(n_arrivals=10000)
    r = q.solve()

    print(f"  理论 Wq = {r['Wq']*60:.2f} 分钟")
    print(f"  仿真 Wq = {sim['sim_Wq']*60:.2f} 分钟")
    print(f"  理论 Lq = {r['Lq']:.2f}")
    print(f"  仿真 Lq = {sim['sim_Lq']:.2f}")
    print(f"  误差: {abs(r['Wq'] - sim['sim_Wq'])/r['Wq']*100:.1f}%")

    # ======== 4. M/M/1/K 有限容量模型 ========
    print("\n\n【4】有限等候区模型 (M/M/1/K)")
    print("-" * 40)

    # 候车区最多容纳 20 辆车
    for K in [10, 15, 20, 30]:
        q_k = QueueModel(arrival_rate=10, service_rate=8, n_servers=1, capacity=K)
        r_k = q_k.solve()
        print(f"  K={K}: 丢失率={r_k['P_loss']:.4f}, 有效到达率={r_k['lambda_eff']:.2f}, L={r_k['L']:.2f}")


if __name__ == "__main__":
    main()
