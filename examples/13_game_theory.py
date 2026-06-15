"""
示例 13: 博弈论 — Nash均衡 + 矩阵博弈

展示博弈论在数模中的应用: 竞争策略、合作博弈。
"""

import sys
sys.path.insert(0, "..")
import numpy as np

from mathflow.game import NashEquilibrium, MatrixGame


def main():
    print("=" * 60)
    print("  示例: 博弈论分析")
    print("=" * 60)

    # ======== 1. 囚徒困境 ========
    print("\n【1】囚徒困境")
    print("-" * 40)

    # 支付矩阵: (合作=0, 背叛=1)
    p1 = [[-1, -3],   # 玩家1: 合作/背叛 vs 玩家2
          [ 0, -2]]
    p2 = [[-1, 0],    # 玩家2: 合作/背叛 vs 玩家1
          [-3, -2]]

    ne = NashEquilibrium(p1, p2,
                         p1_names=["合作", "背叛"],
                         p2_names=["合作", "背叛"])
    result = ne.find_all()
    print(ne.summary())

    # ======== 2. 性别之战 ========
    print("\n\n【2】性别之战 (混合策略)")
    print("-" * 40)

    # 男方偏好: 足球=0, 购物=1
    # 女方偏好: 足球=0, 购物=1
    p1_bof = [[3, 0], [0, 2]]  # 男方收益
    p2_bof = [[2, 0], [0, 3]]  # 女方收益

    bof = NashEquilibrium(p1_bof, p2_bof,
                          p1_names=["足球", "购物"],
                          p2_names=["足球", "购物"])
    bof_result = bof.find_all()
    print(bof.summary())

    # ======== 3. 矩阵博弈 (零和) ========
    print("\n\n【3】矩阵博弈 (零和博弈)")
    print("-" * 40)

    # 田忌赛马
    payoff = np.array([
        [1, -1, -1],   # 上等马
        [-1, 1, -1],   # 中等马
        [-1, -1, 1],   # 下等马
    ])

    game = MatrixGame(payoff)
    result = game.solve()
    print(game.summary())

    # ======== 4. 市场竞争博弈 ========
    print("\n\n【4】市场竞争 — 价格博弈")
    print("-" * 40)

    # 两家企业选择高价/中价/低价
    p1_market = [[10, 6, 3],
                 [12, 8, 4],
                 [7, 5, 2]]
    p2_market = [[10, 12, 7],
                 [6, 8, 5],
                 [3, 4, 2]]

    market = NashEquilibrium(p1_market, p2_market,
                             p1_names=["高价", "中价", "低价"],
                             p2_names=["高价", "中价", "低价"])
    market_result = market.find_all()
    print(market.summary())


if __name__ == "__main__":
    main()
