"""
示例 1: AHP + TOPSIS 综合评价

场景: 选择最佳旅游目的地
- 4个候选方案: 北京、上海、成都、西安
- 4个评价指标: 景点、美食、交通、费用
"""

import sys
sys.path.insert(0, "..")

from mathflow.evaluate import AHP, TOPSIS, EntropyWeight
import numpy as np


def main():
    print("=" * 60)
    print("  示例: AHP + TOPSIS 综合评价 — 旅游目的地选择")
    print("=" * 60)

    # ======== Step 1: AHP 确定主观权重 ========
    print("\n【Step 1】AHP 层次分析法确定主观权重")
    print("-" * 40)

    # 判断矩阵: 景点 vs 美食 vs 交通 vs 费用
    # 行列顺序: 景点、美食、交通、费用
    ahp_matrix = [
        [1,   3,   5,   7],     # 景点比美食重要3倍，比交通重要5倍...
        [1/3, 1,   3,   5],     # 美食比交通重要3倍
        [1/5, 1/3, 1,   3],     # 交通比费用重要3倍
        [1/7, 1/5, 1/3, 1],     # 费用最不重要
    ]

    ahp = AHP(method="eigenvalue")
    ahp.set_matrix(ahp_matrix)
    ahp.fit()
    print(ahp.summary())

    subjective_weights = ahp.weights
    print(f"\n  AHP 主观权重: {subjective_weights}")

    # ======== Step 2: 熵权法确定客观权重 ========
    print("\n\n【Step 2】熵权法确定客观权重")
    print("-" * 40)

    # 评价矩阵 (4个方案 × 4个指标)
    # 指标: 景点(效益型), 美食(效益型), 交通(效益型), 费用(成本型→越低越好)
    eval_data = np.array([
        [85, 90, 70, 800],   # 北京
        [75, 85, 90, 1200],  # 上海
        [90, 95, 60, 600],   # 成都
        [88, 80, 65, 500],   # 西安
    ])

    # 费用是成本型，需要转换
    ew = EntropyWeight(eval_data, types=[1, 1, 1, -1])
    ew.fit()
    print(ew.summary(labels=["景点", "美食", "交通", "费用"]))

    objective_weights = ew.weights
    print(f"\n  熵权法客观权重: {objective_weights}")

    # ======== Step 3: 综合权重 ========
    print("\n\n【Step 3】组合权重 (主观×0.6 + 客观×0.4)")
    print("-" * 40)

    combined_weights = 0.6 * subjective_weights + 0.4 * objective_weights
    combined_weights = combined_weights / combined_weights.sum()
    print(f"  组合权重: {combined_weights}")

    # ======== Step 4: TOPSIS 综合评价 ========
    print("\n\n【Step 4】TOPSIS 综合评价")
    print("-" * 40)

    # 费用是成本型 (type=-1)，其他是效益型 (type=1)
    topsis = TOPSIS(eval_data, weights=combined_weights, types=[1, 1, 1, -1])
    topsis.fit()
    print(topsis.summary(labels=["北京", "上海", "成都", "西安"]))

    # ======== Step 5: 可视化 ========
    print("\n\n【Step 5】生成可视化图表...")

    fig1 = ahp.plot_weights(labels=["景点", "美食", "交通", "费用"])
    fig1.savefig("01_ahp_weights.png", dpi=150, bbox_inches="tight")
    print("  ✅ 已保存: 01_ahp_weights.png")

    fig2 = topsis.plot_scores(labels=["北京", "上海", "成都", "西安"])
    fig2.savefig("01_topsis_scores.png", dpi=150, bbox_inches="tight")
    print("  ✅ 已保存: 01_topsis_scores.png")

    fig3 = ew.plot_weights(labels=["景点", "美食", "交通", "费用"])
    fig3.savefig("01_entropy_weights.png", dpi=150, bbox_inches="tight")
    print("  ✅ 已保存: 01_entropy_weights.png")

    print("\n" + "=" * 60)
    print("  示例完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
