"""
示例 11: LaTeX 报告自动生成

展示如何用 MathFlow 自动生成数模论文的 LaTeX 代码。
"""

import sys
sys.path.insert(0, "..")

from mathflow.report import LatexReport
from mathflow.evaluate import AHP, TOPSIS
import numpy as np


def main():
    print("=" * 60)
    print("  示例: LaTeX 报告自动生成")
    print("=" * 60)

    # 创建报告
    report = LatexReport("2024年C题 农作物种植策略研究", "MathFlow Team")

    # 添加问题重述
    report.add_section("问题重述")
    report.add_text("本题要求根据历史种植数据和市场信息，制定最优的农作物种植策略，"
                    "以最大化预期收益并控制风险。")

    # 模型建立
    report.add_section("模型建立")
    report.add_subsection = lambda t: report.add_section(t, level=2)

    # AHP 权重
    report.add_section("基于 AHP 的指标权重确定", level=2)
    report.add_text("采用层次分析法确定各评价指标的主观权重。")
    report.add_text("构造判断矩阵如公式(1)所示：")

    report.add_equation(
        r"A = \begin{pmatrix} 1 & 3 & 5 \\ 1/3 & 1 & 3 \\ 1/5 & 1/3 & 1 \end{pmmatrix}",
        label="eq:ahp_matrix"
    )

    # 运行 AHP
    ahp = AHP()
    ahp.set_matrix([
        [1, 3, 5],
        [1/3, 1, 3],
        [1/5, 1/3, 1],
    ])
    ahp.fit()
    report.add_ahp_result(ahp)

    # TOPSIS
    report.add_section("TOPSIS 综合评价", level=2)
    data = np.array([
        [85, 90, 70],
        [75, 80, 90],
        [90, 85, 60],
    ])
    topsis = TOPSIS(data, weights=ahp.weights, types=[1, 1, -1])
    topsis.fit()
    report.add_topsis_result(topsis, labels=["方案A", "方案B", "方案C"])

    # 添加公式
    report.add_section("关键公式", level=2)

    report.add_text("SIR 传染病模型的基本方程组为：")
    report.add_equation_system(
        LatexReport.sir_equations(),
        label="eq:sir"
    )

    report.add_text("模型的决定系数计算如公式所示：")
    report.add_equation(LatexReport.r_squared(), label="eq:r2")

    # 列表
    report.add_section("模型优点", level=2)
    report.add_itemize([
        "综合考虑了主观和客观权重",
        "TOPSIS 方法计算简便，物理含义明确",
        "灵敏度分析验证了模型的鲁棒性",
    ])

    # 保存
    report.save("paper_example.tex")
    print(report.summary())

    # 预览生成的 LaTeX
    print("\n--- LaTeX 代码预览 (前50行) ---")
    content = report.build()
    for i, line in enumerate(content.split("\n")[:50]):
        print(f"  {i+1:3d} | {line}")
    print("  ...")


if __name__ == "__main__":
    main()
