"""
示例 16: 论文写作全流程

演示如何用 MathFlow 的论文写作模块，从分析结果一键生成论文框架。
"""

import sys
sys.path.insert(0, "..")

from mathflow.paper import AbstractGenerator, ModelEvaluator, SectionGenerator, FullPaper, Phrases


def main():
    print("=" * 60)
    print("  示例: 数模论文写作全流程")
    print("=" * 60)

    # ======== 1. 摘要生成 ========
    print("\n【1】摘要自动生成")
    print("-" * 40)

    ag = AbstractGenerator(
        "2024年C题 农作物种植策略优化",
        background="本文针对某地区的农作物种植策略优化问题，综合考虑了产量、价格、成本和风险等因素。"
    )
    ag.add_problem_solution(
        description="建立最优种植方案",
        method="线性规划 + 遗传算法",
        result="最优年收益 285.6 万元",
        model_name="多目标优化",
    )
    ag.add_problem_solution(
        description="预测未来产量和价格",
        method="灰色预测 GM(1,1) + ARIMA",
        result="预测精度 R²=0.95",
        model_name="组合预测",
    )
    ag.add_problem_solution(
        description="风险分析与稳健决策",
        method="蒙特卡洛模拟 + 灵敏度分析",
        result="在95%置信水平下收益不低于 250 万元",
        model_name="随机优化",
    )
    ag.add_strength("综合考虑了主观和客观因素")
    ag.add_strength("灵敏度分析验证了模型的稳健性")
    ag.set_keywords(["线性规划", "灰色预测", "蒙特卡洛", "灵敏度分析", "种植优化"])

    print(ag.generate("standard"))
    print(f"\n  摘要字数: {ag.word_count()}")

    # ======== 2. 模型评价 ========
    print("\n\n【2】模型评价自动生成")
    print("-" * 40)

    me = ModelEvaluator("AHP-TOPSIS 综合评价模型")
    me.add_strength("综合考虑了主观权重（AHP）和客观权重（熵权法），避免了单一赋权的偏差",
                    "组合权重的 Kendall 协调系数为 0.85")
    me.add_strength("TOPSIS 方法计算简便，物理含义明确")
    me.add_strength("灵敏度分析表明模型对参数变化具有较好的稳健性")
    me.add_weakness("AHP 判断矩阵依赖专家经验，主观性较强")
    me.add_weakness("未考虑指标之间的相关性")
    me.add_improvement("引入博弈论组合赋权，提高权重确定的客观性", "权重确定")
    me.add_improvement("采用 CRITIC 法替代熵权法，考虑指标间的冲突性", "客观赋权")
    me.add_extension("供应链管理中的供应商评价")
    me.add_extension("人才选拔与绩效评估")
    me.add_extension("环境质量综合评价")
    me.set_metric("R²", "0.92")
    me.set_metric("RMSE", "3.45")

    print(me.generate())

    # ======== 3. 灵敏度分析描述 ========
    print("\n\n【3】灵敏度分析描述")
    print("-" * 40)

    # 龙卷风图描述
    tornado_desc = me.generate_tornado_description(
        param_names=["传染率β", "恢复率γ", "初始感染比例I₀", "接触率"],
        sensitivities=[0.73, 0.52, 0.31, 0.18],
        fig_num=5,
    )
    print(tornado_desc)

    # 灵敏度结论
    conclusion = me.generate_sensitivity_conclusion(
        max_change=8.5, param="传染率β", target="峰值感染率"
    )
    print(f"\n{conclusion}")

    # ======== 4. 常用语料 ========
    print("\n\n【4】常用语料库")
    print("-" * 40)

    print("过渡词示例:")
    for purpose in ["addition", "contrast", "cause", "summary"]:
        words = Phrases.get("transitions", purpose)
        print(f"  {purpose}: {', '.join(words[:3])}")

    print("\n模型描述句式:")
    desc = Phrases.get("model_description", "abstract", type="优化", constraints="资源有限", objective="收益最大")
    print(f"  {desc}")

    # ======== 5. 完整论文生成 ========
    print("\n\n【5】完整论文模板生成")
    print("-" * 40)

    paper = FullPaper("2024年C题 农作物种植策略优化", year=2024)
    paper.set_background(
        "某地区有 12 个种植地块，需要种植 8 种农作物。"
        "已知各地块的面积、土壤条件，以及各作物的历史产量和市场价格。"
        "需要制定最优的种植策略，以最大化预期收益。"
    )
    paper.set_data_source("题目附件1-4提供的历史种植数据和市场数据")

    paper.add_sub_problem("建立最优种植方案", "线性规划", "最优年收益 285.6 万元", "多目标优化模型")
    paper.add_sub_problem("预测未来产量和价格", "灰色预测 + ARIMA", "预测 R²=0.95", "组合预测模型")
    paper.add_sub_problem("风险分析与稳健决策", "蒙特卡洛模拟", "95%置信下≥250万", "随机优化模型")

    paper.add_assumption("假设题目中所给数据真实可靠")
    paper.add_assumption("假设气候条件与历史数据相似")
    paper.add_assumption("假设市场价格波动服从正态分布")
    paper.add_assumption("假设各地块的种植成本相同")

    paper.add_symbol("x_{ij}", "第i个地块种植第j种作物的面积", "亩")
    paper.add_symbol("p_j", "第j种作物的单位价格", "元/kg")
    paper.add_symbol("q_{ij}", "第i个地块种植第j种作物的单产", "kg/亩")
    paper.add_symbol("c_j", "第j种作物的单位成本", "元/亩")
    paper.add_symbol("Z", "总收益", "万元")

    paper.add_sensitivity_param("市场价格", 5.0, "作物价格波动")
    paper.add_sensitivity_param("产量", 400, "单产波动")
    paper.add_sensitivity_param("成本", 800, "种植成本变化")

    paper.set_keywords(["线性规划", "灰色预测", "蒙特卡洛", "灵敏度分析", "种植优化"])

    # 添加模型评价
    paper.add_evaluator(me)

    content = paper.generate("demo_paper.tex")
    print("  ✅ 论文模板已生成: demo_paper.tex")
    print(f"  总行数: {len(content.split(chr(10)))}")

    # 打印前30行预览
    print("\n  --- 论文预览 (前30行) ---")
    for i, line in enumerate(content.split("\n")[:30]):
        print(f"  {i+1:3d} | {line}")


if __name__ == "__main__":
    main()
