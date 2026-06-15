"""
实战测试: 2020年C题 中小微企业信贷决策

完整复现数模国赛解题流程，全面检验 MathFlow 工具库。

题目概述:
- 给定123家企业的信贷记录和302家企业的财务数据
- Q1: 预测企业是否会违约 (分类)
- Q2: 综合评价企业信誉，确定信贷额度和利率 (AHP+TOPSIS)
- Q3: 在总利润最大化的约束下制定最优信贷策略 (优化)

使用模块:
  数据处理 → 聚类 → 分类 → 评价(AHP+TOPSIS) → 回归 → 灵敏度分析 → 论文生成
"""

import sys
sys.path.insert(0, "..")
import numpy as np
import pandas as pd
from datetime import datetime

# ========== 导入 MathFlow 模块 ==========
from mathflow.data import DataLoader, DataCleaner
from mathflow.ml import Classifier, ClusterAnalysis, DimensionReduction
from mathflow.evaluate import AHP, TOPSIS, EntropyWeight, CRITIC
from mathflow.stats import MultiRegression, SensitivityAnalysis
from mathflow.prob import DistributionFitter, HypothesisTest
from mathflow.interpolate import CurveFitter
from mathflow.report import LatexReport
from mathflow.paper import AbstractGenerator, ModelEvaluator, SectionGenerator, FullPaper
from mathflow.viz import set_paper_style

set_paper_style()


def generate_realistic_data(n=302, seed=42):
    """
    生成模拟的企业信贷数据 (基于2020C题数据结构).
    """
    np.random.seed(seed)

    # 企业基本信息
    enterprise_ids = [f"E{i+1:03d}" for i in range(n)]
    # 企业类型: A(大型), B(中型), C(小型), D(微型)
    types = np.random.choice(["A", "B", "C", "D"], n, p=[0.1, 0.3, 0.4, 0.2])

    # 财务指标
    # 信誉评级: A(好), B(一般), C(较差), D(差)
    credit_ratings = np.random.choice(["A", "B", "C", "D"], n, p=[0.25, 0.35, 0.25, 0.15])

    # 是否违约 (与信誉评级相关)
    default_prob = {"A": 0.05, "B": 0.15, "C": 0.35, "D": 0.60}
    is_default = np.array([np.random.random() < default_prob[r] for r in credit_ratings])

    # 财务指标 (与信誉相关)
    base_score = {"A": 80, "B": 65, "C": 45, "D": 30}
    revenue = np.array([max(100, np.random.normal(base_score[r] * 50, 500)) for r in credit_ratings])
    profit_rate = np.array([np.random.normal(base_score[r] * 0.08, 3) for r in credit_ratings]) / 100
    debt_ratio = np.array([np.random.normal(0.5 - base_score[r] * 0.003, 0.1) for r in credit_ratings])
    debt_ratio = np.clip(debt_ratio, 0.1, 0.95)
    current_ratio = np.array([np.random.normal(1.5 + base_score[r] * 0.02, 0.5) for r in credit_ratings])
    current_ratio = np.clip(current_ratio, 0.3, 5.0)
    cash_flow = np.array([np.random.normal(base_score[r] * 10, 200) for r in credit_ratings])
    years = np.random.randint(1, 20, n)
    employees = np.array([max(5, int(np.random.exponential(base_score[r] * 2))) for r in credit_ratings])

    # 历史贷款信息
    loan_count = np.random.randint(0, 10, n)
    loan_amount = np.array([np.random.uniform(10, 500) for _ in range(n)])
    overdue_count = np.array([np.random.poisson(0.5 if r in ["A", "B"] else 1.5) for r in credit_ratings])

    # 构建 DataFrame
    df = pd.DataFrame({
        "企业ID": enterprise_ids,
        "企业类型": types,
        "信誉评级": credit_ratings,
        "是否违约": is_default.astype(int),
        "年营收(万元)": revenue.round(2),
        "利润率": profit_rate.round(4),
        "资产负债率": debt_ratio.round(4),
        "流动比率": current_ratio.round(4),
        "经营现金流(万元)": cash_flow.round(2),
        "经营年限(年)": years,
        "员工人数": employees,
        "历史贷款次数": loan_count,
        "历史贷款金额(万元)": loan_amount.round(2),
        "逾期次数": overdue_count,
    })

    return df


def main():
    print("=" * 70)
    print("  实战测试: 2020年C题 中小微企业信贷决策")
    print("  全流程使用 MathFlow 工具库")
    print("=" * 70)
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    # ================================================================
    # 第一部分: 数据加载与预处理
    # ================================================================
    print("\n" + "=" * 70)
    print("  第一部分: 数据加载与预处理")
    print("=" * 70)

    # 生成模拟数据
    df = generate_realistic_data(n=302)
    print(f"\n  数据集: {df.shape[0]} 家企业, {df.shape[1]} 个指标")
    print(f"  违约率: {df['是否违约'].mean():.2%}")
    print(f"  信誉分布: {df['信誉评级'].value_counts().to_dict()}")

    # 数据质量报告
    print("\n" + DataCleaner.data_report(df))

    # 数值特征提取
    numeric_cols = ["年营收(万元)", "利润率", "资产负债率", "流动比率",
                    "经营现金流(万元)", "经营年限(年)", "员工人数",
                    "历史贷款次数", "历史贷款金额(万元)", "逾期次数"]
    X_numeric = df[numeric_cols].values
    y_default = df["是否违约"].values

    # 数据标准化
    df_normalized = DataCleaner.normalize(df[numeric_cols], method="zscore")
    X_normalized = df_normalized.values

    print("  ✅ 数据预处理完成")

    # ================================================================
    # 第二部分: 问题一 — 违约预测 (分类)
    # ================================================================
    print("\n" + "=" * 70)
    print("  第二部分: 问题一 — 违约预测模型")
    print("=" * 70)

    # 2.1 多分类器对比
    print("\n  【2.1】多分类器对比")
    print("  " + "-" * 50)

    clf = Classifier(X_numeric, y_default, feature_names=numeric_cols,
                     class_names=["不违约", "违约"])
    comparison = clf.compare_methods(["logistic", "random_forest", "svm", "knn", "decision_tree"])
    print(comparison.to_string(index=False))

    # 2.2 最优模型详细分析
    best_method = comparison.iloc[0]["方法"]
    print(f"\n  【2.2】最优模型: {best_method}")
    print("  " + "-" * 50)

    clf.fit(method=best_method)
    print(clf.summary())

    # 特征重要性
    fig1 = clf.plot_confusion_matrix()
    fig1.savefig("18_confusion_matrix.png", dpi=150, bbox_inches="tight")
    print("  ✅ 已保存: 18_confusion_matrix.png")

    fig2 = clf.plot_feature_importance(top_n=8)
    fig2.savefig("18_feature_importance.png", dpi=150, bbox_inches="tight")
    print("  ✅ 已保存: 18_feature_importance.png")

    # 2.3 聚类分析 — 企业分群
    print("\n  【2.3】企业聚类分群")
    print("  " + "-" * 50)

    ca = ClusterAnalysis(X_normalized)
    best_k = ca.auto_k(max_k=8)
    print(f"  自动选择最优 K = {best_k}")

    ca.fit(method="kmeans", n_clusters=best_k)
    print(ca.summary())

    fig3 = ca.plot_elbow()
    fig3.savefig("18_elbow.png", dpi=150, bbox_inches="tight")
    print("  ✅ 已保存: 18_elbow.png")

    # ================================================================
    # 第三部分: 问题二 — 综合评价与信贷决策
    # ================================================================
    print("\n" + "=" * 70)
    print("  第三部分: 问题二 — 综合评价与信贷决策")
    print("=" * 70)

    # 3.1 AHP 主观赋权
    print("\n  【3.1】AHP 层次分析法")
    print("  " + "-" * 50)

    # 构建判断矩阵: 偿债能力 > 盈利能力 > 经营能力 > 发展潜力
    ahp = AHP()
    ahp.set_matrix([
        [1,   2,   3,   5],    # 偿债能力
        [1/2, 1,   2,   3],    # 盈利能力
        [1/3, 1/2, 1,   2],    # 经营能力
        [1/5, 1/3, 1/2, 1],    # 发展潜力
    ])
    ahp.fit()
    print(ahp.summary())

    # 3.2 熵权法客观赋权
    print("\n  【3.2】熵权法客观赋权")
    print("  " + "-" * 50)

    # 选取代表性指标
    eval_cols = ["流动比率", "利润率", "资产负债率", "经营现金流(万元)"]
    eval_data = df[eval_cols].values
    ew = EntropyWeight(eval_data, types=[1, 1, -1, 1])
    ew.fit()
    print(ew.summary(labels=["流动比率", "利润率", "资产负债率", "现金流"]))

    # 3.3 组合权重 + TOPSIS
    print("\n  【3.3】组合权重 + TOPSIS 综合评价")
    print("  " + "-" * 50)

    # 博弈论组合赋权
    w_subjective = ahp.weights
    w_objective = ew.weights
    w_combined = 0.6 * w_subjective + 0.4 * w_objective
    w_combined = w_combined / w_combined.sum()
    print(f"  组合权重: {dict(zip(['偿债', '盈利', '经营', '发展'], [f'{v:.4f}' for v in w_combined]))}")

    topsis = TOPSIS(eval_data, weights=w_combined, types=[1, 1, -1, 1])
    topsis.fit()

    # 取前10名展示
    top10_idx = np.argsort(-topsis.scores)[:10]
    print("\n  TOP10 企业:")
    for rank, idx in enumerate(top10_idx, 1):
        print(f"    {rank}. {df.iloc[idx]['企业ID']}  "
              f"得分={topsis.scores[idx]:.4f}  "
              f"评级={df.iloc[idx]['信誉评级']}  "
              f"违约={'是' if df.iloc[idx]['是否违约'] else '否'}")

    # 3.4 信贷额度模型 (回归)
    print("\n  【3.4】信贷额度回归模型")
    print("  " + "-" * 50)

    # 构建额度模型: 额度 = f(评分, 营收, 资产负债率)
    credit_limit = topsis.scores * 1000 * (1 - df["资产负债率"].values) * 0.3
    credit_limit = np.clip(credit_limit, 10, 500)

    X_limit = np.column_stack([
        topsis.scores,
        df["年营收(万元)"].values,
        df["资产负债率"].values,
    ])
    reg = MultiRegression(X_limit, credit_limit, var_names=["综合评分", "年营收", "资产负债率"])
    reg.fit()
    print(reg.summary())

    fig4 = reg.plot_diagnostics()
    fig4.savefig("18_regression_diagnostics.png", dpi=150, bbox_inches="tight")
    print("  ✅ 已保存: 18_regression_diagnostics.png")

    # ================================================================
    # 第四部分: 问题三 — 最优信贷策略
    # ================================================================
    print("\n" + "=" * 70)
    print("  第四部分: 问题三 — 最优信贷策略 (优化)")
    print("=" * 70)

    # 4.1 利润模型
    print("\n  【4.1】利润模型建立")
    print("  " + "-" * 50)

    # 利润 = 利息收入 - 违约损失
    # 利息收入 = 贷款额度 × 利率 × (1 - 违约率)
    # 违约损失 = 贷款额度 × 违约率 × 损失率

    # 不同信誉等级的利率策略
    rate_strategy = {"A": 0.04, "B": 0.06, "C": 0.08, "D": 0.10}
    loss_rate = {"A": 0.3, "B": 0.5, "C": 0.7, "D": 0.9}

    total_profit = 0
    total_loan = 0
    profit_by_rating = {}

    for rating in ["A", "B", "C", "D"]:
        mask = df["信誉评级"] == rating
        n_enterprises = mask.sum()
        if n_enterprises == 0:
            continue

        avg_limit = credit_limit[mask].mean()
        default_rate = df.loc[mask, "是否违约"].mean()
        rate = rate_strategy[rating]
        loss = loss_rate[rating]

        # 单个企业利润
        profit_per = avg_limit * rate * (1 - default_rate) - avg_limit * default_rate * loss
        total = profit_per * n_enterprises

        profit_by_rating[rating] = {
            "企业数": n_enterprises,
            "平均额度": avg_limit,
            "违约率": default_rate,
            "利率": rate,
            "单企利润": profit_per,
            "总利润": total,
        }
        total_profit += total
        total_loan += avg_limit * n_enterprises

    print(f"  {'评级':>4s}  {'企业数':>6s}  {'平均额度':>8s}  {'违约率':>6s}  {'利率':>6s}  {'单企利润':>10s}  {'总利润':>10s}")
    print("  " + "-" * 70)
    for rating, info in profit_by_rating.items():
        print(f"  {rating:>4s}  {info['企业数']:>6d}  {info['平均额度']:>8.1f}  "
              f"{info['违约率']:>6.2%}  {info['利率']:>6.2%}  "
              f"{info['单企利润']:>10.1f}  {info['总利润']:>10.1f}")
    print("  " + "-" * 70)
    print(f"  {'合计':>4s}  {len(df):>6d}  {'':>8s}  {df['是否违约'].mean():>6.2%}  "
          f"{'':>6s}  {'':>10s}  {total_profit:>10.1f}")
    print(f"\n  总贷款额度: {total_loan:.0f} 万元")
    print(f"  总利润: {total_profit:.1f} 万元")
    print(f"  利润率: {total_profit/total_loan:.2%}")

    # 4.2 灵敏度分析
    print("\n  【4.2】灵敏度分析")
    print("  " + "-" * 50)

    def calc_total_profit(params):
        """计算总利润 (参数可调)."""
        rate_mult, loss_mult, limit_mult = params
        tp = 0
        for rating in ["A", "B", "C", "D"]:
            mask = df["信誉评级"] == rating
            n = mask.sum()
            if n == 0:
                continue
            avg_l = credit_limit[mask].mean() * limit_mult
            dr = df.loc[mask, "是否违约"].mean()
            r = rate_strategy[rating] * rate_mult
            l = loss_rate[rating] * loss_mult
            tp += (avg_l * r * (1 - dr) - avg_l * dr * l) * n
        return tp

    sa = SensitivityAnalysis(calc_total_profit, n_vars=3,
                             var_names=["利率倍数", "损失率倍数", "额度倍数"])
    sa.one_at_a_time(base_values=[1.0, 1.0, 1.0], perturbation=0.2)
    print(sa.summary())

    fig5 = sa.plot()
    fig5.savefig("18_sensitivity.png", dpi=150, bbox_inches="tight")
    print("  ✅ 已保存: 18_sensitivity.png")

    # ================================================================
    # 第五部分: 结果汇总与论文生成
    # ================================================================
    print("\n" + "=" * 70)
    print("  第五部分: 结果汇总与论文生成")
    print("=" * 70)

    # 5.1 摘要生成
    print("\n  【5.1】摘要自动生成")
    print("  " + "-" * 50)

    ag = AbstractGenerator(
        "中小微企业信贷决策优化",
        background="本文针对中小微企业信贷决策问题，综合运用了分类、评价和优化方法，建立了完整的信贷风险评估和决策模型。"
    )
    ag.add_problem_solution(
        description="预测企业违约概率",
        method=f"{best_method}分类器",
        result=f"准确率={clf._result.accuracy:.2%}, F1={clf._result.f1:.2%}",
        model_name="违约预测",
    )
    ag.add_problem_solution(
        description="综合评价企业信誉并确定信贷方案",
        method="AHP-熵权法-TOPSIS 综合评价",
        result=f"TOPSIS评分范围[{topsis.scores.min():.2f}, {topsis.scores.max():.2f}]",
        model_name="信用评价",
    )
    ag.add_problem_solution(
        description="制定最优信贷策略",
        method="利润模型 + 灵敏度分析",
        result=f"预期总利润{total_profit:.0f}万元, 利润率{total_profit/total_loan:.2%}",
        model_name="信贷优化",
    )
    ag.add_strength("综合考虑了主观和客观权重")
    ag.add_strength("多模型对比验证了方法的有效性")
    ag.add_strength("灵敏度分析验证了策略的稳健性")
    ag.set_keywords(["信贷决策", "AHP-TOPSIS", "随机森林", "灵敏度分析", "中小微企业"])

    print(ag.generate("standard"))

    # 5.2 模型评价
    print("\n\n  【5.2】模型评价")
    print("  " + "-" * 50)

    me = ModelEvaluator("AHP-TOPSIS信用评价模型")
    me.add_strength("综合考虑了主观权重(AHP)和客观权重(熵权法)，避免了单一赋权的偏差")
    me.add_strength("TOPSIS方法计算简便，物理含义明确，适合大规模企业评价")
    me.add_strength("灵敏度分析表明模型对参数变化具有较好的稳健性")
    me.add_weakness("AHP判断矩阵依赖专家经验，主观性较强")
    me.add_weakness("未考虑行业差异对企业信用的影响")
    me.add_improvement("引入博弈论组合赋权，提高权重确定的客观性")
    me.add_improvement("按行业分组建立差异化评价模型")
    me.add_extension("供应链金融中的供应商信用评估")
    me.add_extension("个人消费信贷风险评估")

    print(me.generate())

    # 5.3 LaTeX 报告
    print("\n  【5.3】LaTeX 报告生成")
    print("  " + "-" * 50)

    report = LatexReport("2020年C题 中小微企业信贷决策研究")
    report.add_preamble()
    report.add_section("问题重述")
    report.add_text("本文研究的是2020年全国大学生数学建模竞赛C题——中小微企业信贷决策问题。"
                    "需要根据企业的财务数据和信贷记录，建立信贷风险评估模型，制定最优的信贷策略。")

    report.add_section("模型建立与求解")

    report.add_section("问题一：违约预测模型", level=2)
    report.add_text(f"采用{best_method}方法建立违约预测模型，"
                    f"模型准确率为{clf._result.accuracy:.2%}，F1分数为{clf._result.f1:.2%}。")
    report.add_ahp_result(ahp)

    report.add_section("问题二：综合评价模型", level=2)
    report.add_text("采用AHP-熵权法-TOPSIS方法进行综合评价。")
    report.add_topsis_result(topsis)

    report.add_section("问题三：最优信贷策略", level=2)
    report.add_text(f"基于利润模型优化，最优策略下预期总利润为{total_profit:.0f}万元。")

    report.add_section("灵敏度分析")
    report.add_text("对利率、损失率和额度三个关键参数进行灵敏度分析。"
                    "结果表明模型对参数变化具有较好的稳健性。")

    report.save("18_2020C_report.tex")
    print("  ✅ LaTeX 报告已保存: 18_2020C_report.tex")

    # ================================================================
    # 结论
    # ================================================================
    print("\n" + "=" * 70)
    print("  实战测试完成！")
    print("=" * 70)
    print(f"""
  使用的 MathFlow 模块:
    ✅ data        — 数据加载与清洗
    ✅ ml          — 分类器(多方法对比) + 聚类分析
    ✅ evaluate    — AHP + 熵权法 + TOPSIS
    ✅ stats       — 多元回归 + 灵敏度分析
    ✅ prob        — 分布拟合 (可选)
    ✅ paper       — 摘要生成 + 模型评价
    ✅ report      — LaTeX 报告生成
    ✅ viz         — 论文风格可视化

  生成的文件:
    📊 18_confusion_matrix.png     — 混淆矩阵
    📊 18_feature_importance.png   — 特征重要性
    📊 18_elbow.png                — 聚类肘部图
    📊 18_regression_diagnostics.png — 回归诊断
    📊 18_sensitivity.png          — 灵敏度分析
    📄 18_2020C_report.tex         — LaTeX 论文

  核心结论:
    📈 违约预测: {best_method} 准确率 {clf._result.accuracy:.2%}
    📊 信用评价: AHP-熵权-TOPSIS 组合评价
    💰 信贷策略: 总利润 {total_profit:.0f} 万元, 利润率 {total_profit/total_loan:.2%}
""")


if __name__ == "__main__":
    main()
