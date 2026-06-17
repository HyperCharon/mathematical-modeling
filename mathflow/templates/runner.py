"""
赛题模板运行器

内置历年经典题型的完整解题流程，一键运行。

Example:
    >>> from mathflow.templates import TemplateRunner
    >>> tr = TemplateRunner("evaluate")
    >>> tr.run()
"""

import numpy as np


class TemplateRunner:
    """
    赛题模板运行器.

    内置模板:
    - "evaluate": 综合评价类 (类 2020C 信贷决策)
    - "predict": 预测类 (类 2023C 蔬菜定价)
    - "optimize": 优化类 (类 2024B 生产决策)
    - "ode": 动力学建模 (类 2022A 波浪能)
    - "scheduling": 调度优化 (类 2022E 生产调度)
    - "classification": 分类问题 (类 2022C 古代玻璃)
    """

    def __init__(self, template_name: str):
        self.template_name = template_name
        self._templates = {
            "evaluate": self._run_evaluate,
            "predict": self._run_predict,
            "optimize": self._run_optimize,
            "ode": self._run_ode,
            "scheduling": self._run_scheduling,
            "classification": self._run_classification,
        }
        if template_name not in self._templates:
            raise ValueError(f"未知模板: {template_name}, 可选: {list(self._templates.keys())}")

    def run(self):
        """运行模板."""
        return self._templates[self.template_name]()

    def _run_evaluate(self):
        """综合评价类模板 (AHP + 熵权法 + TOPSIS)."""
        from mathflow.evaluate import AHP, TOPSIS, EntropyWeight
        from mathflow.stats import SensitivityAnalysis

        print("=" * 60)
        print("  模板: 综合评价类 (AHP + 熵权法 + TOPSIS)")
        print("  类似: 2020C 中小微企业信贷决策")
        print("=" * 60)

        # 模拟数据: 5家企业的信用评估
        # 指标: 资产负债率(成本), 流动比率(效益), 净资产收益率(效益), 营收增长率(效益)
        data = np.array([
            [45, 1.8, 12, 15],  # 企业A
            [60, 1.2, 8, 10],   # 企业B
            [35, 2.5, 18, 25],  # 企业C
            [55, 1.5, 10, 8],   # 企业D
            [40, 2.0, 15, 20],  # 企业E
        ])
        labels = ["企业A", "企业B", "企业C", "企业D", "企业E"]
        indicators = ["资产负债率", "流动比率", "净资产收益率", "营收增长率"]

        # AHP 主观权重
        ahp = AHP()
        ahp.set_matrix([
            [1,   1/3, 1/2, 2],
            [3,   1,   2,   4],
            [2,   1/2, 1,   3],
            [1/2, 1/4, 1/3, 1],
        ])
        ahp.fit()
        print("\n" + ahp.summary())

        # 熵权法客观权重
        ew = EntropyWeight(data, types=[-1, 1, 1, 1])
        ew.fit()
        print("\n" + ew.summary(labels=indicators))

        # 组合权重
        w = 0.5 * ahp.weights + 0.5 * ew.weights
        w = w / w.sum()
        print(f"\n  组合权重: {dict(zip(indicators, [f'{v:.4f}' for v in w]))}")

        # TOPSIS 排序
        topsis = TOPSIS(data, weights=w, types=[-1, 1, 1, 1])
        topsis.fit()
        print("\n" + topsis.summary(labels=labels))

        # 灵敏度分析
        def eval_score(perturb_w):
            tw = perturb_w / perturb_w.sum()
            t = TOPSIS(data, weights=tw, types=[-1, 1, 1, 1])
            t.fit()
            return t.scores[2]  # 企业C的得分

        sa = SensitivityAnalysis(eval_score, n_vars=4, var_names=indicators)
        sa.one_at_a_time(base_values=w, perturbation=0.2)
        print("\n" + sa.summary())

        return {"ahp": ahp, "entropy": ew, "topsis": topsis, "sensitivity": sa}

    def _run_predict(self):
        """预测类模板 (灰色预测 + ARIMA + 回归)."""
        from mathflow.predict import GreyPrediction, ExponentialSmoothing
        from mathflow.interpolate import CurveFitter

        print("=" * 60)
        print("  模板: 预测类 (灰色预测 + 指数平滑 + 曲线拟合)")
        print("  类似: 2023C 蔬菜定价与补货预测")
        print("=" * 60)

        # 模拟某蔬菜 12 个月的销量数据
        months = np.arange(1, 13)
        sales = np.array([120, 135, 150, 145, 160, 180, 200, 195, 175, 155, 140, 130])

        print(f"\n  历史数据: {list(sales)}")

        # 灰色预测
        print("\n--- GM(1,1) 灰色预测 ---")
        gp = GreyPrediction(sales)
        gp.fit()
        print(gp.summary(steps=3))

        # 指数平滑
        print("\n--- Holt 指数平滑 ---")
        es = ExponentialSmoothing(sales, method="holt", alpha=0.3, beta=0.1)
        es.fit()
        print(es.summary(steps=3))

        # 曲线拟合
        print("\n--- 自动曲线拟合 ---")
        cf = CurveFitter(months, sales)
        cf.auto_fit()
        print(cf.summary())

        # 预测对比
        print("\n--- 预测对比 (未来3个月) ---")
        gp_fc = gp.predict(steps=3)
        es_fc = es.predict(steps=3)
        cf_fc = cf.predict([13, 14, 15])
        for i in range(3):
            print(f"  第{13+i}月: 灰色={gp_fc[i]:.0f}, 指数平滑={es_fc[i]:.0f}, 曲线拟合={cf_fc[i]:.0f}")

        return {"grey": gp, "exp_smooth": es, "curve_fit": cf}

    def _run_optimize(self):
        """优化类模板 (LP + GA)."""
        from mathflow.optimize import LinearProgramming, GeneticAlgorithm

        print("=" * 60)
        print("  模板: 优化类 (线性规划 + 遗传算法)")
        print("  类似: 2024B 生产过程中的决策问题")
        print("=" * 60)

        # 生产计划问题
        # 产品A: 利润5, 耗时2h, 耗料3kg
        # 产品B: 利润8, 耗时3h, 耗料2kg
        # 产品C: 利润6, 耗时2h, 耗料4kg
        # 约束: 总工时≤40h, 总原料≤50kg

        print("\n--- 线性规划 (精确解) ---")
        lp = LinearProgramming()
        lp.set_objective([5, 8, 6], sense="max", var_names=["产品A", "产品B", "产品C"])
        lp.add_constraint([2, 3, 2], "<=", 40)  # 工时
        lp.add_constraint([3, 2, 4], "<=", 50)  # 原料
        result = lp.solve()
        print(lp.summary())

        print("\n--- 遗传算法 (验证) ---")
        def fitness(x):
            profit = 5*x[0] + 8*x[1] + 6*x[2]
            penalty = 0
            if 2*x[0] + 3*x[1] + 2*x[2] > 40:
                penalty += (2*x[0] + 3*x[1] + 2*x[2] - 40) * 100
            if 3*x[0] + 2*x[1] + 4*x[2] > 50:
                penalty += (3*x[0] + 2*x[1] + 4*x[2] - 50) * 100
            return profit - penalty

        ga = GeneticAlgorithm(fitness, n_vars=3, bounds=[(0, 20)]*3, pop_size=100, generations=300)
        ga_result = ga.run()
        print(ga.summary())

        return {"lp": lp, "ga": ga}

    def _run_ode(self):
        """动力学建模模板 (SIR + 灵敏度)."""
        from mathflow.ode import ODESolver, sir_model
        from mathflow.stats import SensitivityAnalysis

        print("=" * 60)
        print("  模板: 动力学建模 (SIR + 灵敏度分析)")
        print("  类似: 2022A 波浪能 / 2020A 炉温曲线")
        print("=" * 60)

        # SIR 模型参数扫描
        print("\n--- SIR 模型: 不同 R0 的影响 ---")
        for R0 in [1.5, 2.0, 2.5, 3.0, 4.0]:
            beta, gamma = 0.3, 0.3/R0
            solver = ODESolver(sir_model(beta=beta, gamma=gamma), y0=[0.99, 0.01, 0])
            result = solver.solve(t_span=(0, 200), dt=0.5)
            peak_I = result.y[:, 1].max()
            final_R = result.y[-1, 2]
            print(f"  R0={R0:.1f}: 峰值感染={peak_I:.2%}, 最终感染={final_R:.2%}")

        # 灵敏度分析
        print("\n--- 灵敏度分析 ---")
        def peak_infection(params):
            b, g = params
            if b <= 0 or g <= 0: return 0
            s = ODESolver(sir_model(beta=b, gamma=g), y0=[0.99, 0.01, 0])
            r = s.solve(t_span=(0, 200), dt=1)
            return r.y[:, 1].max()

        sa = SensitivityAnalysis(peak_infection, n_vars=2, var_names=["β(传染率)", "γ(恢复率)"])
        sa.one_at_a_time(base_values=[0.3, 0.1], perturbation=0.3)
        print(sa.summary())

        return {"sensitivity": sa}

    def _run_scheduling(self):
        """调度优化模板."""
        from mathflow.optimize import GeneticAlgorithm

        print("=" * 60)
        print("  模板: 调度优化 (TSP + GA)")
        print("  类似: 2022E 小批量物料调度 / 2024B 生产决策")
        print("=" * 60)

        # 8 个任务的调度 (最小化总完工时间)
        np.random.seed(42)
        processing_times = np.random.randint(5, 30, 8)
        print(f"\n  任务加工时间: {list(processing_times)}")

        def makespan_fitness(perm):
            """排列编码的 makespan 适应度."""
            order = np.argsort(perm)
            completion = 0
            for i in order:
                completion += processing_times[i]
            return -completion  # 最小化 makespan

        ga = GeneticAlgorithm(
            makespan_fitness, n_vars=8,
            bounds=[(0, 1)] * 8,
            pop_size=100, generations=200,
        )
        result = ga.run()
        best_order = np.argsort(result.best_solution)
        print(f"  最优调度顺序: {list(best_order)}")
        print(f"  最小总完工时间: {-result.best_fitness:.0f}")
        print(f"  各任务完成时间: {np.cumsum(processing_times[best_order]).tolist()}")

        return {"ga": ga}

    def _run_classification(self):
        """分类问题模板."""
        from mathflow.ml import Classifier, ClusterAnalysis, DimensionReduction

        print("=" * 60)
        print("  模板: 分类与聚类")
        print("  类似: 2022C 古代玻璃制品鉴别 / 2020C 信贷决策")
        print("=" * 60)

        # 使用 sklearn 内置数据集
        from sklearn.datasets import load_iris
        data = load_iris()
        X, y = data.data, data.target

        print(f"\n  数据集: Iris, {X.shape[0]} 样本, {X.shape[1]} 特征")
        print(f"  类别: {list(data.target_names)}")

        # 分类
        print("\n--- 多方法对比 ---")
        clf = Classifier(X, y, feature_names=data.feature_names, class_names=list(data.target_names))
        comparison = clf.compare_methods(["logistic", "knn", "svm", "random_forest"])
        print(comparison.to_string(index=False))

        # 最优模型详细结果
        best_method = comparison.iloc[0]["方法"]
        clf.fit(method=best_method)
        print(f"\n--- 最优模型: {best_method} ---")
        print(clf.summary())

        # 聚类
        print("\n--- 聚类分析 ---")
        ca = ClusterAnalysis(X)
        best_k = ca.auto_k(max_k=8)
        print(f"  自动选择 K = {best_k}")
        ca.fit(method="kmeans", n_clusters=best_k)
        print(ca.summary())

        # 降维
        print("\n--- PCA 降维 ---")
        dr = DimensionReduction(X)
        dr.fit(n_components=0.95)
        print(dr.summary(feature_names=data.feature_names))

        return {"classifier": clf, "cluster": ca, "pca": dr}


def list_templates():
    """列出所有可用模板."""
    templates = {
        "evaluate": "综合评价类 (AHP + 熵权法 + TOPSIS + 灵敏度)",
        "predict": "预测类 (灰色预测 + 指数平滑 + 曲线拟合)",
        "optimize": "优化类 (线性规划 + 遗传算法)",
        "ode": "动力学建模 (SIR + 灵敏度分析)",
        "scheduling": "调度优化 (TSP + GA)",
        "classification": "分类与聚类 (多方法对比 + PCA)",
    }
    print("\n可用模板:")
    for name, desc in templates.items():
        print(f"  {name:20s} — {desc}")
    return templates
