#!/usr/bin/env python3
"""
新功能测试示例：元胞自动机、DEA、相关性分析、非线性规划、双因素方差分析
"""

import numpy as np
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_cellular_automata():
    """测试元胞自动机 - 传染病扩散模拟."""
    print("=" * 60)
    print("测试1: 元胞自动机 - SIR传染病扩散模拟")
    print("=" * 60)

    from mathflow.simulation import CellularAutomata

    # 创建50x50网格的SIR模型
    ca = CellularAutomata(grid_size=(50, 50), n_states=3)
    ca.add_rule("SIR", infection_rate=0.3, recovery_rate=0.05)

    # 在中心区域初始化感染
    ca.initialize(pattern="center", radius=5)

    print(f"初始感染人数: {np.sum(ca.grid == 1)}")
    print(f"初始易感人数: {np.sum(ca.grid == 0)}")

    # 运行50步模拟
    result = ca.run(steps=50, verbose=True)

    print(f"\n最终状态:")
    print(f"  易感者(S): {result.stats['state_0'][-1]}")
    print(f"  感染者(I): {result.stats['state_1'][-1]}")
    print(f"  恢复者(R): {result.stats['state_2'][-1]}")

    # 绘制结果
    fig = ca.plot()
    fig.savefig("ca_sir_result.png", dpi=150, bbox_inches="tight")
    print(f"\n结果已保存到 ca_sir_result.png")

    return result


def test_dea():
    """测试DEA - 学校效率评估."""
    print("\n" + "=" * 60)
    print("测试2: DEA数据包络分析 - 学校效率评估")
    print("=" * 60)

    from mathflow.evaluate import DEA

    # 5所学校的投入产出数据
    # 投入: 教师数, 教育经费(万元)
    # 产出: 学生平均成绩
    inputs = np.array([
        [20, 500],   # 学校A
        [25, 600],   # 学校B
        [15, 400],   # 学校C
        [30, 700],   # 学校D
        [18, 450],   # 学校E
    ])

    outputs = np.array([
        [85],   # 学校A平均成绩
        [88],   # 学校B
        [82],   # 学校C
        [90],   # 学校D
        [84],   # 学校E
    ])

    school_names = ["学校A", "学校B", "学校C", "学校D", "学校E"]

    # CCR模型分析
    dea = DEA(inputs, outputs, model="CCR", orientation="input")
    result = dea.fit()

    print("\n效率评估结果:")
    for i, (name, eff, is_eff) in enumerate(zip(school_names, result.efficiencies, result.is_efficient)):
        status = "✅ 有效" if is_eff else "❌ 无效"
        print(f"  {name}: 效率={eff:.4f} {status}")

    # 绘制结果
    fig = dea.plot()
    fig.savefig("dea_result.png", dpi=150, bbox_inches="tight")
    print(f"\n结果已保存到 dea_result.png")

    return result


def test_correlation():
    """测试相关性分析 - 学生成绩相关性."""
    print("\n" + "=" * 60)
    print("测试3: 相关性分析 - 学生成绩相关性")
    print("=" * 60)

    from mathflow.stats import CorrelationAnalysis

    np.random.seed(42)

    # 模拟100名学生的4科成绩
    n_students = 100

    # 数学成绩 (基础变量)
    math = np.random.normal(75, 10, n_students)

    # 物理与数学高度相关
    physics = math * 0.8 + np.random.normal(15, 5, n_students)

    # 英语与数学弱相关
    english = np.random.normal(70, 12, n_students)

    # 化学与数学中等相关
    chemistry = math * 0.5 + np.random.normal(30, 8, n_students)

    data = np.column_stack([math, physics, english, chemistry])

    ca = CorrelationAnalysis(data, var_names=["数学", "物理", "英语", "化学"])
    result = ca.fit(method="pearson")

    print("\n相关系数矩阵:")
    print(f"  {'':>8s}", end="")
    for name in ca.var_names:
        print(f"  {name:>8s}", end="")
    print()

    for i, name in enumerate(ca.var_names):
        print(f"  {name:>8s}", end="")
        for j in range(len(ca.var_names)):
            print(f"  {result.corr_matrix[i, j]:>8.3f}", end="")
        print()

    print("\n显著相关的变量对 (p<0.05):")
    pairs = ca.get_significant_pairs(alpha=0.05)
    for name1, name2, corr, p in pairs:
        print(f"  {name1} - {name2}: r={corr:.4f}, p={p:.4f}")

    # 绘制热力图
    fig = ca.plot()
    fig.savefig("correlation_result.png", dpi=150, bbox_inches="tight")
    print(f"\n结果已保存到 correlation_result.png")

    return result


def test_nonlinear():
    """测试非线性规划 - 工程优化问题."""
    print("\n" + "=" * 60)
    print("测试4: 非线性规划 - 容器设计优化")
    print("=" * 60)

    from mathflow.optimize import NonlinearProgramming

    # 问题: 设计一个圆柱形容器，容积固定为1000cm³
    # 目标: 最小化表面积 (节省材料)
    # 变量: x[0]=半径r, x[1]=高h
    # 约束: π*r²*h = 1000

    nlp = NonlinearProgramming()

    # 目标函数: 表面积 S = 2πr² + 2πrh
    def objective(x):
        r, h = x
        return 2 * np.pi * r**2 + 2 * np.pi * r * h

    # 等式约束: π*r²*h - 1000 = 0
    def volume_constraint(x):
        r, h = x
        return np.pi * r**2 * h - 1000

    nlp.set_objective(objective)
    nlp.add_constraint(volume_constraint, type="eq")
    nlp.set_bounds([(0.1, 100), (0.1, 100)])  # 正数约束

    # 求解
    result = nlp.solve(x0=[5, 50], method="SLSQP")

    r_opt, h_opt = result.solution
    V_opt = np.pi * r_opt**2 * h_opt
    S_opt = result.optimal_value

    print(f"\n优化结果:")
    print(f"  最优半径: r = {r_opt:.4f} cm")
    print(f"  最优高度: h = {h_opt:.4f} cm")
    print(f"  容积验证: V = {V_opt:.4f} cm³ (目标: 1000)")
    print(f"  最小表面积: S = {S_opt:.4f} cm²")
    print(f"  求解状态: {'成功' if result.success else '失败'}")

    # 理论最优解: r = (500/π)^(1/3), h = 2r
    r_theory = (500 / np.pi) ** (1/3)
    h_theory = 2 * r_theory
    print(f"\n理论最优解:")
    print(f"  r = {r_theory:.4f} cm")
    print(f"  h = {h_theory:.4f} cm")

    return result


def test_two_way_anova():
    """测试双因素方差分析 - 药物疗效分析."""
    print("\n" + "=" * 60)
    print("测试5: 双因素方差分析 - 药物疗效分析")
    print("=" * 60)

    from mathflow.stats import ANOVA

    # 问题: 研究两种药物(A,B)在三种剂量(低,中,高)下的疗效
    # 每种组合有4个重复观测

    np.random.seed(42)

    # 数据: [药物A低, 药物A中, 药物A高], [药物B低, 药物B中, 药物B高]
    data = np.array([
        # 药物A
        [10, 12, 15, 11],   # 低剂量
        [15, 18, 20, 16],   # 中剂量
        [20, 22, 25, 21],   # 高剂量
    ])

    # 计算描述统计
    drug_names = ["药物A-低", "药物A-中", "药物A-高"]
    print("\n各组描述统计:")
    for i, name in enumerate(drug_names):
        print(f"  {name}: 均值={data[i].mean():.2f}, 标准差={data[i].std():.2f}")

    # 单因素方差分析
    anova = ANOVA()
    anova.one_way(data[0], data[1], data[2], group_names=drug_names)

    print("\n单因素方差分析结果:")
    print(anova.summary())

    return anova


def test_game_of_life():
    """测试元胞自动机 - Game of Life."""
    print("\n" + "=" * 60)
    print("测试6: 元胞自动机 - Game of Life 滑翔机")
    print("=" * 60)

    from mathflow.simulation import CellularAutomata

    # 创建30x30网格
    ca = CellularAutomata(grid_size=(30, 30), n_states=2)
    ca.add_rule("game_of_life")

    # 初始化滑翔机
    ca.initialize(pattern="glider")

    print(f"初始活细胞数: {np.sum(ca.grid == 1)}")

    # 运行20步
    result = ca.run(steps=20)

    print(f"20步后活细胞数: {np.sum(ca.grid == 1)}")

    # 绘制最终状态
    fig = ca.plot()
    fig.savefig("game_of_life_result.png", dpi=150, bbox_inches="tight")
    print(f"结果已保存到 game_of_life_result.png")

    return result


def test_forest_fire():
    """测试元胞自动机 - 森林火灾."""
    print("\n" + "=" * 60)
    print("测试7: 元胞自动机 - 森林火灾模拟")
    print("=" * 60)

    from mathflow.simulation import CellularAutomata

    # 创建40x40网格
    ca = CellularAutomata(grid_size=(40, 40), n_states=3)
    ca.add_rule("forest_fire", grow_rate=0.02, spread_rate=0.5)

    # 初始化: 大部分是树木，少量空地
    grid = np.ones((40, 40), dtype=int)  # 全是树木
    grid[0:5, 0:5] = 0  # 左上角空地
    grid[19:21, 19:21] = 2  # 中心着火
    ca.set_grid(grid)

    print(f"初始状态:")
    print(f"  空地: {np.sum(ca.grid == 0)}")
    print(f"  树木: {np.sum(ca.grid == 1)}")
    print(f"  燃烧: {np.sum(ca.grid == 2)}")

    # 运行30步
    result = ca.run(steps=30)

    print(f"\n30步后:")
    print(f"  空地: {result.stats['state_0'][-1]}")
    print(f"  树木: {result.stats['state_1'][-1]}")
    print(f"  燃烧: {result.stats['state_2'][-1]}")

    # 绘制结果
    fig = ca.plot()
    fig.savefig("forest_fire_result.png", dpi=150, bbox_inches="tight")
    print(f"结果已保存到 forest_fire_result.png")

    return result


def test_dea_bcc():
    """测试DEA - BCC模型."""
    print("\n" + "=" * 60)
    print("测试8: DEA BCC模型 - 餐厅效率评估")
    print("=" * 60)

    from mathflow.evaluate import DEA

    # 6家餐厅的投入产出数据
    # 投入: 员工数, 座位数
    # 产出: 日均营业额(万元)
    inputs = np.array([
        [10, 50],   # 餐厅A
        [15, 80],   # 餐厅B
        [8, 40],    # 餐厅C
        [20, 100],  # 餐厅D
        [12, 60],   # 餐厅E
        [18, 90],   # 餐厅F
    ])

    outputs = np.array([
        [5],    # 餐厅A
        [8],    # 餐厅B
        [4],    # 餐厅C
        [10],   # 餐厅D
        [6],    # 餐厅E
        [9],    # 餐厅F
    ])

    restaurant_names = ["餐厅A", "餐厅B", "餐厅C", "餐厅D", "餐厅E", "餐厅F"]

    # CCR模型
    print("\nCCR模型 (规模报酬不变):")
    dea_ccr = DEA(inputs, outputs, model="CCR", orientation="input")
    result_ccr = dea_ccr.fit()

    for i, (name, eff) in enumerate(zip(restaurant_names, result_ccr.efficiencies)):
        status = "✅" if result_ccr.is_efficient[i] else "❌"
        print(f"  {name}: 效率={eff:.4f} {status}")

    return result_ccr


def test_correlation_methods():
    """测试不同相关系数方法."""
    print("\n" + "=" * 60)
    print("测试9: 不同相关系数方法对比")
    print("=" * 60)

    from mathflow.stats import CorrelationAnalysis

    np.random.seed(42)

    # 创建非线性关系数据
    n = 100
    x = np.linspace(0, 10, n)
    y_linear = 2 * x + np.random.normal(0, 1, n)
    y_quadratic = x**2 + np.random.normal(0, 5, n)
    y_log = np.log(x + 1) + np.random.normal(0, 0.3, n)

    data = np.column_stack([x, y_linear, y_quadratic, y_log])
    var_names = ["X", "线性Y", "二次Y", "对数Y"]

    # 测试三种方法
    for method in ["pearson", "spearman", "kendall"]:
        ca = CorrelationAnalysis(data, var_names=var_names)
        result = ca.fit(method=method)

        print(f"\n{method.title()} 相关系数:")
        print(f"  X与线性Y: {result.corr_matrix[0, 1]:.4f}")
        print(f"  X与二次Y: {result.corr_matrix[0, 2]:.4f}")
        print(f"  X与对数Y: {result.corr_matrix[0, 3]:.4f}")

    return result


def test_nonlinear_constrained():
    """测试带多个约束的非线性规划."""
    print("\n" + "=" * 60)
    print("测试10: 多约束非线性规划 - 投资组合优化")
    print("=" * 60)

    from mathflow.optimize import NonlinearProgramming

    # 问题: 投资3种资产，最小化风险(方差)
    # 约束1: 期望收益率 >= 10%
    # 约束2: 权重之和 = 1
    # 约束3: 每个权重 >= 0

    np.random.seed(42)

    # 模拟收益率数据
    n_assets = 3
    returns = np.array([0.12, 0.08, 0.15])  # 期望收益率
    cov_matrix = np.array([
        [0.04, 0.006, 0.01],
        [0.006, 0.025, 0.008],
        [0.01, 0.008, 0.06]
    ])

    nlp = NonlinearProgramming()

    # 目标函数: 最小化组合方差 w'Σw
    def portfolio_variance(w):
        return w @ cov_matrix @ w

    # 约束1: 期望收益率 >= 10%
    def return_constraint(w):
        return w @ returns - 0.10

    # 约束2: 权重之和 = 1
    def weight_constraint(w):
        return np.sum(w) - 1

    nlp.set_objective(portfolio_variance)
    nlp.add_constraint(return_constraint, type="ineq")
    nlp.add_constraint(weight_constraint, type="eq")
    nlp.set_bounds([(0, 1), (0, 1), (0, 1)])

    result = nlp.solve(x0=[1/3, 1/3, 1/3], method="SLSQP")

    print(f"\n投资组合优化结果:")
    print(f"  资产1权重: {result.solution[0]:.4f}")
    print(f"  资产2权重: {result.solution[1]:.4f}")
    print(f"  资产3权重: {result.solution[2]:.4f}")
    print(f"  组合方差: {result.optimal_value:.6f}")
    print(f"  组合标准差: {np.sqrt(result.optimal_value):.4f}")
    print(f"  期望收益率: {result.solution @ returns:.4f}")
    print(f"  求解状态: {'成功' if result.success else '失败'}")

    return result


if __name__ == "__main__":
    print("MathFlow 新功能测试")
    print("=" * 60)

    # 运行所有测试
    results = {}

    try:
        results["ca_sir"] = test_cellular_automata()
    except Exception as e:
        print(f"测试1失败: {e}")

    try:
        results["dea"] = test_dea()
    except Exception as e:
        print(f"测试2失败: {e}")

    try:
        results["correlation"] = test_correlation()
    except Exception as e:
        print(f"测试3失败: {e}")

    try:
        results["nonlinear"] = test_nonlinear()
    except Exception as e:
        print(f"测试4失败: {e}")

    try:
        results["anova"] = test_two_way_anova()
    except Exception as e:
        print(f"测试5失败: {e}")

    try:
        results["game_of_life"] = test_game_of_life()
    except Exception as e:
        print(f"测试6失败: {e}")

    try:
        results["forest_fire"] = test_forest_fire()
    except Exception as e:
        print(f"测试7失败: {e}")

    try:
        results["dea_bcc"] = test_dea_bcc()
    except Exception as e:
        print(f"测试8失败: {e}")

    try:
        results["correlation_methods"] = test_correlation_methods()
    except Exception as e:
        print(f"测试9失败: {e}")

    try:
        results["portfolio"] = test_nonlinear_constrained()
    except Exception as e:
        print(f"测试10失败: {e}")

    print("\n" + "=" * 60)
    print("所有测试完成!")
    print("=" * 60)
